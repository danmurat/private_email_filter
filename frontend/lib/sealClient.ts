import type { SEALLibrary } from 'node-seal/implementation/seal';
import type { CipherText } from 'node-seal/implementation/cipher-text';
import type { KeyGenerator } from 'node-seal/implementation/key-generator';
import type { Encryptor } from 'node-seal/implementation/encryptor';

import { transformEmailToSvd } from '@/lib/svd';

export const SEAL_PROTOCOL_VERSION = '1';
export const SEAL_VECTOR_LENGTH = 150;
export const SEAL_POLY_MODULUS_DEGREE = 4096;
export const SEAL_COEFF_MODULUS_BITS = [40, 29, 40] as const;
export const SEAL_SCALE = 2 ** 29;

export type Prediction = {
  score: number;
  classification: 'ham' | 'spam';
};

export type SealFactory = () => Promise<SEALLibrary>;

type FetchLike = typeof fetch;

function dispose(value: { delete: () => void } | undefined): void {
  value?.delete();
}

/** The SVM uses 0 as boundary. Positives are spam, negatives ham */
export function classifyScore(score: number): Prediction['classification'] {
  return score >= 0 ? 'spam' : 'ham';
}

export function validateSvdVector(vector: ArrayLike<number>): void {
  if (vector.length !== SEAL_VECTOR_LENGTH) {
    throw new Error(
      `Encrypted vector has ${vector.length} values; expected ${SEAL_VECTOR_LENGTH}`,
    );
  }
  for (const value of Array.from(vector)) {
    if (!Number.isFinite(value)) {
      throw new Error('Encrypted vector contains a non-finite value');
    }
  }
}

function validateApiUrl(value: string): string {
  const apiUrl = value.trim().replace(/\/$/, '');
  if (!apiUrl) {
    throw new Error('Set EXPO_PUBLIC_API_URL before using encrypted prediction');
  }
  return apiUrl;
}

function getApiUrl(): string {
  const configured = process.env.EXPO_PUBLIC_API_URL;
  if (!configured) {
    throw new Error('Set EXPO_PUBLIC_API_URL before using encrypted prediction');
  }
  return validateApiUrl(configured);
}

async function loadSeal(): Promise<SEALLibrary> {
  // node-seal is intentionally lazy: the WASM runtime is browser-only and
  // should not be loaded by native Expo targets or during preprocessing.
  const module = await import('node-seal');
  return module.default();
}

function appendBinary(form: FormData, name: string, bytes: Uint8Array, filename: string): void {
  form.append(
    name,
    new Blob([bytes as unknown as BlobPart], { type: 'application/octet-stream' }),
    filename,
  );
}

/**
 * Encrypt and classify one 150-value SVD vector.
 *
 * The secret key is held only by this invocation.  The multipart form contains
 * parameters, Galois keys, ciphertext, and protocol metadata; it never
 * contains the public or secret key.
 */
export async function classifySvdVector(
  vector: ArrayLike<number>,
  options: {
    apiUrl?: string;
    fetchImpl?: FetchLike;
    sealFactory?: SealFactory;
  } = {},
): Promise<Prediction> {
  validateSvdVector(vector);
  const apiUrl = validateApiUrl(options.apiUrl ?? getApiUrl());
  const fetchImpl = options.fetchImpl ?? fetch;
  const seal = await (options.sealFactory ?? loadSeal)();

  let parameters: ReturnType<SEALLibrary['EncryptionParameters']> | undefined;
  let coefficientModulus: ReturnType<SEALLibrary['CoeffModulus']['Create']> | undefined;
  let context: ReturnType<SEALLibrary['Context']> | undefined;
  let keyGenerator: KeyGenerator | undefined;
  let secretKey: ReturnType<KeyGenerator['secretKey']> | undefined;
  let publicKey: ReturnType<KeyGenerator['createPublicKey']> | undefined;
  let galoisKeys: ReturnType<KeyGenerator['createGaloisKeys']> | undefined;
  let encoder: ReturnType<SEALLibrary['CKKSEncoder']> | undefined;
  let encryptor: Encryptor | undefined;
  let decryptor: ReturnType<SEALLibrary['Decryptor']> | undefined;
  let plainText: ReturnType<SEALLibrary['PlainText']> | undefined;
  let cipherText: CipherText | undefined;
  let resultCipherText: ReturnType<SEALLibrary['CipherText']> | undefined;
  let resultPlainText: ReturnType<SEALLibrary['PlainText']> | undefined;

  try {
    parameters = seal.EncryptionParameters(seal.SchemeType.ckks);
    parameters.setPolyModulusDegree(SEAL_POLY_MODULUS_DEGREE);
    coefficientModulus = seal.CoeffModulus.Create(
      SEAL_POLY_MODULUS_DEGREE,
      Int32Array.from(SEAL_COEFF_MODULUS_BITS),
    );
    parameters.setCoeffModulus(coefficientModulus);
    context = seal.Context(parameters, true, seal.SecurityLevel.none);
    if (!context.parametersSet()) {
      throw new Error('Unable to construct the CKKS parameter context');
    }

    keyGenerator = seal.KeyGenerator(context);
    secretKey = keyGenerator.secretKey();
    publicKey = keyGenerator.createPublicKey();
    galoisKeys = keyGenerator.createGaloisKeys();
    encoder = seal.CKKSEncoder(context);
    plainText = seal.PlainText();
    encoder.encode(Float64Array.from(vector), SEAL_SCALE, plainText);
    encryptor = seal.Encryptor(context, publicKey);
    cipherText = seal.CipherText();
    encryptor.encrypt(plainText, cipherText);

    const form = new FormData();
    appendBinary(form, 'parameters_file', parameters.saveArray(), 'parameters.bin');
    appendBinary(form, 'galois_keys_file', galoisKeys.saveArray(), 'galois-keys.bin');
    appendBinary(form, 'enc_email_file', cipherText.saveArray(), 'enc-email.bin');
    form.append('protocol_version', SEAL_PROTOCOL_VERSION);
    form.append('vector_length', String(SEAL_VECTOR_LENGTH));

    const response = await fetchImpl(`${apiUrl}/spamfilter/seal`, {
      method: 'POST',
      body: form,
    });
    if (!response.ok) {
      throw new Error(`Encrypted prediction failed (${response.status})`);
    }

    resultCipherText = seal.CipherText();
    resultCipherText.loadArray(context, new Uint8Array(await response.arrayBuffer()));
    decryptor = seal.Decryptor(context, secretKey);
    resultPlainText = seal.PlainText();
    decryptor.decrypt(resultCipherText, resultPlainText);
    const decoded = encoder.decode(resultPlainText);
    const score = decoded[0]; // result lies in [0] which contains the dot product sum - bias for the prediction
    if (!Number.isFinite(score)) {
      throw new Error('SEAL returned a non-finite prediction score');
    }
    return { score, classification: classifyScore(score) };
  } finally {
    // node-seal wraps C++ objects: JavaScript GC does not release their WASM
    // allocations, so every object created above is explicitly destroyed.
    dispose(resultPlainText);
    dispose(resultCipherText);
    dispose(decryptor);
    dispose(cipherText as { delete: () => void } | undefined);
    dispose(plainText);
    dispose(encryptor);
    dispose(encoder);
    dispose(galoisKeys);
    dispose(publicKey);
    dispose(secretKey);
    dispose(keyGenerator);
    dispose(context);
    dispose(coefficientModulus);
    dispose(parameters);
  }
}

export async function classifyEmail(
  text: string,
  options?: Parameters<typeof classifySvdVector>[1],
): Promise<Prediction> {
  return classifySvdVector(transformEmailToSvd(text), options);
}
