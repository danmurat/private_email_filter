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
export type SealClientOptions = {
  apiUrl?: string;
  fetchImpl?: FetchLike;
  sealFactory?: SealFactory;
};

export type EncryptedSvdVector = {
  /** A readable representation of the encrypted vector for the demo UI. */
  ciphertext: string;
  classify: () => Promise<Prediction>;
  dispose: () => void;
};

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
 * Format the serialized ciphertext without exposing any key material.
 */
export function formatCiphertext(bytes: Uint8Array): string {
  return `0x${Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('')}`;
}

/**
 * Encrypt one 150-value SVD vector and return a session that can classify it later.
 *
 * The session keeps the secret key and SEAL context alive between the two user actions.
 * Calling dispose releases those WASM resources and should happen once the session is no
 * longer needed.
 */
export async function encryptSvdVector(
  vector: ArrayLike<number>,
  options: SealClientOptions = {},
): Promise<EncryptedSvdVector> {
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
  let disposed = false;
  let sessionReady = false;

  const disposeSession = (): void => {
    if (disposed) {
      return;
    }
    disposed = true;
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
  };

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

    const ciphertext = formatCiphertext(cipherText.saveArray());

    // These objects are only needed during encryption. Keep the secret key, context,
    // encoder, parameters, keys, and ciphertext for the later classification action.
    dispose(plainText);
    plainText = undefined;
    dispose(encryptor);
    encryptor = undefined;
    dispose(publicKey);
    publicKey = undefined;
    dispose(keyGenerator);
    keyGenerator = undefined;

    const classify = async (): Promise<Prediction> => {
      if (disposed || !context || !secretKey || !encoder || !cipherText) {
        throw new Error('Encrypted session is no longer available');
      }

      let resultCipherText: ReturnType<SEALLibrary['CipherText']> | undefined;
      let resultPlainText: ReturnType<SEALLibrary['PlainText']> | undefined;
      try {
        const form = new FormData();
        appendBinary(form, 'parameters_file', parameters!.saveArray(), 'parameters.bin');
        appendBinary(form, 'galois_keys_file', galoisKeys!.saveArray(), 'galois-keys.bin');
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
        const score = decoded[0]; // The result contains the dot product sum - bias.
        if (!Number.isFinite(score)) {
          throw new Error('SEAL returned a non-finite prediction score');
        }
        return { score, classification: classifyScore(score) };
      } finally {
        dispose(resultPlainText);
        dispose(resultCipherText);
        dispose(decryptor);
        decryptor = undefined;
      }
    };

    sessionReady = true;
    return { ciphertext, classify, dispose: disposeSession };
  } finally {
    if (!sessionReady) {
      disposeSession();
    }
  }
}

export async function encryptEmail(
  text: string,
  options?: SealClientOptions,
): Promise<EncryptedSvdVector> {
  return encryptSvdVector(transformEmailToSvd(text), options);
}

/**
 * Encrypt and classify one 150-value SVD vector in a single call for backwards compatibility.
 *
 * The secret key is held only by this invocation.  The multipart form contains
 * parameters, Galois keys, ciphertext, and protocol metadata; it never
 * contains the public or secret key.
 */
export async function classifySvdVector(
  vector: ArrayLike<number>,
  options: SealClientOptions = {},
): Promise<Prediction> {
  const encrypted = await encryptSvdVector(vector, options);
  try {
    return await encrypted.classify();
  } finally {
    encrypted.dispose();
  }
}

export async function classifyEmail(
  text: string,
  options?: SealClientOptions,
): Promise<Prediction> {
  return classifySvdVector(transformEmailToSvd(text), options);
}
