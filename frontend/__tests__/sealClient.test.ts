import {
  classifyScore,
  encryptSvdVector,
  SEAL_VECTOR_LENGTH,
  validateSvdVector,
} from '@/lib/sealClient';

describe('seal client helpers', () => {
  it('classifies positive and negative SVM scores by sign', () => {
    expect(classifyScore(0.001)).toBe('spam');
    expect(classifyScore(-0.001)).toBe('ham');
    expect(classifyScore(0)).toBe('spam');
  });

  it('validates the reduced vector dimension and values', () => {
    expect(() => validateSvdVector(new Float64Array(SEAL_VECTOR_LENGTH - 1))).toThrow(
      'expected 150',
    );
    const vector = new Float64Array(SEAL_VECTOR_LENGTH);
    vector[3] = Number.NaN;
    expect(() => validateSvdVector(vector)).toThrow('non-finite');
  });

  it('separates encryption from classification and cleans up key material', async () => {
    const deleted = new Map<string, number>();
    const object = <T extends object>(name: string, extra: T = {} as T) => {
      deleted.set(name, 0);
      return {
        ...extra,
        delete: () => deleted.set(name, (deleted.get(name) ?? 0) + 1),
      };
    };
    const plain = () => object('plain');
    const cipher = () => object('cipher', {
      saveArray: () => new Uint8Array([1, 2, 3]),
      loadArray: () => undefined,
    });
    const fakeSeal = {
      SchemeType: { ckks: 1 },
      SecurityLevel: { none: 0 },
      EncryptionParameters: () => object('parameters', {
        setPolyModulusDegree: () => undefined,
        setCoeffModulus: () => undefined,
        saveArray: () => new Uint8Array([1]),
        parametersSet: () => true,
      }),
      CoeffModulus: { Create: () => object('coefficients') },
      Context: () => object('context', { parametersSet: () => true }),
      KeyGenerator: () => object('key-generator', {
        secretKey: () => object('secret-key', { saveArray: () => new Uint8Array([9]) }),
        createPublicKey: () => object('public-key'),
        createGaloisKeys: () => object('galois-keys', {
          saveArray: () => new Uint8Array([4, 5]),
        }),
      }),
      CKKSEncoder: () => object('encoder', {
        encode: (_values: ArrayLike<number>, _scale: number, destination: object) => destination,
        decode: () => new Float64Array([1]),
      }),
      PlainText: plain,
      CipherText: cipher,
      Encryptor: () => object('encryptor', { encrypt: () => undefined }),
      Decryptor: () => object('decryptor', { decrypt: () => undefined }),
    };
    let requestBody: FormData | undefined;
    let requestCount = 0;
    const fetchImpl = async (_url: RequestInfo | URL, init?: RequestInit) => {
      requestCount += 1;
      requestBody = init?.body as FormData;
      return {
        ok: true,
        status: 200,
        arrayBuffer: async () => new ArrayBuffer(0),
      } as Response;
    };

    const encrypted = await encryptSvdVector(new Float64Array(SEAL_VECTOR_LENGTH), {
      apiUrl: 'http://localhost:8000',
      fetchImpl,
      sealFactory: async () => fakeSeal as never,
    });

    expect(requestCount).toBe(0);
    expect(encrypted.ciphertext).toBe('0x010203');
    await expect(encrypted.classify()).resolves.toEqual({ score: 1, classification: 'spam' });
    encrypted.dispose();
    expect(requestCount).toBe(1);

    for (const field of [
      'enc_email_file',
      'galois_keys_file',
      'parameters_file',
      'protocol_version',
      'vector_length',
    ]) {
      expect(requestBody!.has(field)).toBe(true);
    }
    expect(requestBody!.has('secret_key')).toBe(false);
    expect(deleted.get('secret-key')).toBe(1);
    expect(deleted.get('context')).toBe(1);
    expect(deleted.get('parameters')).toBe(1);
  });
});
