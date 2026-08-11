import fittedModel from '@/assets/models/tfidf.json';

export type TfidfModel = {
  nFeatures: number;
  lowercase: boolean;
  analyzer: 'word';
  ngramRange: [number, number];
  norm: 'l2';
  useIdf: boolean;
  smoothIdf: boolean;
  sublinearTf: boolean;
  tokenPattern: string;
  vocabulary: Record<string, number>;
  idf: number[];
  stopWords: string[];
};

const model = fittedModel as unknown as TfidfModel;

// Python's `(?u)\\b\\w\\w+\\b` matches Unicode word characters with a
// minimum length of two. This is the closest browser-native equivalent.
const TOKEN_PATTERN = /[\p{L}\p{N}_]{2,}/gu;

export function transformEmail(text: string, tfidfModel: TfidfModel = model): Float64Array {
  if (tfidfModel.analyzer !== 'word') {
    throw new Error('Only word-level TF-IDF models are supported');
  }
  if (tfidfModel.ngramRange[0] !== 1 || tfidfModel.ngramRange[1] !== 1) {
    throw new Error('Only unigram TF-IDF models are supported');
  }
  if (tfidfModel.norm !== 'l2' || !tfidfModel.useIdf || tfidfModel.sublinearTf) {
    throw new Error('Unsupported TF-IDF configuration');
  }
  if (tfidfModel.idf.length !== tfidfModel.nFeatures) {
    throw new Error('TF-IDF model IDF length does not match nFeatures');
  }

  const normalizedText = tfidfModel.lowercase ? text.toLowerCase() : text;
  const stopWords = new Set(tfidfModel.stopWords);
  const counts = new Map<number, number>();

  for (const token of normalizedText.match(TOKEN_PATTERN) ?? []) {
    if (stopWords.has(token)) {
      continue;
    }

    const featureIndex = tfidfModel.vocabulary[token];
    if (featureIndex !== undefined) {
      counts.set(featureIndex, (counts.get(featureIndex) ?? 0) + 1);
    }
  }

  const vector = new Float64Array(tfidfModel.nFeatures);
  let squaredNorm = 0;

  for (const [featureIndex, count] of counts) {
    const weightedValue = count * tfidfModel.idf[featureIndex];
    vector[featureIndex] = weightedValue;
    squaredNorm += weightedValue * weightedValue;
  }

  const norm = Math.sqrt(squaredNorm);
  if (norm > 0) {
    for (let index = 0; index < vector.length; index += 1) {
      vector[index] /= norm;
    }
  }

  return vector;
}
