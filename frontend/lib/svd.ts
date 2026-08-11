import fittedModel from '@/assets/models/svd.json';
import { transformEmail } from '@/lib/tfidf';

export type SvdModel = {
  nComponents: number;
  nFeaturesIn: number;
  components: number[];
};

const model = fittedModel as unknown as SvdModel;

/**
 * Applies a fitted TruncatedSVD model to one TF-IDF vector.
 *
 * The components are stored row-major: component 0's 3020 values, then
 * component 1's 3020 values, and so on. This is equivalent to
 * sklearn's `svd.transform(tfidf_vector)` for the fitted model.
 */
export function reduceTfidfVector(
  tfidfVector: ArrayLike<number>,
  svdModel: SvdModel = model,
): Float64Array {
  if (tfidfVector.length !== svdModel.nFeaturesIn) {
    throw new Error(
      `TF-IDF vector has ${tfidfVector.length} features; expected ${svdModel.nFeaturesIn}`,
    );
  }

  const expectedComponentValues = svdModel.nComponents * svdModel.nFeaturesIn;
  if (svdModel.components.length !== expectedComponentValues) {
    throw new Error('SVD component matrix has an invalid size');
  }

  const reducedVector = new Float64Array(svdModel.nComponents);

  for (let component = 0; component < svdModel.nComponents; component += 1) {
    const componentOffset = component * svdModel.nFeaturesIn;
    let sum = 0;

    for (let feature = 0; feature < svdModel.nFeaturesIn; feature += 1) {
      sum += tfidfVector[feature] * svdModel.components[componentOffset + feature];
    }

    reducedVector[component] = sum;
  }

  return reducedVector;
}

/** Transforms an email through the fitted TF-IDF and SVD models. */
export function transformEmailToSvd(text: string): Float64Array {
  return reduceTfidfVector(transformEmail(text));
}

