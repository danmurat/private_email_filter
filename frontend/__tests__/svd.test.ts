import { reduceTfidfVector, transformEmailToSvd } from '@/lib/svd';

describe('reduceTfidfVector', () => {
  it('multiplies a TF-IDF vector by the fitted component matrix', () => {
    const vector = reduceTfidfVector(
      new Float64Array([1, 2, 3]),
      {
        nComponents: 2,
        nFeaturesIn: 3,
        components: [1, 2, 3, 4, 5, 6],
      },
    );

    expect(Array.from(vector)).toEqual([14, 32]);
  });

  it('rejects vectors with the wrong feature dimension', () => {
    expect(() => reduceTfidfVector(new Float64Array(3))).toThrow(
      'expected 3020',
    );
  });

  it('rejects malformed component matrices', () => {
    expect(() =>
      reduceTfidfVector(new Float64Array(3), {
        nComponents: 2,
        nFeaturesIn: 3,
        components: [1, 2, 3],
      }),
    ).toThrow('invalid size');
  });

  it('reduces the fitted email pipeline to 150 features', () => {
    const vector = transformEmailToSvd('software software');

    expect(vector).toHaveLength(150);
    expect(Array.from(vector).some((value) => value !== 0)).toBe(true);
  });
});
