import { transformEmail } from '@/lib/tfidf';

describe('transformEmail', () => {
  it('returns the fitted feature dimension', () => {
    const vector = transformEmail('This is a short email about a meeting.');

    expect(vector).toHaveLength(3020);
  });

  it('ignores stop words and unknown terms', () => {
    const vector = transformEmail('the qzxvkjhgfds');

    expect(Array.from(vector).every((value) => value === 0)).toBe(true);
  });

  it('counts repeated terms before applying IDF and normalization', () => {
    const vector = transformEmail('software software');
    const singleTermVector = transformEmail('software');

    expect(vector).toEqual(singleTermVector);
  });

  it('returns a zero vector when no fitted terms are present', () => {
    const vector = transformEmail('x ! @ #');

    expect(Array.from(vector).every((value) => value === 0)).toBe(true);
  });
});
