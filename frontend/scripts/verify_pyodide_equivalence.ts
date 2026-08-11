import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadPyodide } from 'pyodide';
import { transformEmail } from '@/lib/tfidf';
import { reduceTfidfVector, transformEmailToSvd } from '@/lib/svd';

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const projectDirectory = path.resolve(scriptDirectory, '..');
const tfidfPicklePath = path.join(projectDirectory, 'assets/models/tfidf.pkl');
const svdPicklePath = path.join(projectDirectory, 'assets/models/svd.pkl');

const EMAILS = [
  'Subject: Congratulations! You have won a FREE prize!!!',
  'Hi team, the software meeting is scheduled for Tuesday at 15:00.',
  'Please review the attached report and send your feedback.',
  'Café résumé naïve coöperate — Unicode and punctuation test.',
  'the and an x qzxvkjhgfds',
  '',
];

const ABSOLUTE_TOLERANCE = 1e-12;
const RELATIVE_TOLERANCE = 1e-10;

type PythonCase = {
  email: string;
  tfidf: number[];
  svd: number[];
};

type PythonReport = {
  sklearnVersion: string;
  cases: PythonCase[];
};

function compareVectors(name: string, expected: number[], actual: ArrayLike<number>) {
  if (expected.length !== actual.length) {
    throw new Error(
      `${name}: dimensions differ (Python ${expected.length}, TypeScript ${actual.length})`,
    );
  }

  let maximumAbsoluteDifference = 0;
  let mismatches = 0;

  for (let index = 0; index < expected.length; index += 1) {
    const difference = Math.abs(expected[index] - actual[index]);
    const scale = Math.max(Math.abs(expected[index]), Math.abs(actual[index]), 1);
    maximumAbsoluteDifference = Math.max(maximumAbsoluteDifference, difference);

    if (difference > ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * scale) {
      mismatches += 1;
    }
  }

  if (mismatches > 0) {
    throw new Error(
      `${name}: ${mismatches} values differ; maximum absolute difference was ${maximumAbsoluteDifference}`,
    );
  }

  return maximumAbsoluteDifference;
}

async function runPythonModels(): Promise<PythonReport> {
  const pyodide = await loadPyodide();
  await pyodide.loadPackage('scikit-learn');

  pyodide.FS.writeFile('/tmp/tfidf.pkl', new Uint8Array(fs.readFileSync(tfidfPicklePath)));
  pyodide.FS.writeFile('/tmp/svd.pkl', new Uint8Array(fs.readFileSync(svdPicklePath)));
  pyodide.globals.set('emails_json', JSON.stringify(EMAILS));

  const result = await pyodide.runPythonAsync(`
import json
import pickle
import sklearn

with open('/tmp/tfidf.pkl', 'rb') as file:
    tfidf = pickle.load(file)

with open('/tmp/svd.pkl', 'rb') as file:
    svd = pickle.load(file)

cases = []
for email in json.loads(emails_json):
    tfidf_result = tfidf.transform([email])
    svd_result = svd.transform(tfidf_result)
    cases.append({
        'email': email,
        'tfidf': tfidf_result.toarray()[0].tolist(),
        'svd': svd_result[0].tolist(),
    })

json.dumps({
    'sklearnVersion': sklearn.__version__,
    'cases': cases,
})
`);

  return JSON.parse(String(result)) as PythonReport;
}

async function main() {
  console.log('Starting Pyodide and loading scikit-learn...');
  const pythonReport = await runPythonModels();
  let maximumTfidfDifference = 0;
  let maximumSvdDifference = 0;

  for (const pythonCase of pythonReport.cases) {
    const typescriptTfidf = transformEmail(pythonCase.email);
    const typescriptSvd = transformEmailToSvd(pythonCase.email);
    const reducedFromTfidf = reduceTfidfVector(typescriptTfidf);

    maximumTfidfDifference = Math.max(
      maximumTfidfDifference,
      compareVectors(`TF-IDF (${JSON.stringify(pythonCase.email)})`, pythonCase.tfidf, typescriptTfidf),
    );
    maximumSvdDifference = Math.max(
      maximumSvdDifference,
      compareVectors(`SVD (${JSON.stringify(pythonCase.email)})`, pythonCase.svd, typescriptSvd),
      compareVectors(`SVD composition (${JSON.stringify(pythonCase.email)})`, pythonCase.svd, reducedFromTfidf),
    );
  }

  console.log(`Pyodide scikit-learn: ${pythonReport.sklearnVersion}`);
  console.log('Pickle training version: scikit-learn 1.5.0');
  console.log(`Compared ${pythonReport.cases.length} email fixtures.`);
  console.log(`Maximum TF-IDF absolute difference: ${maximumTfidfDifference}`);
  console.log(`Maximum SVD absolute difference: ${maximumSvdDifference}`);
  console.log('Python and TypeScript transforms are equivalent within tolerance.');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
