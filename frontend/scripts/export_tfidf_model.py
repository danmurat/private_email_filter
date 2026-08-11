#!/usr/bin/env python3
"""Export the fitted scikit-learn TF-IDF state for the web client.

This intentionally does not import scikit-learn or NumPy. The frontend does
not need either package at runtime; it only needs the fitted vocabulary, IDF
values, and the exact stop-word list used by the vectorizer.
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import struct
from pathlib import Path


class _Estimator:
    pass


class _DType:
    def __init__(self, *args):
        self.args = args

    def __setstate__(self, state):
        self.state = state


class _Array(list):
    pass


def _frombuffer(data, dtype, shape, order):
    count = 1
    for dimension in shape:
        count *= dimension

    if dtype.args[0] != "f8":
        raise ValueError(f"Unexpected array dtype: {dtype.args[0]}")

    return _Array(struct.unpack("<" + ("d" * count), bytes(data)))


def _scalar(dtype, data):
    if dtype.args[0] == "i8":
        return int.from_bytes(bytes(data), byteorder="little", signed=True)
    if dtype.args[0] == "f8":
        return struct.unpack("<d", bytes(data))[0]
    raise ValueError(f"Unexpected scalar dtype: {dtype.args[0]}")


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        qualified_name = f"{module}.{name}"
        allowed = {
            "sklearn.feature_extraction.text.TfidfVectorizer": _Estimator,
            "sklearn.feature_extraction.text.TfidfTransformer": _Estimator,
            "numpy.core.numeric._frombuffer": _frombuffer,
            "numpy.dtype": _DType,
            "numpy.core.multiarray.scalar": _scalar,
            "numpy.float64": float,
        }

        if qualified_name not in allowed:
            raise pickle.UnpicklingError(f"Unsupported pickle global: {qualified_name}")

        return allowed[qualified_name]


def _load_stop_words(source_path: Path) -> list[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "ENGLISH_STOP_WORDS" for target in node.targets):
            continue
        values = node.value.args[0]
        return sorted(ast.literal_eval(values))

    raise ValueError("Could not find ENGLISH_STOP_WORDS in the supplied source")


def _load_vectorizer(path: Path):
    with path.open("rb") as file:
        return _RestrictedUnpickler(file).load()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_path", type=Path)
    parser.add_argument("stop_words_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    vectorizer = _load_vectorizer(args.pickle_path)
    transformer = vectorizer._tfidf
    vocabulary = vectorizer.vocabulary_
    idf = transformer.idf_

    if len(vocabulary) != len(idf):
        raise ValueError("Vocabulary and IDF lengths do not match")
    if sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise ValueError("Vocabulary indices are not contiguous")

    model = {
        "sklearnVersion": vectorizer._sklearn_version,
        "nFeatures": len(vocabulary),
        "tokenPattern": vectorizer.token_pattern,
        "lowercase": vectorizer.lowercase,
        "analyzer": vectorizer.analyzer,
        "ngramRange": list(vectorizer.ngram_range),
        "norm": vectorizer.norm,
        "useIdf": vectorizer.use_idf,
        "smoothIdf": vectorizer.smooth_idf,
        "sublinearTf": vectorizer.sublinear_tf,
        "vocabulary": vocabulary,
        "idf": idf,
        "stopWords": _load_stop_words(args.stop_words_path),
    }

    args.output_path.write_text(
        json.dumps(model, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
