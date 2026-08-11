#!/usr/bin/env python3
"""Export a fitted scikit-learn TruncatedSVD model for the web client."""

from __future__ import annotations

import argparse
import json
import pickle
import struct
from pathlib import Path


class _TruncatedSVD:
    pass


class _DType:
    def __init__(self, *args):
        self.args = args

    def __setstate__(self, state):
        self.state = state


class _NDArray:
    def __setstate__(self, state):
        version, shape, dtype, fortran_order, raw_data = state
        if version != 1 or dtype.args[0] != "f8" or fortran_order:
            raise ValueError("Unsupported components_ array format")

        count = 1
        for dimension in shape:
            count *= dimension

        self.shape = shape
        self.data = struct.unpack("<" + ("d" * count), raw_data)


def _reconstruct(*args):
    return _NDArray()


def _frombuffer(data, dtype, shape, order):
    count = 1
    for dimension in shape:
        count *= dimension

    if dtype.args[0] != "f8":
        raise ValueError(f"Unexpected array dtype: {dtype.args[0]}")

    return struct.unpack("<" + ("d" * count), bytes(data))


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
            "sklearn.decomposition._truncated_svd.TruncatedSVD": _TruncatedSVD,
            "numpy.core.multiarray._reconstruct": _reconstruct,
            "numpy.ndarray": _NDArray,
            "numpy.dtype": _DType,
            "numpy.core.numeric._frombuffer": _frombuffer,
            "numpy.core.multiarray.scalar": _scalar,
            "numpy.float64": float,
        }

        if qualified_name not in allowed:
            raise pickle.UnpicklingError(f"Unsupported pickle global: {qualified_name}")

        return allowed[qualified_name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    with args.pickle_path.open("rb") as file:
        svd = _RestrictedUnpickler(file).load()

    components = svd.components_
    n_components, n_features = components.shape

    if n_components != svd.n_components or n_features != svd.n_features_in_:
        raise ValueError("SVD component shape does not match fitted metadata")

    model = {
        "sklearnVersion": svd._sklearn_version,
        "nComponents": n_components,
        "nFeaturesIn": n_features,
        "components": components.data,
    }

    args.output_path.write_text(
        json.dumps(model, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
