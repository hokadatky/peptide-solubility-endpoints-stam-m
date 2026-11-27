"""Frozen, transparent and order-aware sequence features."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .io_contracts import assert_finite_matrix, validate_canonical_sequence
except ImportError:  # pragma: no cover - direct execution path
    from io_contracts import assert_finite_matrix, validate_canonical_sequence


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def ordered_dimer_feature_names(alphabet: str = AA_ALPHABET) -> list[str]:
    return [f"dimer_{left}{right}" for left, right in product(alphabet, repeat=2)]


def ordered_dimer_features(sequences: list[str] | pd.Series, alphabet: str = AA_ALPHABET) -> tuple[np.ndarray, list[str]]:
    alphabet_index = {character: index for index, character in enumerate(alphabet)}
    output = np.zeros((len(sequences), len(alphabet) ** 2), dtype=np.float64)
    for row_index, sequence in enumerate(sequences):
        validate_canonical_sequence(str(sequence))
        denominator = max(len(sequence) - 1, 1)
        for position in range(max(len(sequence) - 1, 0)):
            left = alphabet_index[sequence[position]]
            right = alphabet_index[sequence[position + 1]]
            output[row_index, left * len(alphabet) + right] += 1.0 / denominator
    assert_finite_matrix(output, "ordered dimer features")
    return output, ordered_dimer_feature_names(alphabet)


def primary_feature_names(manifest_path: str | Path, group: str = "primary_aac_plus_physicochemical") -> list[str]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    groups = manifest.get("feature_groups", {})
    names = groups.get(group)
    if not isinstance(names, list) or not names:
        raise ValueError(f"Manifest feature group is missing or empty: {group}")
    if int(manifest.get("primary_feature_count", -1)) != 25 and group == "primary_aac_plus_physicochemical":
        raise ValueError("The frozen primary transparent feature count is not 25.")
    return [str(name) for name in names]


def transparent_matrix(features: pd.DataFrame, manifest_path: str | Path, group: str = "primary_aac_plus_physicochemical") -> tuple[np.ndarray, list[str]]:
    names = primary_feature_names(manifest_path, group)
    missing = [name for name in names if name not in features.columns]
    if missing:
        raise ValueError(f"Transparent feature columns are missing: {missing}")
    matrix = features[names].to_numpy(dtype=np.float64)
    assert_finite_matrix(matrix, "transparent feature matrix")
    return matrix, names


def build_b0_b1(features: pd.DataFrame, sequences: list[str] | pd.Series, manifest_path: str | Path, alphabet: str = AA_ALPHABET) -> dict[str, tuple[np.ndarray, list[str]]]:
    b0, b0_names = transparent_matrix(features, manifest_path)
    dimer, dimer_names = ordered_dimer_features(sequences, alphabet)
    b1 = np.hstack([b0, dimer])
    assert_finite_matrix(b1, "B1 ordered baseline")
    return {
        "B0": (b0, b0_names),
        "B1": (b1, b0_names + dimer_names),
    }
