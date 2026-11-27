"""Fail-closed I/O, privacy, and numeric contracts for the add-on analyses."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
FORBIDDEN_FEATURES = {
    "record_id", "source_id", "id", "name", "name_len", "chemical_name",
    "sequence", "source_row", "row_index", "fold", "development_fold",
    "cluster_id", "external_cluster_id", "target", "label", "prediction",
    "historic_prediction", "previous_target", "prev_area", "post_outcome",
    "post_outcome_measurement", "target_rank", "error", "source_dataset",
    "source_split", "partition", "analysis_split",
}


def validate_unique_ids(frame: pd.DataFrame, id_col: str) -> None:
    if id_col not in frame.columns:
        raise ValueError(f"Missing identifier column: {id_col}")
    values = frame[id_col]
    if values.isna().any() or (values.astype(str).str.len() == 0).any():
        raise ValueError(f"Identifier column contains missing/empty values: {id_col}")
    if values.duplicated().any():
        raise ValueError(f"Identifier column is not unique: {id_col}")


def validate_binary_labels(frame: pd.DataFrame, label_col: str) -> None:
    if label_col not in frame.columns:
        raise ValueError(f"Missing binary label column: {label_col}")
    values = frame[label_col]
    if values.isna().any():
        raise ValueError("Binary labels contain missing values.")
    observed = set(values.astype(int).unique().tolist())
    if observed != {0, 1} or not np.allclose(values, values.astype(int)):
        raise ValueError(f"Binary labels must be exactly {{0, 1}}, observed={observed}")


def validate_regression_labels(frame: pd.DataFrame, label_col: str) -> None:
    if label_col not in frame.columns:
        raise ValueError(f"Missing regression label column: {label_col}")
    values = pd.to_numeric(frame[label_col], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Regression labels contain missing or non-finite values.")


def validate_canonical_sequence(sequence: str) -> str:
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("Sequence must be a non-empty string.")
    if any(character not in CANONICAL_ALPHABET for character in sequence):
        raise ValueError("Sequence contains a non-canonical amino-acid character.")
    return sequence


def one_to_one_join(left: pd.DataFrame, right: pd.DataFrame, key: str) -> pd.DataFrame:
    validate_unique_ids(left, key)
    validate_unique_ids(right, key)
    if set(left[key]) != set(right[key]):
        missing_left = len(set(left[key]) - set(right[key]))
        missing_right = len(set(right[key]) - set(left[key]))
        raise ValueError(f"One-to-one join key mismatch: left_only={missing_left}, right_only={missing_right}")
    merged = left.merge(right, on=key, how="left", validate="one_to_one", suffixes=("", "_right"))
    if len(merged) != len(left) or merged[key].duplicated().any():
        raise ValueError("One-to-one join changed row count or duplicated keys.")
    return merged


def assert_forbidden_columns_absent(feature_columns: list[str] | tuple[str, ...], forbidden_columns: set[str] | None = None) -> None:
    forbidden = {str(value).lower() for value in (forbidden_columns or FORBIDDEN_FEATURES)}
    bad = sorted({str(column).lower() for column in feature_columns} & forbidden)
    if bad:
        raise ValueError(f"Forbidden feature columns detected: {bad}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN or Infinity cannot be written to JSON.")
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(payload)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(safe, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)


def atomic_write_text(text: str, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)


def assert_finite_matrix(matrix: np.ndarray, label: str) -> None:
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite values in {label}.")
