"""Validate every frozen V4 SoluProt split without fitting any model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

try:
    from .split_contracts import (
        SPLIT_CONTRACT_VERSION,
        balanced_group_folds,
        fixed_group_folds,
        fold_hash,
        split_diagnostics,
    )
except ImportError:  # pragma: no cover - direct execution path
    from split_contracts import (
        SPLIT_CONTRACT_VERSION,
        balanced_group_folds,
        fixed_group_folds,
        fold_hash,
        split_diagnostics,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_record(
    name: str,
    y: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    n_splits: int,
    min_class_groups_per_fold: int,
) -> dict[str, object]:
    diagnostics = split_diagnostics(
        y,
        groups,
        folds,
        n_splits,
        min_class_groups_per_fold=min_class_groups_per_fold,
    )
    if not diagnostics["all_hard_gates_passed"]:
        raise RuntimeError(f"V4 split hard gate failed: {name}")
    print(f"[split-gate] PASS {name}", flush=True)
    return {
        "name": name,
        "fold_hash": fold_hash(groups, folds),
        "diagnostics": diagnostics,
    }


def validate_all(
    input_path: Path,
    *,
    global_seed: int = 20260809,
    stability_seeds: tuple[int, ...] = (
        20260809,
        20260810,
        20260811,
        20260812,
        20260813,
    ),
    min_class_groups_per_fold: int = 20,
) -> dict[str, object]:
    data = pd.read_csv(input_path)
    required = {"label", "cluster_id", "development_fold", "partition"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing split-validation columns: {sorted(missing)}")
    y = data["label"].to_numpy(dtype=int)
    groups = data["cluster_id"].astype(str).to_numpy()
    supplied_outer = fixed_group_folds(
        y, groups, data["development_fold"].to_numpy(), 7
    )
    records: list[dict[str, object]] = []
    records.append(
        _checked_record(
            "primary_outer_fixed",
            y,
            groups,
            supplied_outer,
            7,
            min_class_groups_per_fold,
        )
    )

    for outer_fold in range(7):
        train = supplied_outer != outer_fold
        inner = balanced_group_folds(
            y[train],
            groups[train],
            5,
            global_seed + outer_fold,
            min_class_groups_per_fold=min_class_groups_per_fold,
        )
        records.append(
            _checked_record(
                f"primary_inner_outer_{outer_fold}",
                y[train],
                groups[train],
                inner,
                5,
                min_class_groups_per_fold,
            )
        )

    stability_hashes: list[str] = []
    for seed in stability_seeds:
        outer = balanced_group_folds(
            y,
            groups,
            7,
            seed,
            min_class_groups_per_fold=min_class_groups_per_fold,
        )
        outer_record = _checked_record(
            f"stability_{seed}_outer",
            y,
            groups,
            outer,
            7,
            min_class_groups_per_fold,
        )
        stability_hashes.append(str(outer_record["fold_hash"]))
        records.append(outer_record)
        for outer_fold in range(7):
            train = outer != outer_fold
            inner = balanced_group_folds(
                y[train],
                groups[train],
                5,
                seed + outer_fold,
                min_class_groups_per_fold=min_class_groups_per_fold,
            )
            records.append(
                _checked_record(
                    f"stability_{seed}_inner_outer_{outer_fold}",
                    y[train],
                    groups[train],
                    inner,
                    5,
                    min_class_groups_per_fold,
                )
            )
    if len(set(stability_hashes)) != len(stability_hashes):
        raise RuntimeError("Two frozen stability seeds produced the same canonical partition.")

    fit = data["partition"].astype(str).to_numpy() == "fit"
    fit_inner = balanced_group_folds(
        y[fit],
        groups[fit],
        5,
        global_seed,
        min_class_groups_per_fold=min_class_groups_per_fold,
    )
    records.append(
        _checked_record(
            "external_procedure_fit_inner",
            y[fit],
            groups[fit],
            fit_inner,
            5,
            min_class_groups_per_fold,
        )
    )

    partition_groups = {
        name: set(data.loc[data["partition"].astype(str) == name, "cluster_id"].astype(str))
        for name in ("fit", "model_selection", "calibration")
    }
    for left, right in (
        ("fit", "model_selection"),
        ("fit", "calibration"),
        ("model_selection", "calibration"),
    ):
        if partition_groups[left] & partition_groups[right]:
            raise RuntimeError(f"Development partition group overlap: {left} vs {right}")

    return {
        "status": "PASS",
        "models_fitted": 0,
        "input": {
            "path": str(input_path.resolve()),
            "sha256": _sha256(input_path),
            "n_rows": int(len(data)),
            "n_groups": int(data["cluster_id"].nunique()),
        },
        "contract": {
            "split_contract_version": SPLIT_CONTRACT_VERSION,
            "sklearn_version": sklearn.__version__,
            "global_seed": int(global_seed),
            "stability_seeds": [int(value) for value in stability_seeds],
            "min_class_groups_per_fold": int(min_class_groups_per_fold),
        },
        "n_validated_assignments": int(len(records)),
        "development_partition_group_overlap": 0,
        "records": records,
    }


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite split audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = validate_all(args.input)
    if args.output is not None:
        _write_atomic(args.output, payload)
        print(f"[split-gate] wrote {args.output}", flush=True)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "models_fitted": payload["models_fitted"],
                "n_validated_assignments": payload["n_validated_assignments"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
