#!/usr/bin/env python3
"""Aggregate-only paired cluster bootstrap for the design-fold rescue probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_ROOT = Path(__file__).resolve().parents[1]
PRIVATE = PROJECT_ROOT / "paper/general_peptide_solubility_scoping/analysis_distillation_rescue_run_20260813_v2/esm35_residual_full_predictions.parquet"
V4_SRC = PROJECT_ROOT / "paper/peptide_solubility_review/analysis_addon/src"
sys.path.insert(0, str(V4_SRC))
from clustered_stats import paired_cluster_bootstrap, cluster_swap_permutation  # noqa: E402


SEED = 20260813
REPLICATES = 10_000


def main() -> None:
    table = pd.read_parquet(PRIVATE)
    if set(table.development_fold.unique()) != {0}:
        raise RuntimeError("Bootstrap input is not design fold 0 only")
    y = table.label.to_numpy(int)
    groups = table.cluster_id.to_numpy(str)
    b1 = table.B1_OFFSET.to_numpy(float)
    teacher = table.PROTT5_TEACHER.to_numpy(float)
    hard = table.ESM35_HARD_RESIDUAL.to_numpy(float)
    distilled = table.ESM35_DISTILLED_RESIDUAL.to_numpy(float)
    contrasts = {
        metric: paired_cluster_bootstrap(
            y, hard, distilled, groups, metric=metric,
            replicates=REPLICATES, seed=SEED,
        )
        for metric in ("roc_auc", "auprc", "brier", "log_loss")
    }
    vs_b1 = paired_cluster_bootstrap(
        y, b1, distilled, groups, metric="roc_auc",
        replicates=REPLICATES, seed=SEED,
    )
    permutation = cluster_swap_permutation(
        y, hard, distilled, groups, metric="roc_auc", replicates=999, seed=SEED
    )

    unique = np.asarray(sorted(np.unique(groups).tolist()))
    row_indices = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(SEED)
    retention = []
    margin = []
    invalid = 0
    for _ in range(REPLICATES):
        sampled = rng.integers(0, len(unique), size=len(unique))
        indices = np.concatenate([row_indices[index] for index in sampled])
        if len(np.unique(y[indices])) < 2:
            invalid += 1
            continue
        b1_auc = roc_auc_score(y[indices], b1[indices])
        teacher_auc = roc_auc_score(y[indices], teacher[indices])
        student_auc = roc_auc_score(y[indices], distilled[indices])
        denominator = teacher_auc - b1_auc
        if denominator <= 0:
            invalid += 1
            continue
        retention.append((student_auc - b1_auc) / denominator)
        margin.append(student_auc - (b1_auc + 0.8 * denominator))
    retention = np.asarray(retention, dtype=float)
    margin = np.asarray(margin, dtype=float)
    point_b1 = roc_auc_score(y, b1)
    point_teacher = roc_auc_score(y, teacher)
    point_student = roc_auc_score(y, distilled)
    point_retention = (point_student - point_b1) / (point_teacher - point_b1)
    point_margin = point_student - (point_b1 + 0.8 * (point_teacher - point_b1))
    output = {
        "schema": "residual-distillation-design-bootstrap-v1",
        "status": "EXPLORATORY_DESIGN_ONLY",
        "confirmation_fold_loaded": False,
        "bootstrap_unit": "30pct_identity_cluster",
        "bootstrap_seed": SEED,
        "replicates_requested": REPLICATES,
        "distilled_minus_identical_hard_control": contrasts,
        "distilled_minus_b1_auc": vs_b1,
        "cluster_swap_auc_test": permutation,
        "teacher_gain_retention": {
            "point": float(point_retention),
            "ci95": [float(x) for x in np.percentile(retention, [2.5, 97.5])],
            "probability_at_least_0_8_under_bootstrap": float(np.mean(retention >= 0.8)),
            "valid_replicates": int(len(retention)),
            "invalid_replicates": int(invalid),
        },
        "auc_margin_to_80pct_target": {
            "point": float(point_margin),
            "ci95": [float(x) for x in np.percentile(margin, [2.5, 97.5])],
            "probability_nonnegative_under_bootstrap": float(np.mean(margin >= 0.0)),
        },
        "interpretation": "Design-fold uncertainty only; not a new untouched confirmation.",
    }
    path = PLAN_ROOT / "RESIDUAL_DESIGN_BOOTSTRAP_AGGREGATE_20260813.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
