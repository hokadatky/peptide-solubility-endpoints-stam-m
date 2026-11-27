#!/usr/bin/env python3
"""Final aggregate bootstrap for the correct-pooling design-fold ensemble."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_ROOT = Path(__file__).resolve().parents[1]
PRIVATE = PROJECT_ROOT / "paper/general_peptide_solubility_scoping/analysis_distillation_rescue_run_20260813_v2/correct_equal_ensemble_design_predictions.parquet"
V4_SRC = PROJECT_ROOT / "paper/peptide_solubility_review/analysis_addon/src"
sys.path.insert(0, str(V4_SRC))
from clustered_stats import paired_cluster_bootstrap, cluster_swap_permutation  # noqa: E402


SEED = 20260813
REPLICATES = 10_000


def main() -> None:
    table = pd.read_parquet(PRIVATE)
    if set(table.development_fold.unique()) != {0}:
        raise RuntimeError("Not design fold 0")
    y = table.label.to_numpy(int)
    groups = table.cluster_id.to_numpy(str)
    b1 = table.B1_OFFSET.to_numpy(float)
    teacher = table.PROTT5_TEACHER.to_numpy(float)
    hard = table.CORRECT_EQUAL_ENSEMBLE_HARD_CONTROL.to_numpy(float)
    distilled = table.CORRECT_EQUAL_ENSEMBLE_DISTILLED.to_numpy(float)
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
    rows = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(SEED)
    retention = []
    target_margin = []
    for _ in range(REPLICATES):
        sampled = rng.integers(0, len(unique), size=len(unique))
        indices = np.concatenate([rows[index] for index in sampled])
        if len(np.unique(y[indices])) < 2:
            continue
        b1_auc = roc_auc_score(y[indices], b1[indices])
        teacher_auc = roc_auc_score(y[indices], teacher[indices])
        student_auc = roc_auc_score(y[indices], distilled[indices])
        denominator = teacher_auc - b1_auc
        if denominator <= 0:
            continue
        retention.append((student_auc - b1_auc) / denominator)
        target_margin.append(student_auc - (b1_auc + 0.8 * denominator))
    retention = np.asarray(retention, dtype=float)
    target_margin = np.asarray(target_margin, dtype=float)
    b1_auc = roc_auc_score(y, b1)
    teacher_auc = roc_auc_score(y, teacher)
    student_auc = roc_auc_score(y, distilled)
    output = {
        "schema": "correct-pooling-equal-ensemble-bootstrap-v1",
        "status": "EXPLORATORY_DESIGN_PASS_NOT_PROMOTED",
        "confirmation_fold_loaded": False,
        "bootstrap_unit": "30pct_identity_cluster",
        "replicates": REPLICATES,
        "seed": SEED,
        "distilled_minus_identical_hard_ensemble": contrasts,
        "distilled_ensemble_minus_b1_auc": vs_b1,
        "cluster_swap_auc_test": permutation,
        "teacher_gain_retention": {
            "point": float((student_auc - b1_auc) / (teacher_auc - b1_auc)),
            "ci95": [float(x) for x in np.percentile(retention, [2.5, 97.5])],
            "bootstrap_probability_at_least_0_8": float(np.mean(retention >= 0.8)),
            "valid_replicates": int(len(retention)),
        },
        "auc_margin_to_80pct_target": {
            "point": float(student_auc - (b1_auc + 0.8 * (teacher_auc - b1_auc))),
            "ci95": [float(x) for x in np.percentile(target_margin, [2.5, 97.5])],
            "bootstrap_probability_nonnegative": float(np.mean(target_margin >= 0.0)),
        },
        "claim_ceiling": "Correct-pooling design-fold feasibility. Candidate must be frozen and tested on a genuinely unused cohort before manuscript or app promotion.",
    }
    path = PLAN_ROOT / "CORRECT_EQUAL_ENSEMBLE_BOOTSTRAP_AGGREGATE_20260813.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
