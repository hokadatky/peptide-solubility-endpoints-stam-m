#!/usr/bin/env python3
"""Post-hoc, no-fit equal-probability ensemble diagnostic on design fold 0."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PROJECT_ROOT / "paper/general_peptide_solubility_scoping/analysis_distillation_rescue_run_20260813_v2"
V4_SRC = PROJECT_ROOT / "paper/peptide_solubility_review/analysis_addon/src"
sys.path.insert(0, str(V4_SRC))
from clustered_stats import paired_cluster_bootstrap, cluster_swap_permutation, metric_bundle  # noqa: E402


SEED = 20260813
REPLICATES = 10_000


def main() -> None:
    residual = pd.read_parquet(PRIVATE_ROOT / "esm35_residual_full_predictions.parquet")
    legacy = pd.read_parquet(PRIVATE_ROOT / "design_predictions.parquet")
    if not np.array_equal(residual.record_id.to_numpy(str), legacy.record_id.to_numpy(str)):
        raise RuntimeError("Prediction order mismatch")
    if set(residual.development_fold.unique()) != {0}:
        raise RuntimeError("Input is not design fold 0")
    y = residual.label.to_numpy(int)
    groups = residual.cluster_id.to_numpy(str)
    b1 = residual.B1_OFFSET.to_numpy(float)
    teacher = residual.PROTT5_TEACHER.to_numpy(float)
    legacy_hard = legacy.LEGACY_ESM35M_HARD_CONTROL.to_numpy(float)
    hard = 0.5 * (residual.ESM35_HARD_RESIDUAL.to_numpy(float) + legacy_hard)
    distilled = 0.5 * (residual.ESM35_DISTILLED_RESIDUAL.to_numpy(float) + legacy_hard)

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
        candidate_auc = roc_auc_score(y[indices], distilled[indices])
        denominator = teacher_auc - b1_auc
        if denominator <= 0:
            continue
        retention.append((candidate_auc - b1_auc) / denominator)
        target_margin.append(candidate_auc - (b1_auc + 0.8 * denominator))
    retention = np.asarray(retention, dtype=float)
    target_margin = np.asarray(target_margin, dtype=float)

    b1_auc = roc_auc_score(y, b1)
    teacher_auc = roc_auc_score(y, teacher)
    candidate_auc = roc_auc_score(y, distilled)
    point_retention = (candidate_auc - b1_auc) / (teacher_auc - b1_auc)
    point_margin = candidate_auc - (b1_auc + 0.8 * (teacher_auc - b1_auc))
    output = {
        "schema": "equal-probability-lightweight-ensemble-design-diagnostic-v1",
        "status": "POST_HOC_EXPLORATORY_DESIGN_PASS",
        "confirmation_fold_loaded": False,
        "selection_warning": "Equal weighting was evaluated after the component design-fold results were known; no weight search was performed.",
        "endpoint": "recombinant soluble-expression status in E. coli",
        "explicit_exclusion": "not direct aqueous solubility of a purified peptide",
        "composition": "0.5 * residual-distilled ESM-2 35M probability + 0.5 * frozen-embedding ESM-2 35M/B1 hard probability",
        "metrics": {
            "B1_OFFSET": metric_bundle(y, b1),
            "PROTT5_TEACHER": metric_bundle(y, teacher),
            "EQUAL_ENSEMBLE_HARD_CONTROL": metric_bundle(y, hard),
            "EQUAL_ENSEMBLE_DISTILLED": metric_bundle(y, distilled),
        },
        "distilled_minus_identical_hard_ensemble": contrasts,
        "distilled_ensemble_minus_b1_auc": vs_b1,
        "cluster_swap_auc_test": permutation,
        "teacher_gain_retention": {
            "point": float(point_retention),
            "ci95": [float(x) for x in np.percentile(retention, [2.5, 97.5])],
            "bootstrap_probability_at_least_0_8": float(np.mean(retention >= 0.8)),
            "valid_replicates": int(len(retention)),
        },
        "auc_margin_to_80pct_target": {
            "point": float(point_margin),
            "ci95": [float(x) for x in np.percentile(target_margin, [2.5, 97.5])],
            "bootstrap_probability_nonnegative": float(np.mean(target_margin >= 0.0)),
        },
        "promotion_decision": "Not promotable from this design-fold diagnostic; freeze as the sole candidate for a future locked evaluation.",
    }
    output_path = PLAN_ROOT / "EQUAL_ENSEMBLE_DESIGN_BOOTSTRAP_AGGREGATE_20260813.json"
    private_path = PRIVATE_ROOT / "equal_ensemble_design_predictions.parquet"
    for path in (output_path, private_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    pd.DataFrame({
        "record_id": residual.record_id,
        "cluster_id": groups,
        "development_fold": residual.development_fold,
        "label": y,
        "B1_OFFSET": b1,
        "PROTT5_TEACHER": teacher,
        "EQUAL_ENSEMBLE_HARD_CONTROL": hard,
        "EQUAL_ENSEMBLE_DISTILLED": distilled,
    }).to_parquet(private_path, index=False)
    output["private_prediction_path"] = str(private_path)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
