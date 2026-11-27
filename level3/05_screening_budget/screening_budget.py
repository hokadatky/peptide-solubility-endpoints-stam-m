#!/usr/bin/env python3
"""Screening-budget re-expression of the sealed retrospective external evaluation.

No model is fitted, refitted, reselected or recalibrated here. The row-level external
predictions and their calibration maps were frozen in the parent run
(paper/stam_nesg_external_20261016) on 2026-10-16; this script consumes them read-only and
applies a different summary statistic: the number of soluble outcomes and the precision within
the k highest-scoring records, for a budget grid fixed in PLAN.md before execution.

Run:
    python -I screening_budget.py

Every quantity defined in PLAN.md section 4 is written out, favorable or not. The script aborts
rather than producing partial output if any precondition in PLAN.md fails.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
PARENT = HERE.parent / "stam_nesg_external_20261016"
PRIMARY_NPZ = PARENT / "private" / "external_predictions.npz"
SEALED_NPZ = PARENT / "exploratory_improvement" / "predictions_sealed.npz"
PRIMARY_JSON = PARENT / "results" / "external_retrospective.json"
EXPLORATORY_JSON = PARENT / "exploratory_improvement" / "results.json"

# PLAN.md section 4: fixed before execution, not to be changed afterwards.
K_GRID = (24, 48, 96, 192, 384, 1550)
REPLICATES = 10_000
SEED = 20261016
POOL_PREVALENCE = 0.5

ARMS = (
    # (arm key in results, column prefix, source file)
    ("B1_ORDER_LOGIT", "B1_ORDER_LOGIT", "primary"),
    ("B1_ORDER_PLUS_PROTT5_PLS_LOGIT", "B1_ORDER_PLUS_PROTT5_PLS_LOGIT", "primary"),
    ("B1_LOGIT", "B1_LOGIT", "sealed"),
    ("FULL_LOGIT", "FULL_LOGIT", "sealed"),
    ("PCA_LOGIT", "PCA_LOGIT", "sealed"),
    ("B1_HGB", "B1_HGB", "sealed"),
    ("PCA_HGB", "PCA_HGB", "sealed"),
)

CONTRASTS = (
    ("primary", "B1_ORDER_LOGIT", "B1_ORDER_PLUS_PROTT5_PLS_LOGIT",
     "prespecified in the parent run"),
    ("exploratory_full_embedding", "B1_LOGIT", "FULL_LOGIT",
     "exploratory in the parent run"),
    ("exploratory_pca_logistic", "B1_LOGIT", "PCA_LOGIT",
     "exploratory in the parent run"),
    ("exploratory_pca_hgb", "B1_HGB", "PCA_HGB",
     "exploratory in the parent run"),
)


def die(message: str) -> None:
    raise SystemExit(f"error: {message}")


def sha256(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def top_k_successes(y: np.ndarray, score: np.ndarray, k: int) -> int:
    """Soluble outcomes among the k highest scores.

    Ranking is deterministic: descending score, ties broken by ascending row index, which is the
    stable order of ``np.argsort`` on the negated score.
    """
    if len(y) < k:
        die(f"cannot take the top {k} of {len(y)} records")
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    return int(np.asarray(y)[order[:k]].sum())


def precision_gain(y: np.ndarray, ref: np.ndarray, cand: np.ndarray, k: int) -> float:
    return (top_k_successes(y, cand, k) - top_k_successes(y, ref, k)) / k


def auc_gain(y: np.ndarray, ref: np.ndarray, cand: np.ndarray) -> float:
    return float(roc_auc_score(y, cand) - roc_auc_score(y, ref))


def paired_cluster_bootstrap(y, ref, cand, groups, gain_fn, replicates=REPLICATES, seed=SEED):
    """Replicates vendor/analysis_addon/src/clustered_stats.py::paired_cluster_bootstrap.

    Same construction, same validity rule, same percentile intervals; the metric is supplied as a
    callable so that the identical estimator serves ROC-AUC (for the reproduction check) and
    top-k precision.
    """
    y = np.asarray(y).astype(int)
    ref = np.asarray(ref, dtype=float)
    cand = np.asarray(cand, dtype=float)
    groups = np.asarray(groups).astype(str)
    if not (len(y) == len(ref) == len(cand) == len(groups)):
        die("bootstrap inputs must have equal length")
    unique_groups = np.asarray(sorted(np.unique(groups).tolist()), dtype=str)
    row_indices = [np.flatnonzero(groups == group) for group in unique_groups]
    point = gain_fn(y, ref, cand)
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    invalid = 0
    sizes: list[int] = []
    for _ in range(int(replicates)):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        indices = np.concatenate([row_indices[index] for index in sampled])
        if len(np.unique(y[indices])) < 2:
            invalid += 1
            continue
        sizes.append(len(indices))
        values.append(gain_fn(y[indices], ref[indices], cand[indices]))
    if not values:
        die("all cluster bootstrap replicates were invalid")
    array = np.asarray(values, dtype=float)
    return {
        "point_gain": float(point),
        "ci95": [float(v) for v in np.percentile(array, [2.5, 97.5])],
        "ci90": [float(v) for v in np.percentile(array, [5, 95])],
        "bootstrap_mean": float(array.mean()),
        "bootstrap_sd": float(array.std(ddof=1)),
        "n_groups": int(len(unique_groups)),
        "n_replicates_requested": int(replicates),
        "n_replicates_valid": int(len(array)),
        "invalid_fraction": invalid / int(replicates),
        "resampled_rows_min": int(min(sizes)),
        "resampled_rows_median": int(np.median(sizes)),
        "resampled_rows_max": int(max(sizes)),
    }


def calibration_intercept_slope(y: np.ndarray, score: np.ndarray) -> dict:
    """Identical to the parent run's definition, kept here so the diagnostic is self-contained."""
    y = np.asarray(y).astype(int)
    score = np.clip(np.asarray(score, dtype=float), 1e-8, 1 - 1e-8)
    logit = np.log(score / (1.0 - score)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    model.fit(logit, y)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}


def main() -> int:
    for path in (PRIMARY_NPZ, SEALED_NPZ, PRIMARY_JSON, EXPLORATORY_JSON):
        if not path.exists():
            die(f"missing input {path}")

    primary = np.load(PRIMARY_NPZ, allow_pickle=True)
    sealed = np.load(SEALED_NPZ, allow_pickle=True)

    # PLAN.md section 3: the join is accepted only on an identical, identically ordered record id.
    pid = np.asarray([str(v) for v in primary["record_id"]])
    sid = np.asarray([str(v) for v in sealed["record_id"]])
    if pid.shape != sid.shape or not np.array_equal(pid, sid):
        die("record_id differs between the primary and sealed prediction files; join refused")

    y = np.asarray(primary["label"]).astype(int)
    groups = np.asarray([str(v) for v in primary["external_cluster_id"]])
    sealed_groups = np.asarray([str(v) for v in sealed["external_cluster_id"]])
    if not np.array_equal(groups, sealed_groups):
        die("external_cluster_id differs between the two prediction files; join refused")

    n_rows = int(len(y))
    n_pos = int(y.sum())
    if n_rows != 3100 or n_pos != 1550:
        die(f"unexpected external pool: {n_rows} rows, {n_pos} positives")
    prevalence = n_pos / n_rows
    if abs(prevalence - POOL_PREVALENCE) > 1e-12:
        die(f"pool prevalence {prevalence} is not the documented {POOL_PREVALENCE}")

    scores: dict[str, dict[str, np.ndarray]] = {}
    for arm, prefix, source in ARMS:
        handle = primary if source == "primary" else sealed
        scores[arm] = {
            "raw": np.asarray(handle[f"{prefix}__raw"], dtype=float),
            "calibrated": np.asarray(handle[f"{prefix}__calibrated"], dtype=float),
        }

    # PLAN.md section 4: B1_ORDER_LOGIT and B1_LOGIT must be the same fitted model.
    for scale in ("raw", "calibrated"):
        if not np.allclose(scores["B1_ORDER_LOGIT"][scale], scores["B1_LOGIT"][scale], atol=0, rtol=0):
            die(f"B1_ORDER_LOGIT and B1_LOGIT {scale} scores differ; the two runs' B1 is not one model")

    # Join check: recomputing every stored external metric from the joined labels must reproduce
    # the parent run's values. This is what establishes that the label join is correct.
    stored_primary = json.loads(PRIMARY_JSON.read_text())
    stored_expl = json.loads(EXPLORATORY_JSON.read_text())
    join_checks = []
    for arm, _, source in ARMS:
        for scale in ("raw", "calibrated"):
            got = float(roc_auc_score(y, scores[arm][scale]))
            if source == "primary":
                want = float(stored_primary["arms"][arm][scale]["metrics"]["roc_auc"])
            else:
                want = float(stored_expl["arms"][arm][scale]["auc"])
            join_checks.append({"arm": arm, "scale": scale, "recomputed": got, "stored": want,
                                "abs_difference": abs(got - want)})
            if abs(got - want) > 1e-12:
                die(f"recomputed ROC-AUC for {arm}/{scale} is {got}, stored is {want}; join rejected")

    # PLAN.md section 5: the reimplemented estimator must reproduce the parent primary contrast.
    want_point = float(stored_primary["primary_contrast"]["point_gain"])
    want_ci95 = [float(v) for v in stored_primary["primary_contrast"]["ci95"]]
    want_seed = int(stored_primary["primary_contrast"]["bootstrap_seed"])
    reproduction = paired_cluster_bootstrap(
        y, scores["B1_ORDER_LOGIT"]["calibrated"], scores["B1_ORDER_PLUS_PROTT5_PLS_LOGIT"]["calibrated"],
        groups, lambda yy, rr, cc: auc_gain(yy, rr, cc), replicates=REPLICATES, seed=want_seed)
    if abs(reproduction["point_gain"] - want_point) > 1e-12:
        die(f"reimplementation point gain {reproduction['point_gain']} != stored {want_point}")
    for got, want in zip(reproduction["ci95"], want_ci95):
        if abs(got - want) > 1e-12:
            die(f"reimplementation ci95 {reproduction['ci95']} != stored {want_ci95}")

    # Arm-level descriptive metrics and the budget table.
    arm_rows = []
    for arm, _, _ in ARMS:
        for scale in ("raw", "calibrated"):
            score = scores[arm][scale]
            entry = {
                "arm": arm,
                "probabilities": scale,
                "roc_auc": float(roc_auc_score(y, score)),
                "average_precision": float(average_precision_score(y, score)),
                "brier": float(brier_score_loss(y, score)),
                "log_loss": float(log_loss(y, score)),
                "calibration": calibration_intercept_slope(y, score),
                "top_k": {},
            }
            for k in K_GRID:
                successes = top_k_successes(y, score, k)
                entry["top_k"][str(k)] = {
                    "n_success": successes,
                    "precision": successes / k,
                    "lift": (successes / k) / POOL_PREVALENCE,
                }
            arm_rows.append(entry)

    # Does the calibration map change the ranking at all?
    ranking_identical = {}
    for arm, _, _ in ARMS:
        same = {}
        for k in K_GRID:
            same[str(k)] = bool(
                top_k_successes(y, scores[arm]["raw"], k) == top_k_successes(y, scores[arm]["calibrated"], k))
        ranking_identical[arm] = same

    # Contrast table: every contrast x every k, reported regardless of sign.
    contrast_rows = []
    for name, ref_arm, cand_arm, status in CONTRASTS:
        for k in K_GRID:
            boot = paired_cluster_bootstrap(
                y, scores[ref_arm]["calibrated"], scores[cand_arm]["calibrated"], groups,
                (lambda kk: (lambda yy, rr, cc: precision_gain(yy, rr, cc, kk)))(k))
            contrast_rows.append({
                "contrast": name, "status": status, "reference": ref_arm, "candidate": cand_arm,
                "k": k,
                "reference_n_success": top_k_successes(y, scores[ref_arm]["calibrated"], k),
                "candidate_n_success": top_k_successes(y, scores[cand_arm]["calibrated"], k),
                "precision_difference": boot["point_gain"],
                "ci95": boot["ci95"],
                "bootstrap_sd": boot["bootstrap_sd"],
                "n_replicates_valid": boot["n_replicates_valid"],
                "invalid_fraction": boot["invalid_fraction"],
                "resampled_rows_median": boot["resampled_rows_median"],
            })

    # Diagnostic assigned in review: a negative calibration slope alongside ROC-AUC above 0.5.
    slope_diagnostic = {}
    for arm, _, _ in ARMS:
        for scale in ("raw", "calibrated"):
            score = np.clip(scores[arm][scale], 1e-8, 1 - 1e-8)
            logit = np.log(score / (1 - score))
            cal = calibration_intercept_slope(y, score)
            rank = np.argsort(np.argsort(logit, kind="stable"), kind="stable").astype(float)
            rank = (rank - rank.mean()) / rank.std()
            slope_diagnostic[f"{arm}/{scale}"] = {
                "roc_auc": float(roc_auc_score(y, score)),
                "calibration_slope": cal["slope"],
                "calibration_intercept": cal["intercept"],
                "mean_logit_positive": float(logit[y == 1].mean()),
                "mean_logit_negative": float(logit[y == 0].mean()),
                "mean_logit_difference": float(logit[y == 1].mean() - logit[y == 0].mean()),
                "logit_sd": float(logit.std()),
                "logit_abs_max": float(np.abs(logit).max()),
                "slope_on_standardized_rank_of_logit": float(
                    LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
                    .fit(rank.reshape(-1, 1), y).coef_[0, 0]),
            }

    results = {
        "run_id": "stam_external_screening_budget_20261025",
        "analysis_scope": "descriptive_re_expression_of_sealed_retrospective_external_predictions",
        "fitting_performed": False,
        "calibration_performed": False,
        "model_selection_performed": False,
        "confirmatory": False,
        "endpoint": "recombinant soluble expression in Escherichia coli, not direct aqueous peptide solubility",
        "pool": {"n_rows": n_rows, "n_positive": n_pos, "n_negative": n_rows - n_pos,
                 "n_groups": int(len(set(groups.tolist()))), "prevalence": prevalence,
                 "prevalence_note": "balanced by the source publication's construction; not a screening-pool prevalence"},
        "k_grid": list(K_GRID),
        "k_grid_provenance": "fixed in PLAN.md before execution; microplate formats plus the positive count (R-precision)",
        "bootstrap": {"replicates": REPLICATES, "seed": SEED, "unit": "external sequence-identity cluster",
                      "ci_scope": "conditional on the sealed predictions; excludes refitting, historical method selection and between-study uncertainty"},
        "input_sha256": {str(p.relative_to(PARENT.parent)): sha256(p)
                         for p in (PRIMARY_NPZ, SEALED_NPZ, PRIMARY_JSON, EXPLORATORY_JSON)},
        "join_checks": join_checks,
        "estimator_reproduction_check": {
            "stored_point_gain": want_point, "stored_ci95": want_ci95, "seed": want_seed,
            "recomputed": reproduction,
            "verdict": "reimplementation reproduces the parent run's primary ROC-AUC contrast exactly",
        },
        "arms": arm_rows,
        "calibration_map_preserves_top_k": ranking_identical,
        "contrasts": contrast_rows,
        "calibration_slope_diagnostic": slope_diagnostic,
        "cells_reported": {"arm_by_k": len(ARMS) * 2 * len(K_GRID),
                           "contrast_by_k": len(CONTRASTS) * len(K_GRID)},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__, "platform": platform.platform()},
    }

    (HERE / "results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")

    with (HERE / "metrics.tsv").open("w", encoding="utf-8") as handle:
        handle.write("arm\tprobabilities\troc_auc\taverage_precision\tbrier\tlog_loss\t"
                     "calibration_intercept\tcalibration_slope\t"
                     + "\t".join(f"n_success_top{k}\tprecision_top{k}" for k in K_GRID) + "\n")
        for row in arm_rows:
            cells = [f"{row['top_k'][str(k)]['n_success']}\t{row['top_k'][str(k)]['precision']:.6f}"
                     for k in K_GRID]
            handle.write(f"{row['arm']}\t{row['probabilities']}\t{row['roc_auc']:.10f}\t"
                         f"{row['average_precision']:.10f}\t{row['brier']:.10f}\t{row['log_loss']:.10f}\t"
                         f"{row['calibration']['intercept']:.10f}\t{row['calibration']['slope']:.10f}\t"
                         + "\t".join(cells) + "\n")

    with (HERE / "contrasts.tsv").open("w", encoding="utf-8") as handle:
        handle.write("contrast\tstatus\treference\tcandidate\tk\treference_n_success\t"
                     "candidate_n_success\tprecision_difference\tci95_low\tci95_high\n")
        for row in contrast_rows:
            handle.write(f"{row['contrast']}\t{row['status']}\t{row['reference']}\t{row['candidate']}\t"
                         f"{row['k']}\t{row['reference_n_success']}\t{row['candidate_n_success']}\t"
                         f"{row['precision_difference']:+.6f}\t{row['ci95'][0]:+.6f}\t{row['ci95'][1]:+.6f}\n")

    print(f"pool {n_rows} rows / {n_pos} positives / {results['pool']['n_groups']} clusters")
    print("join check: all stored external ROC-AUC values reproduced to <1e-12")
    print(f"estimator check: reproduced stored primary contrast {want_point:.16f} and its 95% interval")
    print(f"wrote results.json, metrics.tsv ({len(arm_rows)} rows), contrasts.tsv ({len(contrast_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
