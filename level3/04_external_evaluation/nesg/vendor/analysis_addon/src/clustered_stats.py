"""Paired cluster bootstrap, practical-margin decisions, and diagnostics."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _metric(y: np.ndarray, score: np.ndarray, name: str) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if name == "roc_auc":
        return float(roc_auc_score(y, score))
    if name == "auprc":
        return float(average_precision_score(y, score))
    if name == "brier":
        return float(brier_score_loss(y, score))
    if name == "log_loss":
        return float(log_loss(y, np.clip(score, 1e-12, 1 - 1e-12), labels=[0, 1]))
    raise ValueError(f"Unsupported metric: {name}")


def gain_value(y: np.ndarray, reference: np.ndarray, candidate: np.ndarray, metric: str) -> float:
    ref = _metric(y, reference, metric)
    cand = _metric(y, candidate, metric)
    return float(cand - ref) if metric in {"roc_auc", "auprc"} else float(ref - cand)


def paired_cluster_bootstrap(
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    metric: str = "roc_auc",
    replicates: int = 10_000,
    seed: int = 20260809,
) -> dict[str, object]:
    y = np.asarray(y).astype(int)
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    groups = np.asarray(groups).astype(str)
    if not (len(y) == len(reference) == len(candidate) == len(groups)):
        raise ValueError("Bootstrap inputs must have equal length.")
    unique_groups = np.asarray(sorted(np.unique(groups).tolist()), dtype=str)
    row_indices = [np.flatnonzero(groups == group) for group in unique_groups]
    point = gain_value(y, reference, candidate, metric)
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    invalid = 0
    for _ in range(int(replicates)):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        indices = np.concatenate([row_indices[index] for index in sampled])
        if len(np.unique(y[indices])) < 2:
            invalid += 1
            continue
        values.append(gain_value(y[indices], reference[indices], candidate[indices], metric))
    if not values:
        raise ValueError("All cluster bootstrap replicates were invalid.")
    values_array = np.asarray(values, dtype=float)
    invalid_fraction = invalid / int(replicates)
    return {
        "point_gain": float(point),
        "ci90": [float(value) for value in np.percentile(values_array, [5, 95])],
        "ci95": [float(value) for value in np.percentile(values_array, [2.5, 97.5])],
        "bootstrap_mean": float(values_array.mean()),
        "bootstrap_sd": float(values_array.std(ddof=1)) if len(values_array) > 1 else 0.0,
        "n_groups": int(len(unique_groups)),
        "n_replicates_requested": int(replicates),
        "n_replicates_valid": int(len(values_array)),
        "n_replicates_invalid": int(invalid),
        "invalid_fraction": float(invalid_fraction),
        "bootstrap_seed": int(seed),
        "bootstrap_unit": "cluster",
        "metric": metric,
    }


def decision_flags(ci90: list[float], ci95: list[float], margin: float) -> dict[str, bool | str]:
    lower90, upper90 = ci90
    lower95, upper95 = ci95
    flags = {
        "directional_advantage_95": bool(lower95 > 0),
        "directional_harm_95": bool(upper95 < 0),
        "meaningful_superiority_95": bool(lower95 > margin),
        "meaningful_harm_95": bool(upper95 < -margin),
        "equivalent_within_margin_90": bool(lower90 > -margin and upper90 < margin),
    }
    if flags["meaningful_superiority_95"]:
        status = "MEANINGFUL_GAIN"
    elif flags["meaningful_harm_95"]:
        status = "MEANINGFUL_HARM"
    elif flags["directional_advantage_95"] or flags["directional_harm_95"]:
        status = "STATISTICAL_BUT_SUBMARGIN"
    elif flags["equivalent_within_margin_90"]:
        status = "NO_PRACTICALLY_MEANINGFUL_DIFFERENCE"
    else:
        status = "INCONCLUSIVE"
    flags["decision_status"] = status
    return flags


def mde80_approx(bootstrap_sd: float) -> float:
    return float((1.959963984540054 + 0.8416212335729143) * bootstrap_sd)


def metric_bundle(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    result = {
        "roc_auc": _metric(y, score, "roc_auc"),
        "auprc": _metric(y, score, "auprc"),
        "brier": _metric(y, score, "brier"),
        "log_loss": _metric(y, score, "log_loss"),
    }
    return {key: float(value) for key, value in result.items()}


def calibration_intercept_slope(y: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y).astype(int)
    score = np.clip(np.asarray(score, dtype=float), 1e-8, 1 - 1e-8)
    logit = np.log(score / (1.0 - score)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return {"intercept": None, "slope": None}
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    model.fit(logit, y)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}


def reliability_table(y: np.ndarray, score: np.ndarray, bins: int = 10, min_bin_rows: int = 50) -> list[dict[str, float | int]]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    n_bins = int(bins)
    while n_bins > 1 and len(y) // n_bins < min_bin_rows:
        n_bins -= 1
    order = np.argsort(score, kind="mergesort")
    chunks = np.array_split(order, n_bins)
    rows = []
    for index, chunk in enumerate(chunks, start=1):
        if len(chunk) == 0:
            continue
        rows.append({
            "bin": int(index),
            "n": int(len(chunk)),
            "mean_predicted_probability": float(np.mean(score[chunk])),
            "observed_fraction_positive": float(np.mean(y[chunk])),
            "min_probability": float(np.min(score[chunk])),
            "max_probability": float(np.max(score[chunk])),
        })
    return rows


def cluster_swap_permutation(
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    metric: str = "roc_auc",
    replicates: int = 999,
    seed: int = 20260809,
) -> dict[str, object]:
    y = np.asarray(y).astype(int)
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    groups = np.asarray(groups).astype(str)
    unique_groups, group_inverse = np.unique(groups, return_inverse=True)
    observed = gain_value(y, reference, candidate, metric)
    rng = np.random.default_rng(int(seed))
    null: list[float] = []
    for _ in range(int(replicates)):
        # One Bernoulli draw per cluster, then broadcast it to rows.  This is
        # the same cluster-wise randomization as the scalar implementation but
        # avoids re-scanning every row once for every (mostly singleton)
        # cluster.
        swap_rows = rng.integers(0, 2, size=len(unique_groups), dtype=np.int8)[group_inverse].astype(bool)
        ref = np.where(swap_rows, candidate, reference)
        cand = np.where(swap_rows, reference, candidate)
        if len(np.unique(y)) < 2:
            continue
        null.append(gain_value(y, ref, cand, metric))
    null_array = np.asarray(null, dtype=float)
    p_value = float((1 + np.sum(np.abs(null_array) >= abs(observed))) / (len(null_array) + 1))
    return {
        "observed_gain": float(observed),
        "p_value_two_sided": p_value,
        "n_replicates": int(len(null_array)),
        "null_median": float(np.median(null_array)),
        "null_ci95": [float(value) for value in np.percentile(null_array, [2.5, 97.5])],
        "unit": "cluster_swap",
        "seed": int(seed),
    }
