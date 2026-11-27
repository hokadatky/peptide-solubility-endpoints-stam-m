"""Leakage-safe nested L2-logistic heads for the frozen analysis contract."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning

try:
    from .split_contracts import (
        assert_no_group_overlap,
        balanced_group_folds,
        indices_for_fold,
        validate_group_folds,
    )
except ImportError:  # pragma: no cover - direct execution path
    from split_contracts import (
        assert_no_group_overlap,
        balanced_group_folds,
        indices_for_fold,
        validate_group_folds,
    )


LOGISTIC_MAX_ITER = 20_000
LOGISTIC_TOL = 1e-4
PLS_MAX_ITER = 500
PLS_TOL = 1e-6


class ModelConvergenceError(RuntimeError):
    """A frozen fit failed its numerical acceptance contract."""


@dataclass
class FittedHead:
    mode: str
    c_value: float
    pls_k: int | None
    embedding_scaler: StandardScaler | None
    pls: PLSRegression | None
    matrix_scaler: StandardScaler
    classifier: LogisticRegression
    diagnostics: dict[str, Any]


def _warning_summary(caught: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    return [
        {
            "category": item.category.__name__,
            "message_sha256": hashlib.sha256(str(item.message).encode("utf-8")).hexdigest(),
        }
        for item in caught
    ]


def _fit_logistic(
    matrix: np.ndarray,
    y: np.ndarray,
    c_value: float,
    *,
    require_convergence: bool,
    max_iter: int | None = None,
) -> tuple[LogisticRegression, dict[str, Any]]:
    resolved_max_iter = LOGISTIC_MAX_ITER if max_iter is None else int(max_iter)
    classifier = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        class_weight=None,
        max_iter=resolved_max_iter,
        tol=LOGISTIC_TOL,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classifier.fit(np.asarray(matrix, dtype=np.float64), np.asarray(y).astype(int))
    iterations = [int(value) for value in np.asarray(classifier.n_iter_).ravel()]
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    coefficients_finite = bool(
        np.isfinite(classifier.coef_).all() and np.isfinite(classifier.intercept_).all()
    )
    converged = bool(
        not convergence_warning
        and coefficients_finite
        and iterations
        and max(iterations) < resolved_max_iter
    )
    diagnostics = {
        "solver": "lbfgs",
        "tol": LOGISTIC_TOL,
        "max_iter": resolved_max_iter,
        "n_iter": iterations,
        "warnings": _warning_summary(caught),
        "convergence_warning": convergence_warning,
        "coefficients_finite": coefficients_finite,
        "converged": converged,
    }
    if require_convergence and not converged:
        raise ModelConvergenceError(
            f"Frozen L2 logistic fit did not converge for C={float(c_value)}: {diagnostics}"
        )
    return classifier, diagnostics


def _fit_pls_projection(
    embedding: np.ndarray,
    y: np.ndarray,
    n_components: int,
    *,
    require_convergence: bool,
) -> tuple[StandardScaler, PLSRegression, np.ndarray, dict[str, Any]]:
    embedding = _as_matrix(embedding, len(y))
    scaler = StandardScaler().fit(embedding)
    scaled = scaler.transform(embedding)
    pls = PLSRegression(
        n_components=int(n_components), max_iter=PLS_MAX_ITER, tol=PLS_TOL
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pls.fit(scaled, np.asarray(y, dtype=float))
    projected = pls.transform(scaled)
    iterations = [int(value) for value in getattr(pls, "n_iter_", [])]
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    projected_finite = bool(np.isfinite(projected).all())
    projection_rank = int(np.linalg.matrix_rank(projected)) if projected_finite else 0
    converged = bool(
        not convergence_warning
        and projected_finite
        and len(iterations) == int(n_components)
        and all(value < PLS_MAX_ITER for value in iterations)
    )
    diagnostics = {
        "n_components": int(n_components),
        "tol": PLS_TOL,
        "max_iter": PLS_MAX_ITER,
        "n_iter": iterations,
        "warnings": _warning_summary(caught),
        "convergence_warning": convergence_warning,
        "projected_finite": projected_finite,
        "projection_rank": projection_rank,
        "converged": converged,
    }
    if require_convergence and not converged:
        raise ModelConvergenceError(
            f"Frozen PLS fit did not converge for k={int(n_components)}: {diagnostics}"
        )
    return scaler, pls, projected, diagnostics


def _as_matrix(value: np.ndarray | None, n_rows: int) -> np.ndarray:
    if value is None:
        return np.zeros((n_rows, 0), dtype=np.float64)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != n_rows:
        raise ValueError("Feature matrix has an incompatible shape.")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values.")
    return matrix


def _representation(
    base: np.ndarray,
    embedding: np.ndarray | None,
    y: np.ndarray | None,
    mode: str,
    pls_k: int | None,
    embedding_scaler: StandardScaler | None = None,
    pls: PLSRegression | None = None,
    fit_projection: bool = False,
) -> tuple[np.ndarray, StandardScaler | None, PLSRegression | None]:
    n_rows = len(base)
    base = _as_matrix(base, n_rows)
    if mode == "none":
        return base, None, None
    if embedding is None:
        raise ValueError(f"Embedding is required for mode={mode}.")
    embedding = _as_matrix(embedding, n_rows)
    if mode == "raw":
        return np.hstack([base, embedding]), None, None
    if mode != "pls":
        raise ValueError(f"Unknown representation mode: {mode}")
    if pls_k is None or pls_k < 1:
        raise ValueError("PLS mode requires a positive component count.")
    if fit_projection:
        if y is None:
            raise ValueError("PLS fitting requires labels.")
        embedding_scaler, pls, projected, _ = _fit_pls_projection(
            embedding, np.asarray(y), int(pls_k), require_convergence=True
        )
        return np.hstack([base, projected]), embedding_scaler, pls
    if embedding_scaler is None or pls is None:
        raise ValueError("PLS transform requested without a fitted projection.")
    projected = pls.transform(embedding_scaler.transform(embedding))
    return np.hstack([base, projected]), embedding_scaler, pls


def fit_head(
    base: np.ndarray,
    embedding: np.ndarray | None,
    y: np.ndarray,
    mode: str,
    c_value: float,
    pls_k: int | None = None,
) -> FittedHead:
    y = np.asarray(y).astype(int)
    base = _as_matrix(base, len(y))
    pls_diagnostics: dict[str, Any] | None = None
    if mode == "pls":
        if embedding is None or pls_k is None or int(pls_k) < 1:
            raise ValueError("PLS mode requires an embedding and positive component count.")
        embedding_scaler, pls, projected, pls_diagnostics = _fit_pls_projection(
            embedding, y, int(pls_k), require_convergence=True
        )
        matrix = np.hstack([base, projected])
    else:
        matrix, embedding_scaler, pls = _representation(
            base, embedding, None, mode, pls_k, fit_projection=False
        )
    matrix_scaler = StandardScaler().fit(matrix)
    scaled = matrix_scaler.transform(matrix)
    if not np.isfinite(scaled).all():
        raise ModelConvergenceError("Outer-refit scaled model matrix is non-finite.")
    classifier, logistic_diagnostics = _fit_logistic(
        scaled, y, float(c_value), require_convergence=True
    )
    return FittedHead(
        mode=mode,
        c_value=float(c_value),
        pls_k=int(pls_k) if pls_k is not None else None,
        embedding_scaler=embedding_scaler,
        pls=pls,
        matrix_scaler=matrix_scaler,
        classifier=classifier,
        diagnostics={"pls": pls_diagnostics, "logistic": logistic_diagnostics},
    )


def _fit_transformers(
    base: np.ndarray,
    embedding: np.ndarray | None,
    y: np.ndarray,
    mode: str,
    pls_k: int | None,
) -> tuple[StandardScaler | None, PLSRegression | None, StandardScaler, np.ndarray]:
    matrix, embedding_scaler, pls = _representation(base, embedding, y, mode, pls_k, fit_projection=True)
    matrix_scaler = StandardScaler().fit(matrix)
    return embedding_scaler, pls, matrix_scaler, matrix_scaler.transform(matrix)


def _transform_with_transformers(
    base: np.ndarray,
    embedding: np.ndarray | None,
    mode: str,
    pls_k: int | None,
    embedding_scaler: StandardScaler | None,
    pls: PLSRegression | None,
    matrix_scaler: StandardScaler,
) -> np.ndarray:
    matrix, _, _ = _representation(
        base,
        embedding,
        None,
        mode,
        pls_k,
        embedding_scaler=embedding_scaler,
        pls=pls,
        fit_projection=False,
    )
    return matrix_scaler.transform(matrix)


def predict_head(head: FittedHead, base: np.ndarray, embedding: np.ndarray | None) -> np.ndarray:
    matrix, _, _ = _representation(
        base,
        embedding,
        None,
        head.mode,
        head.pls_k,
        embedding_scaler=head.embedding_scaler,
        pls=head.pls,
        fit_projection=False,
    )
    probability = head.classifier.predict_proba(head.matrix_scaler.transform(matrix))[:, 1]
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise ModelConvergenceError("Model prediction failed probability finiteness/range gates.")
    return probability


def _grid(mode: str, c_grid: list[float], pls_k_grid: list[int]) -> list[tuple[int | None, float]]:
    if mode == "pls":
        return [(int(k), float(c)) for k in pls_k_grid for c in c_grid]
    return [(None, float(c)) for c in c_grid]


def _param_sort_key(params: tuple[int | None, float]) -> tuple[float, float]:
    k, c_value = params
    return (float(k or 0), float(c_value))


def select_hyperparameter(
    base: np.ndarray,
    embedding: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
    mode: str,
    c_grid: list[float],
    pls_k_grid: list[int],
    inner_folds: int,
    seed: int,
    *,
    inner_assignment: np.ndarray | None = None,
) -> tuple[tuple[int | None, float], list[dict[str, Any]], str]:
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups).astype(str)
    if inner_assignment is None:
        inner_assignment = balanced_group_folds(y, groups, inner_folds, seed)
    else:
        inner_assignment = np.asarray(inner_assignment).astype(int)
        validate_group_folds(y, groups, inner_assignment, inner_folds)
    candidates = _grid(mode, c_grid, pls_k_grid)
    predictions = {
        (pls_k, float(c_value)): np.full(len(y), np.nan, dtype=float)
        for pls_k, c_value in candidates
    }
    candidate_diagnostics: dict[tuple[int | None, float], list[dict[str, Any]]] = {
        (pls_k, float(c_value)): [] for pls_k, c_value in candidates
    }
    for fold in range(inner_folds):
        train, test = indices_for_fold(inner_assignment, fold)
        assert_no_group_overlap(groups[train], groups[test])
        if mode == "pls":
            # PLS components are generated sequentially.  A max-k fit therefore
            # contains the same first k score coordinates (to floating-point
            # precision) as a separate k-component fit.  Fit the projection
            # once per inner fold, then keep the declared k-specific matrix
            # scaling and L2-logistic fits unchanged.
            if embedding is None:
                raise ValueError("PLS mode requires an embedding matrix.")
            max_k = max(int(value) for value in pls_k_grid)
            embedding_scaler, pls_family, projected_train, pls_diagnostics = _fit_pls_projection(
                embedding[train], y[train], max_k, require_convergence=False
            )
            projected_test = pls_family.transform(embedding_scaler.transform(embedding[test]))
            for pls_k in sorted({int(key[0]) for key in candidates if key[0] is not None}):
                prefix_iterations = list(pls_diagnostics["n_iter"][:pls_k])
                prefix_finite = bool(
                    np.isfinite(projected_train[:, :pls_k]).all()
                    and np.isfinite(projected_test[:, :pls_k]).all()
                )
                prefix_rank = int(np.linalg.matrix_rank(projected_train[:, :pls_k])) if prefix_finite else 0
                prefix_eligible = bool(
                    len(prefix_iterations) == pls_k
                    and all(value < PLS_MAX_ITER for value in prefix_iterations)
                    and prefix_finite
                    and prefix_rank == pls_k
                )
                train_matrix = np.hstack([base[train], projected_train[:, :pls_k]])
                test_matrix = np.hstack([base[test], projected_test[:, :pls_k]])
                matrix_scaler = StandardScaler().fit(train_matrix)
                train_scaled = matrix_scaler.transform(train_matrix)
                test_scaled = matrix_scaler.transform(test_matrix)
                for candidate_k, c_value in candidates:
                    if candidate_k != pls_k:
                        continue
                    key = (candidate_k, float(c_value))
                    if not prefix_eligible:
                        candidate_diagnostics[key].append({
                            "inner_fold": int(fold),
                            "eligible": False,
                            "reason": "PLS_PREFIX_NUMERICAL_GATE_FAILED",
                            "pls": {**pls_diagnostics, "prefix_k": pls_k, "prefix_rank": prefix_rank},
                            "logistic": None,
                        })
                        continue
                    classifier, logistic_diagnostics = _fit_logistic(
                        train_scaled, y[train], float(c_value), require_convergence=False
                    )
                    fold_probability = classifier.predict_proba(test_scaled)[:, 1]
                    probability_valid = bool(
                        np.isfinite(fold_probability).all()
                        and np.all((fold_probability >= 0.0) & (fold_probability <= 1.0))
                    )
                    eligible = bool(logistic_diagnostics["converged"] and probability_valid)
                    candidate_diagnostics[key].append({
                        "inner_fold": int(fold),
                        "eligible": eligible,
                        "reason": None if eligible else "LOGISTIC_NUMERICAL_GATE_FAILED",
                        "pls": {**pls_diagnostics, "prefix_k": pls_k, "prefix_rank": prefix_rank},
                        "logistic": logistic_diagnostics,
                        "probability_finite_and_in_range": probability_valid,
                    })
                    if eligible:
                        predictions[key][test] = fold_probability
            continue
        k_values = sorted({key[0] for key in candidates}, key=lambda value: int(value or 0))
        for pls_k in k_values:
            emb_scaler, pls, matrix_scaler, train_scaled = _fit_transformers(
                base[train],
                embedding[train] if embedding is not None else None,
                y[train],
                mode,
                pls_k,
            )
            test_scaled = _transform_with_transformers(
                base[test],
                embedding[test] if embedding is not None else None,
                mode,
                pls_k,
                emb_scaler,
                pls,
                matrix_scaler,
            )
            for candidate_k, c_value in candidates:
                if candidate_k != pls_k:
                    continue
                key = (candidate_k, float(c_value))
                classifier, logistic_diagnostics = _fit_logistic(
                    train_scaled, y[train], float(c_value), require_convergence=False
                )
                fold_probability = classifier.predict_proba(test_scaled)[:, 1]
                probability_valid = bool(
                    np.isfinite(fold_probability).all()
                    and np.all((fold_probability >= 0.0) & (fold_probability <= 1.0))
                )
                eligible = bool(logistic_diagnostics["converged"] and probability_valid)
                candidate_diagnostics[key].append({
                    "inner_fold": int(fold),
                    "eligible": eligible,
                    "reason": None if eligible else "LOGISTIC_NUMERICAL_GATE_FAILED",
                    "pls": None,
                    "logistic": logistic_diagnostics,
                    "probability_finite_and_in_range": probability_valid,
                })
                if eligible:
                    predictions[key][test] = fold_probability
    scores: list[dict[str, Any]] = []
    for pls_k, c_value in candidates:
        key = (pls_k, float(c_value))
        prediction = predictions[key]
        diagnostics = candidate_diagnostics[key]
        eligible = bool(
            len(diagnostics) == inner_folds
            and all(bool(row["eligible"]) for row in diagnostics)
            and np.isfinite(prediction).all()
        )
        scores.append({
            "pls_k": pls_k,
            "c": float(c_value),
            "eligible": eligible,
            "inner_roc_auc": float(roc_auc_score(y, prediction)) if eligible else None,
            "fold_diagnostics": diagnostics,
        })
    eligible_scores = [row for row in scores if row["eligible"]]
    if not eligible_scores:
        raise ModelConvergenceError("No frozen hyperparameter candidate passed all inner numerical gates.")
    eligible_scores.sort(
        key=lambda row: (-float(row["inner_roc_auc"]), float(row["pls_k"] or 0), float(row["c"]))
    )
    ineligible_scores = [row for row in scores if not row["eligible"]]
    ranked_scores = eligible_scores + ineligible_scores
    selected = eligible_scores[0]
    return (selected["pls_k"], selected["c"]), ranked_scores, _assignment_hash(groups, inner_assignment)


def _assignment_hash(groups: np.ndarray, folds: np.ndarray) -> str:
    try:
        from .split_contracts import fold_hash
    except ImportError:  # pragma: no cover - direct execution path
        from split_contracts import fold_hash

    return fold_hash(groups, folds)


def nested_oof(
    base: np.ndarray,
    embedding: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
    outer_folds: np.ndarray,
    mode: str,
    c_grid: list[float],
    pls_k_grid: list[int],
    inner_folds: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    y = np.asarray(y).astype(int)
    outer_folds = np.asarray(outer_folds).astype(int)
    n_outer = len(np.unique(outer_folds))
    prediction = np.full(len(y), np.nan, dtype=float)
    fold_info: list[dict[str, Any]] = []
    for outer_fold in sorted(np.unique(outer_folds).tolist()):
        train, test = indices_for_fold(outer_folds, int(outer_fold))
        assert_no_group_overlap(groups[train], groups[test])
        selected, grid_scores, inner_hash = select_hyperparameter(
            base[train],
            embedding[train] if embedding is not None else None,
            y[train],
            groups[train],
            mode,
            c_grid,
            pls_k_grid,
            inner_folds,
            seed + int(outer_fold),
        )
        head = fit_head(
            base[train],
            embedding[train] if embedding is not None else None,
            y[train],
            mode,
            selected[1],
            selected[0],
        )
        prediction[test] = predict_head(
            head,
            base[test],
            embedding[test] if embedding is not None else None,
        )
        fold_info.append({
            "outer_fold": int(outer_fold),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "selected_pls_k": selected[0],
            "selected_c": float(selected[1]),
            "inner_fold_hash": inner_hash,
            "inner_grid": grid_scores,
            "outer_fit_diagnostics": head.diagnostics,
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("Outer OOF prediction is incomplete.")
    if len(fold_info) != n_outer:
        raise RuntimeError("Outer fold count mismatch.")
    return prediction, fold_info


def fit_final_with_reserved_calibration(
    base: np.ndarray,
    embedding: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    mode: str,
    c_grid: list[float],
    pls_k_grid: list[int],
    inner_folds: int,
    seed: int,
) -> tuple[FittedHead, tuple[int | None, float], list[dict[str, Any]], np.ndarray, np.ndarray]:
    assert_no_group_overlap(groups[fit_indices], groups[calibration_indices])
    selected, grid_scores, inner_hash = select_hyperparameter(
        base[fit_indices],
        embedding[fit_indices] if embedding is not None else None,
        y[fit_indices],
        groups[fit_indices],
        mode,
        c_grid,
        pls_k_grid,
        inner_folds,
        seed,
    )
    head = fit_head(
        base[fit_indices],
        embedding[fit_indices] if embedding is not None else None,
        y[fit_indices],
        mode,
        selected[1],
        selected[0],
    )
    raw_calibration = predict_head(
        head,
        base[calibration_indices],
        embedding[calibration_indices] if embedding is not None else None,
    )
    return head, selected, [{"inner_fold_hash": inner_hash, "grid": grid_scores}], raw_calibration, calibration_indices


def select_on_reserved_partition(
    base: np.ndarray,
    embedding: np.ndarray | None,
    y: np.ndarray,
    fit_indices: np.ndarray,
    model_selection_indices: np.ndarray,
    mode: str,
    c_grid: list[float],
    pls_k_grid: list[int],
) -> tuple[tuple[int | None, float], list[dict[str, Any]]]:
    """Select the frozen specification using fit and model-selection only."""
    y = np.asarray(y).astype(int)
    base = _as_matrix(base, len(y))
    embedding = None if embedding is None else _as_matrix(embedding, len(y))
    fit_indices = np.asarray(fit_indices, dtype=int)
    model_selection_indices = np.asarray(model_selection_indices, dtype=int)
    if np.intersect1d(fit_indices, model_selection_indices).size:
        raise ValueError("fit and model_selection rows overlap.")
    if set(np.unique(y[fit_indices])) != {0, 1} or set(np.unique(y[model_selection_indices])) != {0, 1}:
        raise ValueError("fit and model_selection must each contain both classes.")
    candidates = _grid(mode, c_grid, pls_k_grid)
    rows: list[dict[str, Any]] = []

    if mode == "pls":
        if embedding is None:
            raise ValueError("PLS mode requires embeddings.")
        max_k = max(int(value) for value in pls_k_grid)
        emb_scaler, pls, projected_fit, pls_diagnostics = _fit_pls_projection(
            embedding[fit_indices], y[fit_indices], max_k, require_convergence=False
        )
        projected_selection = pls.transform(emb_scaler.transform(embedding[model_selection_indices]))
        for pls_k, c_value in candidates:
            assert pls_k is not None
            prefix_fit = projected_fit[:, :pls_k]
            prefix_selection = projected_selection[:, :pls_k]
            prefix_finite = bool(np.isfinite(prefix_fit).all() and np.isfinite(prefix_selection).all())
            prefix_rank = int(np.linalg.matrix_rank(prefix_fit)) if prefix_finite else 0
            prefix_iterations = list(pls_diagnostics["n_iter"][:pls_k])
            prefix_eligible = bool(
                len(prefix_iterations) == pls_k
                and all(value < PLS_MAX_ITER for value in prefix_iterations)
                and prefix_finite
                and prefix_rank == pls_k
            )
            if not prefix_eligible:
                rows.append({"pls_k": pls_k, "c": c_value, "eligible": False, "model_selection_roc_auc": None, "pls": {**pls_diagnostics, "prefix_rank": prefix_rank}, "logistic": None})
                continue
            fit_matrix = np.hstack([base[fit_indices], prefix_fit])
            selection_matrix = np.hstack([base[model_selection_indices], prefix_selection])
            scaler = StandardScaler().fit(fit_matrix)
            classifier, logistic_diagnostics = _fit_logistic(
                scaler.transform(fit_matrix), y[fit_indices], c_value, require_convergence=False
            )
            probability = classifier.predict_proba(scaler.transform(selection_matrix))[:, 1]
            probability_valid = bool(np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1)))
            eligible = bool(logistic_diagnostics["converged"] and probability_valid)
            rows.append({"pls_k": pls_k, "c": c_value, "eligible": eligible, "model_selection_roc_auc": float(roc_auc_score(y[model_selection_indices], probability)) if eligible else None, "pls": {**pls_diagnostics, "prefix_rank": prefix_rank}, "logistic": logistic_diagnostics})
    else:
        fit_matrix, _, _ = _representation(base[fit_indices], None if embedding is None else embedding[fit_indices], None, mode, None, fit_projection=False)
        selection_matrix, _, _ = _representation(base[model_selection_indices], None if embedding is None else embedding[model_selection_indices], None, mode, None, fit_projection=False)
        scaler = StandardScaler().fit(fit_matrix)
        fit_scaled = scaler.transform(fit_matrix)
        selection_scaled = scaler.transform(selection_matrix)
        for pls_k, c_value in candidates:
            classifier, logistic_diagnostics = _fit_logistic(
                fit_scaled, y[fit_indices], c_value, require_convergence=False
            )
            probability = classifier.predict_proba(selection_scaled)[:, 1]
            probability_valid = bool(np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1)))
            eligible = bool(logistic_diagnostics["converged"] and probability_valid)
            rows.append({"pls_k": pls_k, "c": c_value, "eligible": eligible, "model_selection_roc_auc": float(roc_auc_score(y[model_selection_indices], probability)) if eligible else None, "pls": None, "logistic": logistic_diagnostics})

    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        raise ModelConvergenceError("No candidate passed reserved model-selection numerical gates.")
    eligible_rows.sort(key=lambda row: (-float(row["model_selection_roc_auc"]), float(row["pls_k"] or 0), float(row["c"])))
    selected = eligible_rows[0]
    return (selected["pls_k"], float(selected["c"])), eligible_rows + [row for row in rows if not row["eligible"]]


class PlattCalibrator:
    """A minimal, explicit Platt sigmoid fit on a reserved partition."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.diagnostics: dict[str, Any] | None = None

    @staticmethod
    def _logit(score: np.ndarray) -> np.ndarray:
        score = np.clip(np.asarray(score, dtype=float), 1e-8, 1 - 1e-8)
        return np.log(score / (1.0 - score)).reshape(-1, 1)

    def fit(self, score: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.model, self.diagnostics = _fit_logistic(
            self._logit(score), np.asarray(y).astype(int), 1e6, require_convergence=True
        )
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Platt calibrator is not fitted.")
        probability = self.model.predict_proba(self._logit(score))[:, 1]
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ModelConvergenceError("Platt-calibrated probabilities failed numerical gates.")
        return probability


def fit_external_development_seal(
    base: np.ndarray,
    embedding: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
    fit_indices: np.ndarray,
    model_selection_indices: np.ndarray,
    calibration_indices: np.ndarray,
    mode: str,
    c_grid: list[float],
    pls_k_grid: list[int],
) -> tuple[FittedHead, PlattCalibrator, tuple[int | None, float], list[dict[str, Any]], dict[str, Any]]:
    """Seal selection, refit, and calibration using development rows only."""
    groups = np.asarray(groups).astype(str)
    fit_indices = np.asarray(fit_indices, dtype=int)
    model_selection_indices = np.asarray(model_selection_indices, dtype=int)
    calibration_indices = np.asarray(calibration_indices, dtype=int)
    assert_no_group_overlap(groups[fit_indices], groups[model_selection_indices])
    assert_no_group_overlap(groups[fit_indices], groups[calibration_indices])
    assert_no_group_overlap(groups[model_selection_indices], groups[calibration_indices])
    selected, selection_grid = select_on_reserved_partition(
        base, embedding, y, fit_indices, model_selection_indices, mode, c_grid, pls_k_grid
    )
    refit_indices = np.concatenate([fit_indices, model_selection_indices])
    head = fit_head(
        np.asarray(base)[refit_indices],
        None if embedding is None else np.asarray(embedding)[refit_indices],
        np.asarray(y)[refit_indices],
        mode,
        selected[1],
        selected[0],
    )
    raw_calibration = predict_head(
        head,
        np.asarray(base)[calibration_indices],
        None if embedding is None else np.asarray(embedding)[calibration_indices],
    )
    calibrator = PlattCalibrator().fit(raw_calibration, np.asarray(y)[calibration_indices])
    diagnostics = {
        "selected": [selected[0], selected[1]],
        "selection_grid": selection_grid,
        "refit": head.diagnostics,
        "calibration": calibrator.diagnostics,
        "n_fit": int(len(fit_indices)),
        "n_model_selection": int(len(model_selection_indices)),
        "n_refit": int(len(refit_indices)),
        "n_calibration": int(len(calibration_indices)),
    }
    return head, calibrator, selected, selection_grid, diagnostics
