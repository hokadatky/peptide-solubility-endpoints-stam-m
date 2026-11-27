"""Immutable, validation-first checkpoints for nested OOF model fits."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

try:
    from .nested_models import (
        LOGISTIC_MAX_ITER,
        LOGISTIC_TOL,
        PLS_MAX_ITER,
        PLS_TOL,
        fit_head,
        predict_head,
        select_hyperparameter,
    )
    from .split_contracts import (
        SPLIT_CONTRACT_VERSION,
        assert_no_group_overlap,
        balanced_group_folds,
        fold_hash,
        indices_for_fold,
    )
except ImportError:  # pragma: no cover - direct execution path
    from nested_models import (
        LOGISTIC_MAX_ITER,
        LOGISTIC_TOL,
        PLS_MAX_ITER,
        PLS_TOL,
        fit_head,
        predict_head,
        select_hyperparameter,
    )
    from split_contracts import (
        SPLIT_CONTRACT_VERSION,
        assert_no_group_overlap,
        balanced_group_folds,
        fold_hash,
        indices_for_fold,
    )


CHECKPOINT_SCHEMA = "nested_oof_outer_fold_checkpoint_v2"


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _contract_json(
    *,
    contract_tag: str,
    global_hashes: dict[str, str | None],
    n_rows: int,
    base_shape: tuple[int, ...],
    embedding_shape: tuple[int, ...] | None,
    outer_fold_hash: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    inner_assignment_hash: str,
    outer_fold: int,
    mode: str,
    c_grid: list[float],
    pls_k_grid: list[int],
    inner_folds: int,
    seed: int,
) -> str:
    contract = {
        "schema": CHECKPOINT_SCHEMA,
        "contract_tag": str(contract_tag),
        "split_contract_version": SPLIT_CONTRACT_VERSION,
        "sklearn_version": sklearn.__version__,
        "n_rows": int(n_rows),
        "base_shape": [int(value) for value in base_shape],
        "embedding_shape": None if embedding_shape is None else [int(value) for value in embedding_shape],
        "global_hashes": global_hashes,
        "outer_fold_hash": str(outer_fold_hash),
        "train_indices_hash": _array_hash(np.asarray(train_indices, dtype=np.int64)),
        "test_indices_hash": _array_hash(np.asarray(test_indices, dtype=np.int64)),
        "inner_assignment_hash": str(inner_assignment_hash),
        "outer_fold": int(outer_fold),
        "mode": str(mode),
        "c_grid": [float(value) for value in c_grid],
        "pls_k_grid": [int(value) for value in pls_k_grid],
        "inner_folds": int(inner_folds),
        "seed": int(seed),
        "inner_seed": int(seed) + int(outer_fold),
        "numerical_policy": {
            "logistic_solver": "lbfgs",
            "logistic_tol": LOGISTIC_TOL,
            "logistic_max_iter": LOGISTIC_MAX_ITER,
            "pls_tol": PLS_TOL,
            "pls_max_iter": PLS_MAX_ITER,
            "inner_nonconverged_candidate": "ineligible",
            "outer_nonconvergence": "hard_stop",
        },
    }
    return json.dumps(contract, sort_keys=True, separators=(",", ":"))


def _artifact_hash(
    contract_json: str,
    test_indices: np.ndarray,
    scores: np.ndarray,
    fold_info_json: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(contract_json.encode("utf-8"))
    digest.update(_array_hash(np.asarray(test_indices, dtype=np.int64)).encode("ascii"))
    digest.update(_array_hash(np.asarray(scores, dtype=np.float64)).encode("ascii"))
    digest.update(fold_info_json.encode("utf-8"))
    return digest.hexdigest()


def _save_checkpoint(
    path: Path,
    contract_json: str,
    test_indices: np.ndarray,
    scores: np.ndarray,
    fold_info: dict[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite model checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    fold_info_json = json.dumps(fold_info, sort_keys=True, separators=(",", ":"))
    artifact_sha256 = _artifact_hash(
        contract_json, test_indices, scores, fold_info_json
    )
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            contract_json=np.asarray([contract_json]),
            test_indices=np.asarray(test_indices, dtype=np.int64),
            scores=np.asarray(scores, dtype=np.float64),
            fold_info_json=np.asarray([fold_info_json]),
            artifact_sha256=np.asarray([artifact_sha256]),
        )
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    expected_contract_json: str,
    expected_test_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "contract_json",
            "test_indices",
            "scores",
            "fold_info_json",
            "artifact_sha256",
        }
        if not required.issubset(payload.files):
            raise ValueError(f"Checkpoint schema/integrity fields missing: {path}")
        observed_contract = str(payload["contract_json"][0])
        test_indices = np.asarray(payload["test_indices"], dtype=np.int64)
        scores = np.asarray(payload["scores"], dtype=np.float64)
        fold_info = json.loads(str(payload["fold_info_json"][0]))
        observed_artifact_sha256 = str(payload["artifact_sha256"][0])
    if observed_contract != expected_contract_json:
        raise ValueError(f"Checkpoint contract mismatch: {path}")
    if not np.array_equal(test_indices, expected_test_indices):
        raise ValueError(f"Checkpoint test-index mismatch: {path}")
    if scores.shape != (len(test_indices),) or not np.isfinite(scores).all():
        raise ValueError(f"Checkpoint score contract failed: {path}")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"Checkpoint probabilities outside [0, 1]: {path}")
    fold_info_json = json.dumps(fold_info, sort_keys=True, separators=(",", ":"))
    expected_artifact_sha256 = _artifact_hash(
        observed_contract, test_indices, scores, fold_info_json
    )
    if observed_artifact_sha256 != expected_artifact_sha256:
        raise ValueError(f"Checkpoint artifact digest mismatch: {path}")
    contract = json.loads(expected_contract_json)
    if fold_info.get("inner_fold_hash") != contract["inner_assignment_hash"]:
        raise ValueError(f"Checkpoint inner-assignment binding failed: {path}")
    outer_diagnostics = fold_info.get("outer_fit_diagnostics", {})
    if not outer_diagnostics.get("logistic", {}).get("converged", False):
        raise ValueError(f"Checkpoint outer logistic convergence gate failed: {path}")
    if contract["mode"] == "pls" and not outer_diagnostics.get("pls", {}).get("converged", False):
        raise ValueError(f"Checkpoint outer PLS convergence gate failed: {path}")
    return scores, fold_info


def checkpointed_nested_oof(
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
    checkpoint_dir: Path,
    checkpoint_name: str,
    contract_tag: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit or resume nested OOF predictions one immutable outer fold at a time."""
    base = np.asarray(base, dtype=np.float64)
    embedding = None if embedding is None else np.asarray(embedding, dtype=np.float64)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups).astype(str)
    outer_folds = np.asarray(outer_folds).astype(int)
    if base.ndim != 2 or base.shape[0] != len(y):
        raise ValueError("Checkpointed OOF base matrix shape mismatch.")
    if embedding is not None and (embedding.ndim != 2 or embedding.shape[0] != len(y)):
        raise ValueError("Checkpointed OOF embedding matrix shape mismatch.")
    if not np.isfinite(base).all() or (embedding is not None and not np.isfinite(embedding).all()):
        raise ValueError("Checkpointed OOF feature matrix contains non-finite values.")
    global_hashes: dict[str, str | None] = {
        "base": _array_hash(base),
        "embedding": None if embedding is None else _array_hash(embedding),
        "labels": _array_hash(np.asarray(y, dtype=np.int8)),
        "groups": _array_hash(np.asarray(groups).astype("U")),
        "outer_assignment": _array_hash(np.asarray(outer_folds, dtype=np.int64)),
    }
    outer_assignment_hash = fold_hash(groups, outer_folds)
    prediction = np.full(len(y), np.nan, dtype=float)
    fold_info: list[dict[str, Any]] = []
    for outer_fold in sorted(np.unique(outer_folds).tolist()):
        train, test = indices_for_fold(outer_folds, int(outer_fold))
        assert_no_group_overlap(groups[train], groups[test])
        inner_assignment = balanced_group_folds(
            y[train], groups[train], inner_folds, int(seed) + int(outer_fold)
        )
        inner_assignment_hash = fold_hash(groups[train], inner_assignment)
        checkpoint = checkpoint_dir / f"{checkpoint_name}_outer{int(outer_fold):02d}.npz"
        contract = _contract_json(
            contract_tag=contract_tag,
            global_hashes=global_hashes,
            n_rows=len(y),
            base_shape=base.shape,
            embedding_shape=None if embedding is None else embedding.shape,
            outer_fold_hash=outer_assignment_hash,
            train_indices=train,
            test_indices=test,
            inner_assignment_hash=inner_assignment_hash,
            outer_fold=int(outer_fold),
            mode=mode,
            c_grid=c_grid,
            pls_k_grid=pls_k_grid,
            inner_folds=inner_folds,
            seed=seed,
        )
        if checkpoint.is_file():
            fold_score, row = _load_checkpoint(checkpoint, contract, test)
            print(f"[checkpoint] reused {checkpoint.name}", flush=True)
        else:
            started = time.perf_counter()
            selected, grid_scores, inner_hash = select_hyperparameter(
                base[train],
                embedding[train] if embedding is not None else None,
                y[train],
                groups[train],
                mode,
                c_grid,
                pls_k_grid,
                inner_folds,
                int(seed) + int(outer_fold),
                inner_assignment=inner_assignment,
            )
            if inner_hash != inner_assignment_hash:
                raise RuntimeError("Selected-model inner assignment hash changed after sealing.")
            head = fit_head(
                base[train],
                embedding[train] if embedding is not None else None,
                y[train],
                mode,
                float(selected[1]),
                selected[0],
            )
            fold_score = predict_head(
                head,
                base[test],
                embedding[test] if embedding is not None else None,
            )
            row = {
                "outer_fold": int(outer_fold),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "selected_pls_k": selected[0],
                "selected_c": float(selected[1]),
                "inner_fold_hash": inner_hash,
                "inner_grid": grid_scores,
                "outer_fit_diagnostics": head.diagnostics,
                "fit_seconds": float(time.perf_counter() - started),
            }
            _save_checkpoint(checkpoint, contract, test, fold_score, row)
            print(f"[checkpoint] created {checkpoint.name} in {row['fit_seconds']:.1f}s", flush=True)
        prediction[test] = fold_score
        fold_info.append(row)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Checkpointed outer OOF prediction is incomplete.")
    if len(fold_info) != len(np.unique(outer_folds)):
        raise RuntimeError("Checkpointed outer fold count mismatch.")
    return prediction, fold_info
