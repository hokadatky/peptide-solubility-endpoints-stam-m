"""Training-label permutation controls with immutable repeat checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

try:
    from .clustered_stats import metric_bundle
    from .nested_models import fit_head, predict_head
    from .split_contracts import SPLIT_CONTRACT_VERSION, assert_no_group_overlap, fold_hash, indices_for_fold
except ImportError:  # pragma: no cover - direct execution path
    from clustered_stats import metric_bundle
    from nested_models import fit_head, predict_head
    from split_contracts import SPLIT_CONTRACT_VERSION, assert_no_group_overlap, fold_hash, indices_for_fold


def _hash_values(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _contract(
    contract_tag: str,
    y: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    reference_info: list[dict],
    candidate_info: list[dict],
    repeat: int,
    seed: int,
    base_hash: str,
    embedding_hash: str,
) -> str:
    value = {
        "schema": "training_label_permutation_checkpoint_v2",
        "split_contract_version": SPLIT_CONTRACT_VERSION,
        "contract_tag": contract_tag,
        "base_hash": base_hash,
        "embedding_hash": embedding_hash,
        "label_hash": _hash_values(np.asarray(y, dtype=np.int8)),
        "group_fold_hash": fold_hash(np.asarray(groups).astype(str), np.asarray(folds).astype(int)),
        "reference_selected": [
            [int(row["outer_fold"]), None, float(row["selected_c"])] for row in reference_info
        ],
        "candidate_selected": [
            [int(row["outer_fold"]), int(row["selected_pls_k"]), float(row["selected_c"])]
            for row in candidate_info
        ],
        "repeat": int(repeat),
        "seed": int(seed),
        "permutation_scope": "independent_within_each_outer_training_partition",
        "test_labels_shuffled": False,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_checkpoint(path: Path, contract: str, reference: np.ndarray, candidate: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite negative-control checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(contract.encode("utf-8"))
    digest.update(_hash_values(reference).encode("ascii"))
    digest.update(_hash_values(candidate).encode("ascii"))
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            contract_json=np.asarray([contract]),
            reference=reference,
            candidate=candidate,
            artifact_sha256=np.asarray([digest.hexdigest()]),
        )
    os.replace(temporary, path)


def _read_checkpoint(path: Path, contract: str, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if "artifact_sha256" not in payload.files:
            raise ValueError(f"Negative-control checkpoint schema/integrity field missing: {path}")
        observed = str(payload["contract_json"][0])
        reference = np.asarray(payload["reference"], dtype=np.float64)
        candidate = np.asarray(payload["candidate"], dtype=np.float64)
        artifact_sha256 = str(payload["artifact_sha256"][0])
    if observed != contract:
        raise ValueError(f"Negative-control checkpoint contract mismatch: {path}")
    for name, values in (("reference", reference), ("candidate", candidate)):
        if values.shape != (n_rows,) or not np.isfinite(values).all():
            raise ValueError(f"Negative-control {name} checkpoint shape/finiteness failed: {path}")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"Negative-control {name} probabilities outside [0, 1]: {path}")
    digest = hashlib.sha256()
    digest.update(observed.encode("utf-8"))
    digest.update(_hash_values(reference).encode("ascii"))
    digest.update(_hash_values(candidate).encode("ascii"))
    if digest.hexdigest() != artifact_sha256:
        raise ValueError(f"Negative-control checkpoint artifact digest mismatch: {path}")
    return reference, candidate


def training_label_permutation_control(
    base: np.ndarray,
    embedding: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    reference_info: list[dict],
    candidate_info: list[dict],
    replicates: int,
    seed: int,
    checkpoint_dir: Path,
    contract_tag: str,
) -> dict[str, object]:
    """Refit both frozen arms after shuffling only outer-training labels.

    Hyperparameters are held at those selected by the real nested fit.  This is
    a leakage/alignment negative control, not a second performance estimand.
    """
    base = np.asarray(base, dtype=np.float64)
    embedding = np.asarray(embedding, dtype=np.float64)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups).astype(str)
    folds = np.asarray(folds).astype(int)
    ref_by_fold = {int(row["outer_fold"]): row for row in reference_info}
    cand_by_fold = {int(row["outer_fold"]): row for row in candidate_info}
    expected_folds = set(np.unique(folds).tolist())
    if set(ref_by_fold) != expected_folds or set(cand_by_fold) != expected_folds:
        raise ValueError("Negative-control hyperparameters do not cover every outer fold.")
    auc_reference: list[float] = []
    auc_candidate: list[float] = []
    auc_gain: list[float] = []
    base_hash = _hash_values(base)
    embedding_hash = _hash_values(embedding)
    for repeat in range(int(replicates)):
        checkpoint = checkpoint_dir / f"label_permutation_{repeat:03d}.npz"
        contract = _contract(
            contract_tag, y, groups, folds, reference_info, candidate_info, repeat, seed,
            base_hash, embedding_hash,
        )
        if checkpoint.is_file():
            reference, candidate = _read_checkpoint(checkpoint, contract, len(y))
            print(f"[negative-control] reused {checkpoint.name}", flush=True)
        else:
            reference = np.full(len(y), np.nan, dtype=float)
            candidate = np.full(len(y), np.nan, dtype=float)
            for outer_fold in sorted(expected_folds):
                train, test = indices_for_fold(folds, int(outer_fold))
                assert_no_group_overlap(groups[train], groups[test])
                rng = np.random.default_rng(int(seed) + repeat * 1009 + int(outer_fold))
                permuted_train_y = rng.permutation(y[train])
                ref_row = ref_by_fold[int(outer_fold)]
                cand_row = cand_by_fold[int(outer_fold)]
                ref_head = fit_head(
                    base[train], None, permuted_train_y, "none", float(ref_row["selected_c"]), None
                )
                cand_head = fit_head(
                    base[train],
                    embedding[train],
                    permuted_train_y,
                    "pls",
                    float(cand_row["selected_c"]),
                    int(cand_row["selected_pls_k"]),
                )
                reference[test] = predict_head(ref_head, base[test], None)
                candidate[test] = predict_head(cand_head, base[test], embedding[test])
            if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
                raise RuntimeError("Negative-control OOF prediction is incomplete.")
            _write_checkpoint(checkpoint, contract, reference, candidate)
            print(f"[negative-control] created {checkpoint.name}", flush=True)
        ref_auc = float(metric_bundle(y, reference)["roc_auc"])
        cand_auc = float(metric_bundle(y, candidate)["roc_auc"])
        auc_reference.append(ref_auc)
        auc_candidate.append(cand_auc)
        auc_gain.append(cand_auc - ref_auc)

    def summarize(values: list[float], null_value: float) -> dict[str, object]:
        array = np.asarray(values, dtype=float)
        interval = [float(value) for value in np.percentile(array, [2.5, 97.5])]
        return {
            "median": float(np.median(array)),
            "mean": float(np.mean(array)),
            "sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
            "range95_percentile": interval,
            "null_value": float(null_value),
            "null_value_inside_range95": bool(interval[0] <= null_value <= interval[1]),
        }

    reference_summary = summarize(auc_reference, 0.5)
    candidate_summary = summarize(auc_candidate, 0.5)
    gain_summary = summarize(auc_gain, 0.0)
    passed = bool(
        reference_summary["null_value_inside_range95"]
        and candidate_summary["null_value_inside_range95"]
        and gain_summary["null_value_inside_range95"]
    )
    return {
        "status": "PASSED" if passed else "PIPELINE_INVALID",
        "n_replicates": int(replicates),
        "seed": int(seed),
        "training_labels_permuted": True,
        "test_labels_permuted": False,
        "permutation_scope": "independent within each outer training partition",
        "hyperparameter_policy": "fixed at each real-data outer fold selection; PLS is refit using permuted training labels",
        "reference_auc_null": reference_summary,
        "candidate_auc_null": candidate_summary,
        "candidate_minus_reference_auc_gain_null": gain_summary,
        "gate_definition": "0.5 must lie inside both model AUC 95% percentile ranges and 0 inside the gain range",
        "row_level_predictions_private_checkpoint_only": True,
    }
