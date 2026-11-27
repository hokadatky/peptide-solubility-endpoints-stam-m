"""Fixed, leakage-audited PLM teacher utilities for the distillation extension."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


sys.dont_write_bytecode = True


PLAN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PLAN_ROOT.parents[1]
V4_SRC = (
    PROJECT_ROOT / "paper" / "peptide_solubility_review" / "analysis_addon" / "src"
)
sys.path.insert(0, str(V4_SRC))

from nested_models import fit_head, predict_head  # noqa: E402
from sequence_features import build_b0_b1  # noqa: E402
from split_contracts import assert_no_group_overlap  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


@dataclass(frozen=True)
class DevelopmentData:
    record_ids: np.ndarray
    sequences: list[str]
    labels: np.ndarray
    groups: np.ndarray
    folds: np.ndarray
    b0: np.ndarray
    b1: np.ndarray
    b0_names: list[str]
    b1_names: list[str]


@dataclass(frozen=True)
class CrossfitSlice:
    heldout_fold: int
    train_indices: np.ndarray
    target_indices: np.ndarray
    train_folds: tuple[int, ...]


def load_config(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "soluprot-distillation-contract-v1":
        raise ValueError("Unsupported distillation config schema.")
    return payload


def resolve_project_path(config: dict[str, object], key: str) -> Path:
    root = Path(config["paths"]["project_root"])
    candidate = Path(config["paths"][key])
    return candidate if candidate.is_absolute() else root / candidate


def validate_input_hashes(config: dict[str, object], *, include_esm2: bool = False) -> None:
    path_key = {
        "development": "development",
        "transparent_features": "transparent_features",
        "feature_manifest": "feature_manifest",
        "embedding_index": "embedding_index",
        "prott5": "prott5",
        "external": "external",
        "external_features": "external_features",
    }
    for hash_key, expected in config["input_sha256"].items():
        path = resolve_project_path(config, path_key[hash_key])
        if not path.is_file():
            raise FileNotFoundError(f"Required input is absent: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Input hash mismatch for {hash_key}: expected={expected}, actual={actual}"
            )
    if include_esm2:
        validate_esm2_final(config)


def load_development(config: dict[str, object]) -> DevelopmentData:
    development_path = resolve_project_path(config, "development")
    feature_path = resolve_project_path(config, "transparent_features")
    manifest_path = resolve_project_path(config, "feature_manifest")
    index_path = resolve_project_path(config, "embedding_index")
    development = pd.read_csv(development_path, dtype={"record_id": str})
    features = pd.read_csv(feature_path, dtype={"record_id": str})
    index = pd.read_csv(index_path, dtype={"record_id": str})
    expected_rows = int(config["data_contract"]["development_rows"])
    expected_groups = int(config["data_contract"]["development_groups"])
    if len(development) != expected_rows or development.cluster_id.nunique() != expected_groups:
        raise ValueError("Development row/group cardinality changed.")
    if not np.array_equal(development.record_id.to_numpy(), features.record_id.to_numpy()):
        raise ValueError("Development and transparent-feature rows are misaligned.")
    development_index = index[index.analysis_split == "development"].reset_index(drop=True)
    if not np.array_equal(
        development.record_id.to_numpy(), development_index.record_id.to_numpy()
    ):
        raise ValueError("Development and PLM-embedding index rows are misaligned.")
    labels = development.label.to_numpy(dtype=np.int64)
    if np.bincount(labels, minlength=2).tolist() != list(
        config["data_contract"]["label_counts"]
    ):
        raise ValueError("Development label counts changed.")
    sequences = development.sequence.astype(str).tolist()
    alphabet = set(str(config["data_contract"]["canonical_alphabet"]))
    if any(not sequence or set(sequence) - alphabet for sequence in sequences):
        raise ValueError("Development contains a non-canonical sequence.")
    blocks = build_b0_b1(features, sequences, manifest_path)
    b0, b0_names = blocks["B0"]
    b1, b1_names = blocks["B1"]
    folds = development.development_fold.to_numpy(dtype=np.int64)
    groups = development.cluster_id.astype(str).to_numpy()
    if set(np.unique(folds).tolist()) != set(config["data_contract"]["outer_folds"]):
        raise ValueError("Development fold labels changed.")
    for group in np.unique(groups):
        if len(np.unique(folds[groups == group])) != 1:
            raise ValueError("A development group occurs in multiple outer folds.")
    return DevelopmentData(
        record_ids=development.record_id.to_numpy(dtype=str),
        sequences=sequences,
        labels=labels,
        groups=groups,
        folds=folds,
        b0=b0.astype(np.float32),
        b1=b1.astype(np.float32),
        b0_names=b0_names,
        b1_names=b1_names,
    )


def load_prott5(config: dict[str, object]) -> np.ndarray:
    array = np.load(resolve_project_path(config, "prott5"), mmap_mode="r")
    expected_rows = int(config["data_contract"]["development_rows"])
    expected_dimension = int(config["teacher"]["prott5"]["dimension"])
    if array.shape[0] < expected_rows or array.shape[1] != expected_dimension:
        raise ValueError(f"Unexpected ProtT5 array shape: {array.shape}")
    development = np.asarray(array[:expected_rows], dtype=np.float32)
    if not np.isfinite(development).all():
        raise ValueError("ProtT5 development embeddings contain non-finite values.")
    return development


def validate_esm2_final(config: dict[str, object]) -> dict[str, object]:
    array_path = resolve_project_path(config, "esm2")
    metadata_path = array_path.with_name("soluprot_esm2_metadata.json")
    if not array_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Final ESM-2 array/metadata are absent. Resume embedding extraction first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = config["teacher"]["hybrid"]
    configuration = metadata.get("configuration", {})
    required = {
        "checkpoint": expected["esm2_checkpoint"],
        "revision": expected["esm2_revision"],
    }
    for key, value in required.items():
        if configuration.get(key) != value:
            raise ValueError(f"ESM-2 metadata mismatch for {key}.")
    if metadata.get("pooling") != "residue-only arithmetic mean across every biological residue":
        raise ValueError("ESM-2 pooling contract mismatch.")
    if metadata.get("special_tokens_excluded") is not True:
        raise ValueError("ESM-2 special-token exclusion is not confirmed.")
    actual_hash = sha256_file(array_path)
    if metadata.get("output_sha256") != actual_hash:
        raise ValueError("ESM-2 array hash does not match metadata.")
    return metadata


def load_esm2(config: dict[str, object]) -> np.ndarray:
    validate_esm2_final(config)
    array = np.load(resolve_project_path(config, "esm2"), mmap_mode="r")
    expected_rows = int(config["data_contract"]["development_rows"])
    expected_dimension = int(config["teacher"]["hybrid"]["esm2_dimension"])
    if array.shape[0] < expected_rows or array.shape[1] != expected_dimension:
        raise ValueError(f"Unexpected ESM-2 array shape: {array.shape}")
    development = np.asarray(array[:expected_rows], dtype=np.float32)
    if not np.isfinite(development).all():
        raise ValueError("ESM-2 development embeddings contain non-finite values.")
    return development


def teacher_embedding(
    config: dict[str, object], teacher_name: str, prott5: np.ndarray, esm2: np.ndarray | None
) -> np.ndarray:
    if teacher_name == "prott5":
        return prott5
    if teacher_name != "hybrid":
        raise ValueError(f"Unknown teacher: {teacher_name}")
    if esm2 is None:
        raise ValueError("Hybrid teacher requires ESM-2 embeddings.")
    if len(prott5) != len(esm2):
        raise ValueError("ProtT5/ESM-2 cardinality mismatch.")
    hybrid = np.hstack([prott5, esm2]).astype(np.float32, copy=False)
    expected = int(config["teacher"]["hybrid"]["joint_embedding_dimension"])
    if hybrid.shape[1] != expected:
        raise ValueError("Hybrid embedding dimension mismatch.")
    return hybrid


def make_crossfit_plan(
    folds: np.ndarray, groups: np.ndarray, train_folds: Sequence[int]
) -> list[CrossfitSlice]:
    folds = np.asarray(folds, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    allowed = tuple(sorted(int(fold) for fold in train_folds))
    if len(set(allowed)) != len(allowed) or len(allowed) < 2:
        raise ValueError("Cross-fitting requires at least two distinct train folds.")
    if not set(allowed).issubset(set(np.unique(folds).tolist())):
        raise ValueError("A requested train fold is absent.")
    plan: list[CrossfitSlice] = []
    for heldout in allowed:
        target = np.flatnonzero(folds == heldout)
        train = np.flatnonzero(np.isin(folds, allowed) & (folds != heldout))
        if len(target) == 0 or len(train) == 0:
            raise ValueError("Crossfit slice has an empty train or target set.")
        assert_no_group_overlap(groups[train], groups[target])
        plan.append(
            CrossfitSlice(
                heldout_fold=heldout,
                train_indices=train,
                target_indices=target,
                train_folds=tuple(fold for fold in allowed if fold != heldout),
            )
        )
    covered = np.concatenate([item.target_indices for item in plan])
    expected = np.flatnonzero(np.isin(folds, allowed))
    if not np.array_equal(np.sort(covered), expected):
        raise RuntimeError("Crossfit plan does not cover each training row exactly once.")
    return plan


def crossfit_plan_manifest(
    data: DevelopmentData, train_folds: Sequence[int]
) -> dict[str, object]:
    plan = make_crossfit_plan(data.folds, data.groups, train_folds)
    slices = []
    for item in plan:
        slices.append(
            {
                "heldout_fold": item.heldout_fold,
                "train_folds": list(item.train_folds),
                "n_train_rows": int(len(item.train_indices)),
                "n_target_rows": int(len(item.target_indices)),
                "n_train_groups": int(len(np.unique(data.groups[item.train_indices]))),
                "n_target_groups": int(len(np.unique(data.groups[item.target_indices]))),
                "group_overlap": 0,
            }
        )
    return {
        "train_folds": sorted(int(value) for value in train_folds),
        "slices": slices,
        "plan_hash": stable_hash(slices),
    }


def _checkpoint_contract(
    *,
    run_id: str,
    stage: str,
    teacher_name: str,
    heldout_fold: int | str,
    train_folds: Sequence[int],
    target_indices: np.ndarray,
    embedding_source_hash: str,
    config: dict[str, object],
) -> str:
    payload = {
        "run_id": run_id,
        "stage": stage,
        "teacher_name": teacher_name,
        "heldout_fold": heldout_fold,
        "train_folds": sorted(int(value) for value in train_folds),
        "target_index_hash": stable_hash(np.asarray(target_indices, dtype=int).tolist()),
        "pls_components": int(config["teacher"]["pls_components"]),
        "logistic_c": float(config["teacher"]["logistic_c"]),
        "development_sha256": config["input_sha256"]["development"],
        "prott5_sha256": config["input_sha256"]["prott5"],
        "embedding_source_hash": str(embedding_source_hash),
    }
    return stable_hash(payload)


def _load_prediction_checkpoint(
    path: Path, expected_contract: str, expected_indices: np.ndarray
) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as checkpoint:
        contract = str(checkpoint["contract_hash"][0])
        indices = checkpoint["row_indices"].astype(np.int64)
        probabilities = checkpoint["probabilities"].astype(np.float64)
    if contract != expected_contract:
        raise ValueError(f"Existing checkpoint contract mismatch: {path}")
    if not np.array_equal(indices, np.asarray(expected_indices, dtype=np.int64)):
        raise ValueError(f"Existing checkpoint row-index mismatch: {path}")
    if probabilities.shape != (len(indices),) or not np.isfinite(probabilities).all():
        raise ValueError(f"Existing checkpoint probabilities are invalid: {path}")
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise ValueError(f"Existing checkpoint probabilities are outside (0,1): {path}")
    return probabilities


def fixed_teacher_crossfit(
    data: DevelopmentData,
    embedding: np.ndarray,
    train_folds: Sequence[int],
    *,
    config: dict[str, object],
    run_root: Path,
    stage: str,
    teacher_name: str,
    embedding_source_hash: str,
) -> np.ndarray:
    """Cross-fit teacher targets using only the designated training folds."""
    train_folds = tuple(sorted(int(value) for value in train_folds))
    output = np.full(len(data.labels), np.nan, dtype=np.float64)
    for item in make_crossfit_plan(data.folds, data.groups, train_folds):
        checkpoint = (
            Path(run_root)
            / "checkpoints"
            / stage
            / teacher_name
            / f"crossfit_fold_{item.heldout_fold}.npz"
        )
        contract = _checkpoint_contract(
            run_id=str(config["run_id"]),
            stage=stage,
            teacher_name=teacher_name,
            heldout_fold=item.heldout_fold,
            train_folds=item.train_folds,
            target_indices=item.target_indices,
            embedding_source_hash=embedding_source_hash,
            config=config,
        )
        probabilities = _load_prediction_checkpoint(
            checkpoint, contract, item.target_indices
        )
        if probabilities is None:
            head = fit_head(
                data.b1[item.train_indices],
                embedding[item.train_indices],
                data.labels[item.train_indices],
                "pls",
                c_value=float(config["teacher"]["logistic_c"]),
                pls_k=int(config["teacher"]["pls_components"]),
            )
            probabilities = predict_head(
                head, data.b1[item.target_indices], embedding[item.target_indices]
            )
            probabilities = np.clip(
                probabilities,
                float(config["teacher"]["probability_clip"][0]),
                float(config["teacher"]["probability_clip"][1]),
            )
            probabilities = probabilities.astype(np.float32).astype(np.float64)
            atomic_npz(
                checkpoint,
                contract_hash=np.asarray([contract]),
                row_indices=item.target_indices.astype(np.int64),
                probabilities=probabilities.astype(np.float32),
            )
        probabilities = probabilities.astype(np.float32).astype(np.float64)
        output[item.target_indices] = probabilities
    expected = np.isin(data.folds, train_folds)
    if not np.isfinite(output[expected]).all() or np.isfinite(output[~expected]).any():
        raise RuntimeError("Teacher crossfit output coverage failed.")
    return output


def fixed_teacher_evaluation(
    data: DevelopmentData,
    embedding: np.ndarray,
    train_folds: Sequence[int],
    evaluation_fold: int,
    *,
    config: dict[str, object],
    run_root: Path,
    stage: str,
    teacher_name: str,
    embedding_source_hash: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one teacher on all stage-training folds and score one held-out fold."""
    train = np.flatnonzero(np.isin(data.folds, list(train_folds)))
    evaluation = np.flatnonzero(data.folds == int(evaluation_fold))
    if int(evaluation_fold) in set(int(value) for value in train_folds):
        raise ValueError("Evaluation fold is included in training folds.")
    assert_no_group_overlap(data.groups[train], data.groups[evaluation])
    checkpoint = (
        Path(run_root)
        / "checkpoints"
        / stage
        / teacher_name
        / f"evaluation_fold_{int(evaluation_fold)}.npz"
    )
    contract = _checkpoint_contract(
        run_id=str(config["run_id"]),
        stage=stage,
        teacher_name=teacher_name,
        heldout_fold=f"evaluation_{int(evaluation_fold)}",
        train_folds=train_folds,
        target_indices=evaluation,
        embedding_source_hash=embedding_source_hash,
        config=config,
    )
    probabilities = _load_prediction_checkpoint(checkpoint, contract, evaluation)
    if probabilities is None:
        head = fit_head(
            data.b1[train],
            embedding[train],
            data.labels[train],
            "pls",
            c_value=float(config["teacher"]["logistic_c"]),
            pls_k=int(config["teacher"]["pls_components"]),
        )
        probabilities = predict_head(
            head, data.b1[evaluation], embedding[evaluation]
        )
        probabilities = np.clip(
            probabilities,
            float(config["teacher"]["probability_clip"][0]),
            float(config["teacher"]["probability_clip"][1]),
        )
        probabilities = probabilities.astype(np.float32).astype(np.float64)
        atomic_npz(
            checkpoint,
            contract_hash=np.asarray([contract]),
            row_indices=evaluation.astype(np.int64),
            probabilities=probabilities.astype(np.float32),
        )
    return evaluation, probabilities.astype(np.float32).astype(np.float64)
