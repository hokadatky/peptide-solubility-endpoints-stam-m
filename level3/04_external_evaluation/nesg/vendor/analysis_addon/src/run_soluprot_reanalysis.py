"""Run the historical bridge, primary internal reanalysis, and external audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from .clustered_stats import (
        calibration_intercept_slope,
        cluster_swap_permutation,
        decision_flags,
        metric_bundle,
        paired_cluster_bootstrap,
        reliability_table,
    )
    from .io_contracts import (
        assert_finite_matrix,
        assert_forbidden_columns_absent,
        atomic_write_json,
        one_to_one_join,
        sha256_file,
        validate_binary_labels,
        validate_canonical_sequence,
        validate_unique_ids,
    )
    from .nested_models import (
        fit_external_development_seal,
        fit_head,
        nested_oof,
        predict_head,
    )
    from .negative_controls import training_label_permutation_control
    from .resumable_models import checkpointed_nested_oof
    from .sequence_features import build_b0_b1, primary_feature_names
    from .split_contracts import (
        balanced_group_folds,
        fixed_group_folds,
        fold_hash,
        indices_for_fold,
        validate_group_folds,
    )
    from .stage_utils import refuse_existing, write_stage
except ImportError:  # pragma: no cover - direct execution path
    from clustered_stats import calibration_intercept_slope, cluster_swap_permutation, decision_flags, metric_bundle, paired_cluster_bootstrap, reliability_table
    from io_contracts import assert_finite_matrix, assert_forbidden_columns_absent, atomic_write_json, one_to_one_join, sha256_file, validate_binary_labels, validate_canonical_sequence, validate_unique_ids
    from nested_models import fit_external_development_seal, fit_head, nested_oof, predict_head
    from negative_controls import training_label_permutation_control
    from resumable_models import checkpointed_nested_oof
    from sequence_features import build_b0_b1, primary_feature_names
    from split_contracts import balanced_group_folds, fixed_group_folds, fold_hash, indices_for_fold, validate_group_folds
    from stage_utils import refuse_existing, write_stage


AAC20 = [f"aac_{letter}" for letter in "ACDEFGHIKLMNPQRSTVWY"]
HISTORICAL_PHYS = [
    "length", "molecular_weight_da", "isoelectric_point", "charge_at_ph7",
    "charge_density_ph7", "instability_index", "normalized_sequence_entropy",
    "max_kd_window9", "helix_propensity_fraction", "turn_propensity_fraction",
    "sheet_propensity_fraction", "whole_sequence_gravy", "aromaticity_fraction",
    "aliphatic_index",
]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"record_id": str, "source_id": str})


def _align_features(source: pd.DataFrame, features: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    validate_unique_ids(source, "record_id")
    validate_unique_ids(features, "record_id")
    selected = features[["record_id"] + names].copy()
    aligned = one_to_one_join(source[["record_id"]], selected, "record_id")
    output = aligned[names]
    assert_finite_matrix(output.to_numpy(dtype=float), "aligned transparent features")
    return output.reset_index(drop=True)


def assemble(config: dict) -> dict[str, object]:
    data_root = Path(config["paths"]["data_root"])
    train = _read_csv(data_root / "soluprot_train_partitions.csv")
    external = _read_csv(data_root / "soluprot_external_test_clustered.csv")
    train_features = _read_csv(data_root / "solubility_transparent_features_development.csv")
    external_features = _read_csv(data_root / "solubility_transparent_features_external.csv")
    validate_unique_ids(train, "record_id")
    validate_unique_ids(external, "record_id")
    validate_binary_labels(train, "label")
    validate_binary_labels(external, "label")
    expected = config["soluprot"]
    if len(train) != int(expected["development_rows_expected"]):
        raise ValueError("Development row count mismatch.")
    if len(external) != int(expected["external_rows_expected"]):
        raise ValueError("External row count mismatch.")
    if train["cluster_id"].nunique() != int(expected["development_groups_expected_30pct"]):
        raise ValueError("Development cluster count mismatch.")
    if external["external_cluster_id"].nunique() != int(expected["external_groups_expected"]):
        raise ValueError("External cluster count mismatch.")
    for sequence in pd.concat([train["sequence"], external["sequence"]], ignore_index=True):
        validate_canonical_sequence(sequence)
    manifest_path = data_root / "solubility_transparent_features.json"
    b0_names = primary_feature_names(manifest_path, expected["transparent_manifest_group"])
    dev_b0 = _align_features(train, train_features, b0_names)
    ext_b0 = _align_features(external, external_features, b0_names)
    all_source = pd.concat([train, external], ignore_index=True)
    all_features = pd.concat([dev_b0, ext_b0], ignore_index=True)
    all_sequences = all_source["sequence"].tolist()
    blocks = build_b0_b1(all_features, all_sequences, manifest_path, expected["ordered_dimer_alphabet"])
    index = pd.read_csv(data_root / "embeddings/soluprot_embedding_index.csv", dtype={"record_id": str})
    validate_unique_ids(index, "record_id")
    expected_ids = all_source["record_id"].astype(str).to_numpy()
    if not np.array_equal(index["record_id"].astype(str).to_numpy(), expected_ids):
        raise ValueError("Embedding index order does not equal development-then-external source order.")
    embeddings = np.load(data_root / "embeddings/soluprot_prott5.npy", mmap_mode="r")
    if embeddings.shape != (len(all_source), int(expected["prott5_dimension"])):
        raise ValueError(f"ProtT5 array shape mismatch: {embeddings.shape}")
    embeddings = np.asarray(embeddings, dtype=np.float64)
    assert_finite_matrix(embeddings, "ProtT5 embeddings")
    for name in b0_names + blocks["B1"][1][len(b0_names):]:
        assert_forbidden_columns_absent([name])
    primary_folds = fixed_group_folds(
        train["label"].to_numpy(),
        train[expected["primary_group_column"]].astype(str).to_numpy(),
        train[expected["primary_outer_fold_column"]].to_numpy(),
        int(expected["primary_outer_folds"]),
    )
    validate_group_folds(
        train["label"].to_numpy(),
        train[expected["primary_group_column"]].astype(str).to_numpy(),
        primary_folds,
        int(expected["primary_outer_folds"]),
    )
    return {
        "train": train,
        "external": external,
        "all_source": all_source,
        "all_features": all_features,
        "b0": blocks["B0"][0],
        "b1": blocks["B1"][0],
        "b0_names": blocks["B0"][1],
        "b1_names": blocks["B1"][1],
        "embeddings": embeddings,
        "primary_folds": primary_folds,
    }


def _save_predictions(run_dir: Path, filename: str, frame: pd.DataFrame) -> Path:
    path = run_dir / "predictions" / filename
    refuse_existing(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _save_predictions_resumable(run_dir: Path, filename: str, frame: pd.DataFrame) -> Path:
    """Write once, or accept an already-written byte-equivalent table."""
    path = run_dir / "predictions" / filename
    if path.exists():
        observed = pd.read_parquet(path)
        try:
            pd.testing.assert_frame_equal(observed, frame, check_exact=True, check_dtype=True)
        except AssertionError as error:
            raise FileExistsError(f"Existing resumable prediction differs: {path}") from error
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path


def _save_raw(run_dir: Path, filename: str, payload: dict) -> Path:
    path = run_dir / "raw_results" / filename
    refuse_existing(path)
    atomic_write_json(payload, path)
    return path


def _arm_metrics(y: np.ndarray, score: np.ndarray, groups: np.ndarray) -> dict[str, object]:
    return {
        "metrics": metric_bundle(y, score),
        "calibration": calibration_intercept_slope(y, score),
        "reliability": reliability_table(y, score),
        "n_rows": int(len(y)),
        "n_groups": int(len(np.unique(groups))),
        "largest_group_rows": int(pd.Series(groups).value_counts().max()),
        "singleton_group_fraction": float(np.mean(pd.Series(groups).value_counts().to_numpy() == 1)),
    }


def _primary_contrast(y: np.ndarray, ref: np.ndarray, cand: np.ndarray, groups: np.ndarray, margin: float, replicates: int, seed: int) -> dict[str, object]:
    bootstrap = paired_cluster_bootstrap(y, ref, cand, groups, "roc_auc", replicates, seed)
    flags = decision_flags(bootstrap["ci90"], bootstrap["ci95"], margin)
    bootstrap["practical_margin"] = float(margin)
    bootstrap["decision_flags"] = flags
    bootstrap["mde80_approx"] = float((1.959963984540054 + 0.8416212335729143) * bootstrap["bootstrap_sd"])
    bootstrap["mde_method"] = "normal approximation using SD of paired cluster-bootstrap gain"
    bootstrap["alpha_two_sided"] = 0.05
    bootstrap["target_power"] = 0.80
    bootstrap["retrospective_sensitivity_not_prospective_power"] = True
    return bootstrap


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.zeros(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * float(values[index])))
        adjusted[index] = running
    return adjusted.tolist()


def _fold_metrics(y: np.ndarray, scores: np.ndarray, folds: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold
        rows.append({"fold": int(fold), "n_rows": int(mask.sum()), **metric_bundle(y[mask], scores[mask])})
    return rows


def _read_recovered_oof(path: Path, train: pd.DataFrame, expected_folds: np.ndarray) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"record_id", "cluster_id", "fold", "label", "score"}
    if not required.issubset(frame.columns) or len(frame) != len(train):
        raise ValueError(f"Recovered OOF schema/row count failed: {path}")
    checks = (
        np.array_equal(frame["record_id"].astype(str).to_numpy(), train["record_id"].astype(str).to_numpy()),
        np.array_equal(frame["cluster_id"].astype(str).to_numpy(), train["cluster_id"].astype(str).to_numpy()),
        np.array_equal(frame["label"].to_numpy(dtype=int), train["label"].to_numpy(dtype=int)),
        np.array_equal(frame["fold"].to_numpy(dtype=int), np.asarray(expected_folds).astype(int)),
    )
    if not all(checks):
        raise ValueError(f"Recovered OOF row alignment failed: {path}")
    score = frame["score"].to_numpy(dtype=float)
    if not np.isfinite(score).all() or np.any((score < 0.0) | (score > 1.0)):
        raise ValueError(f"Recovered OOF score contract failed: {path}")
    return score


def _save_external_checkpoint(
    path: Path,
    contract: str,
    raw: np.ndarray,
    calibrated: np.ndarray,
    selected: tuple[int | None, float],
    selection_info: list[dict[str, object]],
    calibration_n: int,
    development_seal_diagnostics: dict[str, object],
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite external checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    metadata = {
        "selected": [selected[0], float(selected[1])],
        "selection_info": selection_info,
        "calibration_n": int(calibration_n),
        "development_seal_diagnostics": development_seal_diagnostics,
    }
    raw = np.asarray(raw, dtype=np.float64)
    calibrated = np.asarray(calibrated, dtype=np.float64)
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256()
    digest.update(contract.encode("utf-8"))
    digest.update(str(raw.shape).encode("ascii"))
    digest.update(raw.tobytes())
    digest.update(str(calibrated.shape).encode("ascii"))
    digest.update(calibrated.tobytes())
    digest.update(metadata_json.encode("utf-8"))
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            contract_json=np.asarray([contract]),
            raw=raw,
            calibrated=calibrated,
            metadata_json=np.asarray([metadata_json]),
            artifact_sha256=np.asarray([digest.hexdigest()]),
        )
    os.replace(temporary, path)


def _load_external_checkpoint(path: Path, contract: str, n_external: int) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as payload:
        if "artifact_sha256" not in payload.files:
            raise ValueError(f"External checkpoint schema/integrity field missing: {path}")
        observed_contract = str(payload["contract_json"][0])
        raw = np.asarray(payload["raw"], dtype=np.float64)
        calibrated = np.asarray(payload["calibrated"], dtype=np.float64)
        metadata = json.loads(str(payload["metadata_json"][0]))
        observed_artifact_sha256 = str(payload["artifact_sha256"][0])
    if observed_contract != contract:
        raise ValueError(f"External checkpoint contract mismatch: {path}")
    for name, values in (("raw", raw), ("calibrated", calibrated)):
        if values.shape != (n_external,) or not np.isfinite(values).all():
            raise ValueError(f"External checkpoint {name} shape/finiteness failed: {path}")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"External checkpoint {name} probabilities outside [0, 1]: {path}")
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256()
    digest.update(observed_contract.encode("utf-8"))
    digest.update(str(raw.shape).encode("ascii"))
    digest.update(raw.tobytes())
    digest.update(str(calibrated.shape).encode("ascii"))
    digest.update(calibrated.tobytes())
    digest.update(metadata_json.encode("utf-8"))
    if digest.hexdigest() != observed_artifact_sha256:
        raise ValueError(f"External checkpoint artifact digest mismatch: {path}")
    return {"raw": raw, "calibrated": calibrated, **metadata}


def _run_internal_recovery(config: dict, run_dir: Path, data: dict[str, object]) -> dict[str, object]:
    """V3 primary-first internal analysis with validated v2 secondary imports."""
    train = data["train"]
    labels = train["label"].to_numpy().astype(int)
    groups = train["cluster_id"].astype(str).to_numpy()
    folds = np.asarray(data["primary_folds"]).astype(int)
    n = len(train)
    b0 = np.asarray(data["b0"][:n], dtype=np.float64)
    b1 = np.asarray(data["b1"][:n], dtype=np.float64)
    embeddings = np.asarray(data["embeddings"][:n], dtype=np.float64)
    c_grid = [float(value) for value in config["soluprot"]["logistic_c_grid"]]
    k_grid = [int(value) for value in config["soluprot"]["pls_k_grid"]]
    inner_folds = int(config["soluprot"]["inner_folds"])
    seed = int(config["global_seed"])
    recovery = config["recovery"]
    if not (run_dir / "manifests/recovery-v2.json").is_file():
        raise FileNotFoundError("Validated v2 recovery stage is required for the v3 internal analysis.")

    primary_specs = {
        "B1_ORDER_LOGIT": (None, "none"),
        "B1_ORDER_PLUS_PROTT5_PLS_LOGIT": (embeddings, "pls"),
    }
    scores: dict[str, np.ndarray] = {}
    info: dict[str, object] = {}
    private_paths: list[str] = []
    for arm, (embedding, mode) in primary_specs.items():
        score, fold_info = checkpointed_nested_oof(
            b1,
            embedding,
            labels,
            groups,
            folds,
            mode,
            c_grid,
            k_grid,
            inner_folds,
            seed,
            run_dir / "checkpoints/internal_primary" / arm,
            arm,
            f"{config['run_id']}:internal-primary:{arm}",
        )
        scores[arm] = score
        info[arm] = fold_info
        frame = pd.DataFrame(
            {
                "record_id": train["record_id"].astype(str),
                "cluster_id": groups,
                "fold": folds,
                "label": labels,
                "score": score,
            }
        )
        private_paths.append(
            str(_save_predictions_resumable(run_dir, f"soluprot_dev_oof_{arm}.parquet", frame))
        )

    imported_arms = list(recovery["import_secondary_oof"])
    for arm in imported_arms:
        path = run_dir / "predictions" / f"soluprot_dev_oof_{arm}.parquet"
        scores[arm] = _read_recovered_oof(path, train, folds)
        private_paths.append(str(path))
        info[arm] = {
            "status": "OOF_PREDICTIONS_RECOVERED_FROM_V2",
            "selected_hyperparameters": "not recoverable because v2 was interrupted before aggregate metadata write",
            "prediction_sha256": sha256_file(path),
        }

    source_reference_path = run_dir / "recovery_imports" / "soluprot_dev_oof_B1_ORDER_LOGIT_v2.parquet"
    source_reference = _read_recovered_oof(source_reference_path, train, folds)
    reference_difference = np.abs(source_reference - scores["B1_ORDER_LOGIT"])
    reference_parity = {
        "source_run_id": recovery["source_run_id"],
        "source_prediction_sha256": sha256_file(source_reference_path),
        "max_absolute_probability_difference": float(reference_difference.max()),
        "mean_absolute_probability_difference": float(reference_difference.mean()),
        "absolute_tolerance": 1e-12,
        "passed": bool(reference_difference.max() <= 1e-12),
    }
    if not reference_parity["passed"]:
        raise RuntimeError(f"Recomputed B1 reference differs from v2: {reference_parity}")

    metrics: dict[str, object] = {}
    base_dimensions = {
        "B0_TRANSPARENT_LOGIT": int(b0.shape[1]),
        "B1_ORDER_LOGIT": int(b1.shape[1]),
        "B0_PLUS_PROTT5_PLS_LOGIT": int(b0.shape[1]),
        "PROTT5_PLS_ONLY_LOGIT": 0,
        "B1_ORDER_PLUS_PROTT5_PLS_LOGIT": int(b1.shape[1]),
    }
    for arm, score in scores.items():
        arm_result = {
            **_arm_metrics(labels, score, groups),
            "fold_metrics": _fold_metrics(labels, score, folds),
            "base_feature_dimension": base_dimensions[arm],
        }
        if arm == "B1_ORDER_PLUS_PROTT5_PLS_LOGIT":
            arm_result["selected_pls_k_by_outer_fold"] = [
                int(row["selected_pls_k"]) for row in info[arm]
            ]
        elif arm in {"B0_PLUS_PROTT5_PLS_LOGIT", "PROTT5_PLS_ONLY_LOGIT"}:
            arm_result["selected_pls_k_by_outer_fold"] = "unavailable_from_interrupted_v2_metadata"
        metrics[arm] = arm_result

    deferred_arms = list(recovery["deferred_secondary_arms"])
    for arm in deferred_arms:
        metrics[arm] = {
            "status": "DEFERRED_COMPUTE_LIMIT_NOT_EVALUATED",
            "reason": recovery["reason"],
            "eligible_for_claim": False,
        }
        info[arm] = {"status": "NOT_FIT", "reason": recovery["reason"]}

    reference = scores["B1_ORDER_LOGIT"]
    primary_candidate = scores["B1_ORDER_PLUS_PROTT5_PLS_LOGIT"]
    contrast = _primary_contrast(
        labels,
        reference,
        primary_candidate,
        groups,
        float(config["statistics"]["auc_gain_margin"]),
        int(config["statistics"]["bootstrap_replicates_final"]),
        int(config["statistics"]["bootstrap_seed"]),
    )

    planned_secondary = imported_arms + deferred_arms
    if len(planned_secondary) != int(recovery["planned_secondary_family_size"]):
        raise ValueError("Recovered/deferred secondary family size differs from the amendment.")
    secondary: list[dict[str, object]] = []
    secondary_p: list[float] = []
    for index, arm in enumerate(planned_secondary, start=1):
        permutation_seed = int(config["statistics"]["bootstrap_seed"]) + index
        if arm in scores:
            boot = _primary_contrast(
                labels,
                reference,
                scores[arm],
                groups,
                float(config["statistics"]["auc_gain_margin"]),
                int(config["statistics"]["bootstrap_replicates_final"]),
                permutation_seed,
            )
            permutation = cluster_swap_permutation(
                labels, reference, scores[arm], groups, "roc_auc", 999, permutation_seed
            )
            p_value = float(permutation["p_value_two_sided"])
            row = {
                "status": "completed_recovered_v2_oof",
                "reference": "B1_ORDER_LOGIT",
                "candidate": arm,
                "gain_definition": "AUC(candidate) - AUC(reference)",
                "bootstrap": boot,
                "cluster_swap_permutation": permutation,
            }
        else:
            p_value = float(recovery["deferred_arm_multiplicity_p_value"])
            row = {
                "status": "DEFERRED_COMPUTE_LIMIT_NOT_EVALUATED",
                "reference": "B1_ORDER_LOGIT",
                "candidate": arm,
                "eligible_for_claim": False,
                "reason": recovery["reason"],
                "cluster_swap_permutation": {
                    "status": "not_computed",
                    "p_value_two_sided_for_multiplicity": p_value,
                },
            }
        secondary.append(row)
        secondary_p.append(p_value)
    for row, value in zip(secondary, _holm(secondary_p)):
        row["holm_adjusted_p_value"] = float(value)
        row["holm_family_size"] = int(len(planned_secondary))

    stability = []
    for stability_seed in config["soluprot"]["fold_stability_seeds"]:
        stability_seed = int(stability_seed)
        stability_folds = balanced_group_folds(
            labels, groups, int(config["soluprot"]["primary_outer_folds"]), stability_seed
        )
        root = run_dir / "checkpoints/fold_stability" / f"seed_{stability_seed}"
        ref_stability, ref_stability_info = checkpointed_nested_oof(
            b1,
            None,
            labels,
            groups,
            stability_folds,
            "none",
            c_grid,
            k_grid,
            inner_folds,
            stability_seed,
            root / "reference",
            "B1_ORDER_LOGIT",
            f"{config['run_id']}:fold-stability:{stability_seed}:reference",
        )
        cand_stability, cand_stability_info = checkpointed_nested_oof(
            b1,
            embeddings,
            labels,
            groups,
            stability_folds,
            "pls",
            c_grid,
            k_grid,
            inner_folds,
            stability_seed,
            root / "candidate",
            "B1_ORDER_PLUS_PROTT5_PLS_LOGIT",
            f"{config['run_id']}:fold-stability:{stability_seed}:candidate",
        )
        ref_auc = float(metric_bundle(labels, ref_stability)["roc_auc"])
        cand_auc = float(metric_bundle(labels, cand_stability)["roc_auc"])
        stability.append(
            {
                "seed": stability_seed,
                "reference_auc": ref_auc,
                "candidate_auc": cand_auc,
                "auc_gain_candidate_minus_reference": cand_auc - ref_auc,
                "fold_hash": fold_hash(groups, stability_folds),
                "reference_selected": [
                    {"outer_fold": row["outer_fold"], "c": row["selected_c"]}
                    for row in ref_stability_info
                ],
                "candidate_selected": [
                    {
                        "outer_fold": row["outer_fold"],
                        "pls_k": row["selected_pls_k"],
                        "c": row["selected_c"],
                    }
                    for row in cand_stability_info
                ],
            }
        )

    label_permutation = training_label_permutation_control(
        b1,
        embeddings,
        labels,
        groups,
        folds,
        info["B1_ORDER_LOGIT"],
        info["B1_ORDER_PLUS_PROTT5_PLS_LOGIT"],
        int(config["soluprot"]["label_permutation_replicates"]),
        int(config["soluprot"]["label_permutation_seed"]),
        run_dir / "checkpoints/negative_control",
        f"{config['run_id']}:internal-primary:training-label-permutation",
    )
    status = "completed" if label_permutation["status"] == "PASSED" else "PIPELINE_INVALID"
    result = {
        "status": status,
        "analysis_scope": "retrospective_data_informed",
        "endpoint": "recombinant soluble-expression status in Escherichia coli; not direct aqueous solubility",
        "input_hashes": {
            "train_partitions": sha256_file(Path(config["paths"]["project_root"]) / "data/solubility/soluprot_train_partitions.csv"),
            "development_features": sha256_file(Path(config["paths"]["project_root"]) / "data/solubility/solubility_transparent_features_development.csv"),
            "embedding_index": sha256_file(Path(config["paths"]["project_root"]) / "data/solubility/embeddings/soluprot_embedding_index.csv"),
            "prott5_array": sha256_file(Path(config["paths"]["project_root"]) / "data/solubility/embeddings/soluprot_prott5.npy"),
        },
        "split": {
            "group_definition": "existing 30% MMseqs2 cluster_id",
            "n_rows": int(n),
            "n_groups": int(len(np.unique(groups))),
            "largest_group_rows": int(pd.Series(groups).value_counts().max()),
            "singleton_group_fraction": float(np.mean(pd.Series(groups).value_counts().to_numpy() == 1)),
            "fold_hash": fold_hash(groups, folds),
            "outer_folds": int(config["soluprot"]["primary_outer_folds"]),
            "inner_folds": inner_folds,
            "group_overlap_checks": "passed",
        },
        "primary_contrast": {
            "reference": "B1_ORDER_LOGIT",
            "candidate": "B1_ORDER_PLUS_PROTT5_PLS_LOGIT",
            "metric": "roc_auc",
            "gain_definition": "AUC(candidate) - AUC(reference)",
            **contrast,
        },
        "arms": metrics,
        "selected_hyperparameters": info,
        "fold_stability": stability,
        "secondary": secondary,
        "negative_controls": {
            "alignment_guard": "passed by synthetic tests",
            "forbidden_feature_guard": "passed by synthetic tests",
            "training_label_permutation": label_permutation,
        },
        "recovery": {
            "source_run_id": recovery["source_run_id"],
            "reference_recompute_parity": reference_parity,
            "imported_secondary_arms": imported_arms,
            "deferred_secondary_arms": deferred_arms,
            "planned_holm_family_preserved": True,
        },
        "limitations": [
            "The raw high-dimensional fusion secondary arm was not completed and cannot support a claim.",
            "Imported v2 secondary OOF predictions lack selected-hyperparameter metadata because interruption preceded aggregate metadata writing.",
        ],
        "private_prediction_files": sorted(set(private_paths)),
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
    }
    _save_raw(run_dir, "internal_primary.json", result)
    return result


def run_historical_bridge(config: dict, run_dir: Path, data: dict[str, object]) -> dict[str, object]:
    train = data["train"]
    labels = train["label"].to_numpy().astype(int)
    groups = train["cluster_id"].astype(str).to_numpy()
    # The current BASE used the full descriptive physicochemical block plus AAC20.
    # This is deliberately kept separate from the B0/B1 primary reference.
    # Re-read once for the historical descriptive block, then align by source order.
    dev_features = _read_csv(Path(config["paths"]["data_root"]) / "solubility_transparent_features_development.csv")
    hist = dev_features[HISTORICAL_PHYS + AAC20].fillna(0.0).to_numpy(dtype=np.float64)
    prott5 = data["embeddings"][: len(train)]
    split = list(GroupKFold(n_splits=int(config["historical_bridge"]["outer_folds"])).split(hist, labels, groups))
    p_base = np.zeros(len(labels), dtype=float)
    p_pls = np.zeros(len(labels), dtype=float)
    for tr, te in split:
        learner = lambda: HistGradientBoostingClassifier(
            max_iter=int(config["historical_bridge"]["max_iter"]),
            learning_rate=float(config["historical_bridge"]["learning_rate"]),
            max_depth=int(config["historical_bridge"]["max_depth"]),
            early_stopping=True,
            validation_fraction=0.1,
            random_state=int(config["historical_bridge"]["random_state"]),
        )
        p_base[te] = learner().fit(hist[tr], labels[tr]).predict_proba(hist[te])[:, 1]
        emb_scaler = StandardScaler().fit(prott5[tr])
        pls = PLSRegression(n_components=int(config["historical_bridge"]["pls_k"]), max_iter=500).fit(
            emb_scaler.transform(prott5[tr]), labels[tr].astype(float)
        )
        train_matrix = np.hstack([hist[tr], pls.transform(emb_scaler.transform(prott5[tr]))])
        test_matrix = np.hstack([hist[te], pls.transform(emb_scaler.transform(prott5[te]))])
        p_pls[te] = learner().fit(train_matrix, labels[tr]).predict_proba(test_matrix)[:, 1]
    ref_path = _save_predictions(run_dir, "soluprot_dev_oof_H0_HIST_BASE.parquet", pd.DataFrame({"label": labels, "cluster_id": groups, "fold": [next(i for i, (_, te) in enumerate(split) if index in te) for index in range(len(labels))], "score": p_base}))
    cand_path = _save_predictions(run_dir, "soluprot_dev_oof_H1_HIST_BASE_PLUS_PROTT5_PLS3.parquet", pd.DataFrame({"label": labels, "cluster_id": groups, "fold": [next(i for i, (_, te) in enumerate(split) if index in te) for index in range(len(labels))], "score": p_pls}))
    contrast = _primary_contrast(labels, p_base, p_pls, groups, float(config["statistics"]["auc_gain_margin"]), int(config["statistics"]["bootstrap_replicates_final"]), int(config["statistics"]["bootstrap_seed"]))
    result = {
        "status": "completed",
        "analysis_scope": "retrospective_data_informed",
        "endpoint": "recombinant soluble-expression status in Escherichia coli; not direct aqueous solubility",
        "historical_bridge": {"reference": "H0_CURRENT_BASE_HISTGBM", "candidate": "H1_CURRENT_BASE_PLUS_PROTT5_PLS3_HISTGBM", "metric": "roc_auc", "reference_metrics": _arm_metrics(labels, p_base, groups), "candidate_metrics": _arm_metrics(labels, p_pls, groups), "primary_contrast": contrast, "split": {"n_rows": len(labels), "n_groups": len(np.unique(groups)), "outer_folds": 5, "group_definition": "existing 30% cluster_id", "fold_hash": fold_hash(groups, np.asarray([next(i for i, (_, te) in enumerate(split) if index in te) for index in range(len(labels))]))}},
        "private_prediction_files": [str(ref_path), str(cand_path)],
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
    }
    _save_raw(run_dir, "historical_bridge.json", result)
    return result


def run_internal(config: dict, run_dir: Path, data: dict[str, object]) -> dict[str, object]:
    if "recovery" in config:
        return _run_internal_recovery(config, run_dir, data)
    train = data["train"]
    labels = train["label"].to_numpy().astype(int)
    groups = train["cluster_id"].astype(str).to_numpy()
    folds = np.asarray(data["primary_folds"]).astype(int)
    n = len(train)
    base_blocks = {
        "B0": np.asarray(data["b0"][:n], dtype=np.float64),
        "B1": np.asarray(data["b1"][:n], dtype=np.float64),
        "NONE": np.zeros((n, 0), dtype=np.float64),
    }
    embeddings = np.asarray(data["embeddings"][:n], dtype=np.float64)
    c_grid = [float(value) for value in config["soluprot"]["logistic_c_grid"]]
    k_grid = [int(value) for value in config["soluprot"]["pls_k_grid"]]
    inner_folds = int(config["soluprot"]["inner_folds"])
    seed = int(config["global_seed"])
    arms = {
        "B0_TRANSPARENT_LOGIT": ("B0", None, "none"),
        "B1_ORDER_LOGIT": ("B1", None, "none"),
        "B0_PLUS_PROTT5_PLS_LOGIT": ("B0", "prott5", "pls"),
        "PROTT5_PLS_ONLY_LOGIT": ("NONE", "prott5", "pls"),
        "B1_ORDER_PLUS_PROTT5_PLS_LOGIT": ("B1", "prott5", "pls"),
    }
    scores: dict[str, np.ndarray] = {}
    info: dict[str, list[dict[str, object]]] = {}
    private_paths: list[str] = []
    for arm, (base_key, embedding_key, mode) in arms.items():
        score, fold_info = checkpointed_nested_oof(
            base_blocks[base_key],
            embeddings if embedding_key else None,
            labels,
            groups,
            folds,
            mode,
            c_grid,
            k_grid,
            inner_folds,
            seed,
            run_dir / "checkpoints/internal_primary" / arm,
            arm,
            f"{config['run_id']}:internal-primary:{arm}",
        )
        scores[arm] = score
        info[arm] = fold_info
        frame = pd.DataFrame({"record_id": train["record_id"].astype(str), "cluster_id": groups, "fold": folds, "label": labels, "score": score})
        private_paths.append(str(_save_predictions_resumable(run_dir, f"soluprot_dev_oof_{arm}.parquet", frame)))

    metrics = {}
    for arm, (base_key, embedding_key, mode) in arms.items():
        metrics[arm] = {
            **_arm_metrics(labels, scores[arm], groups),
            "fold_metrics": _fold_metrics(labels, scores[arm], folds),
            "base_feature_dimension": int(base_blocks[base_key].shape[1]),
            "selected_pls_k_by_outer_fold": [row["selected_pls_k"] for row in info[arm]] if mode == "pls" else None,
        }
    deferred_arm = "B1_PLUS_PROTT5_RAW_LOGIT"
    metrics[deferred_arm] = {
        "status": "DEFERRED_COMPUTE_LIMIT_NOT_EVALUATED",
        "eligible_for_claim": False,
        "reason": config["execution"]["raw_fusion_defer_reason"],
    }
    info[deferred_arm] = {"status": "NOT_FIT", "eligible_for_claim": False}

    reference = scores["B1_ORDER_LOGIT"]
    primary_candidate = scores["B1_ORDER_PLUS_PROTT5_PLS_LOGIT"]
    contrast = _primary_contrast(labels, reference, primary_candidate, groups, float(config["statistics"]["auc_gain_margin"]), int(config["statistics"]["bootstrap_replicates_final"]), int(config["statistics"]["bootstrap_seed"]))
    secondary_names = ["B0_TRANSPARENT_LOGIT", "B0_PLUS_PROTT5_PLS_LOGIT", "PROTT5_PLS_ONLY_LOGIT", deferred_arm]
    secondary = []
    secondary_p = []
    for index, arm in enumerate(secondary_names, start=1):
        permutation_seed = int(config["statistics"]["bootstrap_seed"]) + index
        if arm == deferred_arm:
            p_value = 1.0
            row = {"status": "DEFERRED_COMPUTE_LIMIT_NOT_EVALUATED", "reference": "B1_ORDER_LOGIT", "candidate": arm, "eligible_for_claim": False, "cluster_swap_permutation": {"status": "not_computed", "p_value_two_sided_for_multiplicity": p_value}}
        else:
            boot = _primary_contrast(labels, reference, scores[arm], groups, float(config["statistics"]["auc_gain_margin"]), int(config["statistics"]["bootstrap_replicates_final"]), permutation_seed)
            permutation = cluster_swap_permutation(labels, reference, scores[arm], groups, "roc_auc", 999, permutation_seed)
            p_value = float(permutation["p_value_two_sided"])
            row = {"status": "completed", "reference": "B1_ORDER_LOGIT", "candidate": arm, "gain_definition": "AUC(candidate) - AUC(reference)", "bootstrap": boot, "cluster_swap_permutation": permutation}
        secondary.append(row)
        secondary_p.append(p_value)
    for row, value in zip(secondary, _holm(secondary_p)):
        row["holm_adjusted_p_value"] = float(value)
        row["holm_family_size"] = 4

    stability = []
    for stability_seed in config["soluprot"]["fold_stability_seeds"]:
        stability_seed = int(stability_seed)
        stability_folds = balanced_group_folds(labels, groups, int(config["soluprot"]["primary_outer_folds"]), stability_seed, min_class_groups_per_fold=20)
        root = run_dir / "checkpoints/fold_stability" / f"seed_{stability_seed}"
        ref_stability, ref_info = checkpointed_nested_oof(base_blocks["B1"], None, labels, groups, stability_folds, "none", c_grid, k_grid, inner_folds, stability_seed, root / "reference", "B1_ORDER_LOGIT", f"{config['run_id']}:stability:{stability_seed}:reference")
        cand_stability, cand_info = checkpointed_nested_oof(base_blocks["B1"], embeddings, labels, groups, stability_folds, "pls", c_grid, k_grid, inner_folds, stability_seed, root / "candidate", "B1_ORDER_PLUS_PROTT5_PLS_LOGIT", f"{config['run_id']}:stability:{stability_seed}:candidate")
        ref_auc = float(metric_bundle(labels, ref_stability)["roc_auc"])
        cand_auc = float(metric_bundle(labels, cand_stability)["roc_auc"])
        stability.append({"seed": stability_seed, "reference_auc": ref_auc, "candidate_auc": cand_auc, "auc_gain_candidate_minus_reference": cand_auc - ref_auc, "fold_hash": fold_hash(groups, stability_folds), "reference_selected": [{"outer_fold": row["outer_fold"], "c": row["selected_c"]} for row in ref_info], "candidate_selected": [{"outer_fold": row["outer_fold"], "pls_k": row["selected_pls_k"], "c": row["selected_c"]} for row in cand_info]})

    label_permutation = training_label_permutation_control(
        base_blocks["B1"], embeddings, labels, groups, folds,
        info["B1_ORDER_LOGIT"], info["B1_ORDER_PLUS_PROTT5_PLS_LOGIT"],
        int(config["soluprot"]["label_permutation_replicates"]),
        int(config["soluprot"]["label_permutation_seed"]),
        run_dir / "checkpoints/negative_control",
        f"{config['run_id']}:internal-primary:training-label-permutation",
    )
    status = "completed" if label_permutation["status"] == "PASSED" else "PIPELINE_INVALID"
    result = {
        "status": status,
        "analysis_scope": "retrospective_data_informed",
        "endpoint": "recombinant soluble-expression status in Escherichia coli; not direct aqueous solubility",
        "input_hashes": {"train_partitions": sha256_file(Path(config["paths"]["project_root"]) / "data/solubility/soluprot_train_partitions.csv"), "development_features": sha256_file(Path(config["paths"]["project_root"]) / "data/solubility/solubility_transparent_features_development.csv"), "embedding_index": sha256_file(Path(config["paths"]["project_root"]) / "data/solubility/embeddings/soluprot_embedding_index.csv"), "prott5_array": sha256_file(Path(config["paths"]["project_root"]) / "data/solubility/embeddings/soluprot_prott5.npy")},
        "split": {"group_definition": "existing 30% MMseqs2 cluster_id", "n_rows": int(n), "n_groups": int(len(np.unique(groups))), "largest_group_rows": int(pd.Series(groups).value_counts().max()), "singleton_group_fraction": float(np.mean(pd.Series(groups).value_counts().to_numpy() == 1)), "fold_hash": fold_hash(groups, folds), "outer_folds": 7, "inner_folds": inner_folds, "group_overlap_checks": "passed"},
        "primary_contrast": {"reference": "B1_ORDER_LOGIT", "candidate": "B1_ORDER_PLUS_PROTT5_PLS_LOGIT", "metric": "roc_auc", "gain_definition": "AUC(candidate) - AUC(reference)", **contrast},
        "arms": metrics,
        "selected_hyperparameters": info,
        "fold_stability": stability,
        "secondary": secondary,
        "negative_controls": {"alignment_guard": "passed by synthetic tests", "forbidden_feature_guard": "passed by synthetic tests", "training_label_permutation": label_permutation},
        "deferred_arms": [deferred_arm],
        "limitations": ["The raw high-dimensional fusion secondary arm was not fit and is not evidence of a null effect."],
        "private_prediction_files": sorted(private_paths),
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
    }
    _save_raw(run_dir, "internal_primary.json", result)
    return result


def run_external(config: dict, run_dir: Path, data: dict[str, object]) -> dict[str, object]:
    train = data["train"]
    external = data["external"]
    y_dev = train["label"].to_numpy().astype(int)
    dev_groups = train["cluster_id"].astype(str).to_numpy()
    ext_groups = external["external_cluster_id"].astype(str).to_numpy()
    dev_base = data["b1"][: len(train)]
    ext_base = data["b1"][len(train):]
    dev_emb = data["embeddings"][: len(train)]
    ext_emb = data["embeddings"][len(train):]
    partition = train["partition"].astype(str).to_numpy()
    if set(np.unique(partition)) != {"fit", "model_selection", "calibration"}:
        raise ValueError("Unexpected development partition labels.")
    fit_indices = np.flatnonzero(partition == "fit")
    model_selection_indices = np.flatnonzero(partition == "model_selection")
    calibration_indices = np.flatnonzero(partition == "calibration")
    partition_codes = np.asarray(
        [{"fit": 0, "model_selection": 1, "calibration": 2}[value] for value in partition],
        dtype=int,
    )
    partition_group_hash = fold_hash(dev_groups, partition_codes)
    c_grid = [float(value) for value in config["soluprot"]["logistic_c_grid"]]
    k_grid = [int(value) for value in config["soluprot"]["pls_k_grid"]]
    results = {}
    private_paths = []
    for arm, mode in (("B1_ORDER_LOGIT", "none"), ("B1_ORDER_PLUS_PROTT5_PLS_LOGIT", "pls")):
        contract = json.dumps(
            {
                "schema": "external_development_seal_checkpoint_v2",
                "run_id": config["run_id"],
                "arm": arm,
                "mode": mode,
                "input_hashes": config["input_hashes"],
                "fit_rows": int(len(fit_indices)),
                "model_selection_rows": int(len(model_selection_indices)),
                "calibration_rows": int(len(calibration_indices)),
                "development_partition_group_hash": partition_group_hash,
                "selection_metric": "roc_auc",
                "selection_rule": "fit candidates on fit; select on model_selection; tie smaller k then smaller C",
                "refit_rule": "selected specification on fit plus model_selection",
                "calibration_rule": "Platt on calibration only",
                "c_grid": c_grid,
                "pls_k_grid": k_grid,
                "seed": int(config["global_seed"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        checkpoint = run_dir / "checkpoints/external" / f"{arm}.npz"
        if checkpoint.is_file():
            arm_result = _load_external_checkpoint(checkpoint, contract, len(external))
            print(f"[checkpoint] reused {checkpoint.name}", flush=True)
        else:
            emb = dev_emb if mode == "pls" else None
            ext_embedding = ext_emb if mode == "pls" else None
            head, calibrator, selected, selection_info, seal_diagnostics = fit_external_development_seal(
                dev_base,
                emb,
                y_dev,
                dev_groups,
                fit_indices,
                model_selection_indices,
                calibration_indices,
                mode,
                c_grid,
                k_grid,
            )
            raw_ext = predict_head(head, ext_base, ext_embedding)
            calibrated_ext = calibrator.predict(raw_ext)
            _save_external_checkpoint(
                checkpoint,
                contract,
                raw_ext,
                calibrated_ext,
                selected,
                selection_info,
                int(len(calibration_indices)),
                seal_diagnostics,
            )
            arm_result = {
                "raw": raw_ext,
                "calibrated": calibrated_ext,
                "selected": list(selected),
                "selection_info": selection_info,
                "calibration_n": int(len(calibration_indices)),
                "development_seal_diagnostics": seal_diagnostics,
            }
            print(f"[checkpoint] created {checkpoint.name}", flush=True)
        results[arm] = arm_result
    def probability_hash(values: np.ndarray) -> str:
        values = np.ascontiguousarray(values, dtype=np.float64)
        digest = hashlib.sha256()
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
        return digest.hexdigest()

    seal_payload = {
        "schema": "external_development_only_seal_v1",
        "run_id": config["run_id"],
        "external_labels_accessed": False,
        "development_partition_group_hash": partition_group_hash,
        "partition_rows": {
            "fit": int(len(fit_indices)),
            "model_selection": int(len(model_selection_indices)),
            "calibration": int(len(calibration_indices)),
        },
        "arms": {
            arm: {
                "selected": values["selected"],
                "calibration_n": values["calibration_n"],
                "raw_external_prediction_sha256": probability_hash(values["raw"]),
                "calibrated_external_prediction_sha256": probability_hash(values["calibrated"]),
                "checkpoint_sha256": sha256_file(run_dir / "checkpoints/external" / f"{arm}.npz"),
                "development_seal_diagnostics": values["development_seal_diagnostics"],
            }
            for arm, values in results.items()
        },
    }
    seal_path = run_dir / "manifests/external_development_only_seal.json"
    if seal_path.is_file():
        if json.loads(seal_path.read_text(encoding="utf-8")) != json.loads(json.dumps(seal_payload)):
            raise ValueError("Existing external development-only seal differs.")
    else:
        atomic_write_json(seal_payload, seal_path)

    # External outcomes are first accessed only after the development-only
    # selections, refits, calibrators, predictions, and seal are immutable.
    y_ext = external["label"].to_numpy().astype(int)
    for arm, arm_result in results.items():
        frame = pd.DataFrame(
            {
                "record_id": external["record_id"].astype(str),
                "external_cluster_id": ext_groups,
                "label": y_ext,
                "raw_probability": arm_result["raw"],
                "calibrated_probability": arm_result["calibrated"],
            }
        )
        private_paths.append(
            str(_save_predictions_resumable(run_dir, f"soluprot_external_{arm}.parquet", frame))
        )
    reference = results["B1_ORDER_LOGIT"]["calibrated"]
    candidate = results["B1_ORDER_PLUS_PROTT5_PLS_LOGIT"]["calibrated"]
    contrast = _primary_contrast(y_ext, reference, candidate, ext_groups, float(config["statistics"]["auc_gain_margin"]), int(config["statistics"]["bootstrap_replicates_final"]), int(config["statistics"]["bootstrap_seed"]) + 100)
    result = {
        "status": "completed",
        "analysis_scope": "retrospective_data_informed",
        "endpoint": "recombinant soluble-expression status in Escherichia coli; not direct aqueous solubility",
        "split": {"group_definition": "external_cluster_id joined one-to-one by record_id", "n_rows": int(len(external)), "n_groups": int(len(np.unique(ext_groups))), "largest_group_rows": int(pd.Series(ext_groups).value_counts().max()), "singleton_group_fraction": float(np.mean(pd.Series(ext_groups).value_counts().to_numpy() == 1)), "external_tuning": "none", "external_join": "3100/3100 rows and 2723/2723 groups"},
        "primary_contrast": {"reference": "B1_ORDER_LOGIT", "candidate": "B1_ORDER_PLUS_PROTT5_PLS_LOGIT", "metric": "roc_auc", "gain_definition": "AUC(candidate) - AUC(reference)", **contrast},
        "arms": {arm: {"raw": _arm_metrics(y_ext, values["raw"], ext_groups), "calibrated": _arm_metrics(y_ext, values["calibrated"], ext_groups), "selected_hyperparameter": values["selected"], "calibration_n": values["calibration_n"]} for arm, values in results.items()},
        "calibration_contract": "fit-only candidate training; model-selection-only k/C selection; refit on fit plus model_selection; Platt on calibration only; development-only seal written before external labels were accessed",
        "development_only_seal": str(seal_path),
        "private_prediction_files": private_paths,
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
    }
    _save_raw(run_dir, "external_retrospective.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["historical-bridge", "internal-primary", "external-retrospective"], required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    run_dir = args.run_dir.resolve()
    if str(run_dir) != str(Path(config["paths"]["run_root"]).resolve()):
        raise ValueError("--run-dir does not match config run_root.")
    if args.stage == "historical-bridge":
        required = ["preflight"]
    elif args.stage == "internal-primary":
        required = ["preflight", "historical_bridge"]
    else:
        required = ["preflight", "historical_bridge", "internal_primary"]
    # The preflight manifest has a different filename from stage manifests.
    if "preflight" in required and not (run_dir / "manifests/preflight_manifest.json").is_file():
        raise FileNotFoundError("Phase 0 preflight manifest is missing.")
    if "historical_bridge" in required and not (run_dir / "manifests/historical-bridge.json").is_file():
        raise FileNotFoundError("Historical bridge stage is missing.")
    if "internal_primary" in required and not (run_dir / "manifests/internal-primary.json").is_file():
        raise FileNotFoundError("Internal primary stage is missing.")
    data = assemble(config)
    if args.stage == "historical-bridge":
        payload = run_historical_bridge(config, run_dir, data)
        write_stage(run_dir, "historical-bridge", {"result_path": str(run_dir / "raw_results/historical_bridge.json"), "input_rows": 11436, "status_detail": payload["status"]})
    elif args.stage == "internal-primary":
        payload = run_internal(config, run_dir, data)
        if payload["status"] != "completed":
            raise RuntimeError(
                "Internal primary negative-control gate failed; raw diagnostics were saved but no stage manifest was released."
            )
        write_stage(run_dir, "internal-primary", {"result_path": str(run_dir / "raw_results/internal_primary.json"), "status_detail": payload["status"]}, requires=["historical-bridge"])
    else:
        payload = run_external(config, run_dir, data)
        write_stage(run_dir, "external-retrospective", {"result_path": str(run_dir / "raw_results/external_retrospective.json"), "status_detail": payload["status"]}, requires=["historical-bridge", "internal-primary"])
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
