"""Validate and import immutable v2 partial artifacts into the v3 run.

This stage never edits the source run.  Every row-level artifact remains under
the private run root; only aggregate hashes and validation summaries enter the
stage manifests.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from .clustered_stats import metric_bundle
    from .io_contracts import atomic_write_json, sha256_file, validate_binary_labels, validate_unique_ids
    from .stage_utils import refuse_existing, write_stage
except ImportError:  # pragma: no cover - direct execution path
    from clustered_stats import metric_bundle
    from io_contracts import atomic_write_json, sha256_file, validate_binary_labels, validate_unique_ids
    from stage_utils import refuse_existing, write_stage


HISTORICAL_FILES = (
    "soluprot_dev_oof_H0_HIST_BASE.parquet",
    "soluprot_dev_oof_H1_HIST_BASE_PLUS_PROTT5_PLS3.parquet",
)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _copy_verified(source: Path, destination: Path) -> str:
    """Copy once, atomically, while allowing a byte-identical interrupted resume."""
    source_hash = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != source_hash:
            raise FileExistsError(f"Existing recovery target differs from source: {destination}")
        return source_hash
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        if sha256_file(temporary) != source_hash:
            raise FileExistsError(f"Stale recovery temporary differs from source: {temporary}")
    else:
        shutil.copyfile(source, temporary)
    if sha256_file(temporary) != source_hash:
        raise IOError(f"Recovery copy hash mismatch: {source} -> {temporary}")
    os.replace(temporary, destination)
    return source_hash


def _validate_oof(
    path: Path,
    train: pd.DataFrame,
    *,
    expected_folds: np.ndarray | None,
    expected_n_folds: int,
    require_record_id: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_parquet(path)
    required = {"cluster_id", "fold", "label", "score"}
    if require_record_id:
        required.add("record_id")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Recovery OOF missing columns {missing}: {path}")
    if len(frame) != len(train):
        raise ValueError(f"Recovery OOF row count mismatch: {path}")
    if require_record_id:
        validate_unique_ids(frame, "record_id")
        if not np.array_equal(frame["record_id"].astype(str).to_numpy(), train["record_id"].astype(str).to_numpy()):
            raise ValueError(f"Recovery OOF record order mismatch: {path}")
    labels = frame["label"].to_numpy(dtype=int)
    groups = frame["cluster_id"].astype(str).to_numpy()
    folds = frame["fold"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    if not np.array_equal(labels, train["label"].to_numpy(dtype=int)):
        raise ValueError(f"Recovery OOF label order mismatch: {path}")
    if not np.array_equal(groups, train["cluster_id"].astype(str).to_numpy()):
        raise ValueError(f"Recovery OOF group order mismatch: {path}")
    if expected_folds is not None and not np.array_equal(folds, expected_folds):
        raise ValueError(f"Recovery OOF fold order mismatch: {path}")
    if set(np.unique(folds).tolist()) != set(range(expected_n_folds)):
        raise ValueError(f"Recovery OOF fold labels are not 0..{expected_n_folds - 1}: {path}")
    if pd.DataFrame({"group": groups, "fold": folds}).groupby("group", sort=False)["fold"].nunique().max() != 1:
        raise ValueError(f"Recovery OOF has a group split across folds: {path}")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"Recovery OOF score contract failed: {path}")
    return frame, {
        "sha256": sha256_file(path),
        "n_rows": int(len(frame)),
        "n_groups": int(len(np.unique(groups))),
        "n_folds": int(len(np.unique(folds))),
        "metrics": metric_bundle(labels, scores),
        "record_order_checked": bool(require_record_id),
        "label_group_fold_alignment_checked": True,
        "finite_probability_checked": True,
    }


def recover(config: dict, run_dir: Path) -> dict[str, object]:
    recovery = config.get("recovery")
    if not isinstance(recovery, dict):
        raise ValueError("The selected config has no recovery contract.")
    source_run = Path(recovery["source_run_root"]).resolve()
    if source_run == run_dir:
        raise ValueError("Recovery source and destination must differ.")
    active_preflight_path = run_dir / "manifests/preflight_manifest.json"
    active_preflight = _load_json(active_preflight_path)
    source_preflight_path = source_run / "manifests/preflight_manifest.json"
    source_preflight = _load_json(source_preflight_path)
    if active_preflight.get("run_id") != config["run_id"]:
        raise ValueError("Active preflight run_id mismatch.")
    if source_preflight.get("run_id") != recovery["source_run_id"]:
        raise ValueError("Source preflight run_id mismatch.")
    if source_preflight.get("input_hashes") != config["input_hashes"]:
        raise ValueError("Source and active input canaries differ.")
    if source_preflight.get("author_decisions", {}).get("author_decisions_sha256") != active_preflight.get("author_decisions", {}).get("author_decisions_sha256"):
        raise ValueError("Source and active author-decision hashes differ.")

    data_root = Path(config["paths"]["data_root"])
    train = pd.read_csv(data_root / "soluprot_train_partitions.csv", dtype={"record_id": str})
    validate_unique_ids(train, "record_id")
    validate_binary_labels(train, "label")
    expected_folds = train[config["soluprot"]["primary_outer_fold_column"]].to_numpy(dtype=int)

    validations: dict[str, dict[str, object]] = {}
    imported: dict[str, dict[str, str]] = {}
    historical_frames: list[pd.DataFrame] = []
    for filename in HISTORICAL_FILES:
        source = source_run / "predictions" / filename
        frame, validation = _validate_oof(
            source,
            train,
            expected_folds=None,
            expected_n_folds=int(config["historical_bridge"]["outer_folds"]),
            require_record_id=False,
        )
        historical_frames.append(frame)
        destination = run_dir / "predictions" / filename
        digest = _copy_verified(source, destination)
        validations[filename] = validation
        imported[filename] = {"source": str(source), "destination": str(destination), "sha256": digest}
    if not np.array_equal(historical_frames[0][["label", "cluster_id", "fold"]].to_numpy(), historical_frames[1][["label", "cluster_id", "fold"]].to_numpy()):
        raise ValueError("Historical reference and candidate OOF rows/folds differ.")

    import_arms = list(recovery["import_secondary_oof"])
    reference_arm = str(recovery["verify_recomputed_reference_against_v2"])
    for arm in import_arms + [reference_arm]:
        filename = f"soluprot_dev_oof_{arm}.parquet"
        source = source_run / "predictions" / filename
        _, validation = _validate_oof(
            source,
            train,
            expected_folds=expected_folds,
            expected_n_folds=int(config["soluprot"]["primary_outer_folds"]),
            require_record_id=True,
        )
        destination = (
            run_dir / "recovery_imports" / f"soluprot_dev_oof_{arm}_v2.parquet"
            if arm == reference_arm
            else run_dir / "predictions" / filename
        )
        digest = _copy_verified(source, destination)
        validations[filename] = validation
        imported[filename] = {"source": str(source), "destination": str(destination), "sha256": digest}

    source_historical_path = source_run / "raw_results/historical_bridge.json"
    historical = _load_json(source_historical_path)
    ref_auc = validations[HISTORICAL_FILES[0]]["metrics"]["roc_auc"]
    cand_auc = validations[HISTORICAL_FILES[1]]["metrics"]["roc_auc"]
    reported = historical["historical_bridge"]
    if not np.isclose(ref_auc, reported["reference_metrics"]["metrics"]["roc_auc"], atol=1e-12, rtol=0.0):
        raise ValueError("Recovered historical reference AUC differs from source aggregate.")
    if not np.isclose(cand_auc, reported["candidate_metrics"]["metrics"]["roc_auc"], atol=1e-12, rtol=0.0):
        raise ValueError("Recovered historical candidate AUC differs from source aggregate.")
    historical["status"] = "completed_recovered_byte_identical_predictions"
    historical["private_prediction_files"] = [str(run_dir / "predictions" / name) for name in HISTORICAL_FILES]
    historical["recovery_provenance"] = {
        "source_run_id": recovery["source_run_id"],
        "source_result_sha256": sha256_file(source_historical_path),
        "prediction_import_mode": "byte_identical_after_full_alignment_validation",
    }
    historical_path = run_dir / "raw_results/historical_bridge.json"
    refuse_existing(historical_path)
    atomic_write_json(historical, historical_path)

    payload = {
        "status": "completed",
        "source_run_id": recovery["source_run_id"],
        "active_run_id": config["run_id"],
        "source_preflight_sha256": sha256_file(source_preflight_path),
        "active_preflight_sha256": sha256_file(active_preflight_path),
        "source_historical_result_sha256": sha256_file(source_historical_path),
        "validated_imports": imported,
        "validation_summaries": validations,
        "deferred_secondary_arms": list(recovery["deferred_secondary_arms"]),
        "deferred_arm_multiplicity_p_value": float(recovery["deferred_arm_multiplicity_p_value"]),
        "reason": recovery["reason"],
        "source_run_unchanged": True,
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
    }
    recovery_path = run_dir / "raw_results/recovery_v2.json"
    refuse_existing(recovery_path)
    atomic_write_json(payload, recovery_path)
    write_stage(
        run_dir,
        "recovery-v2",
        {"result_path": str(recovery_path), "status_detail": payload["status"]},
    )
    write_stage(
        run_dir,
        "historical-bridge",
        {"result_path": str(historical_path), "input_rows": int(len(train)), "status_detail": historical["status"]},
        requires=["recovery-v2"],
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    if run_dir != Path(config["paths"]["run_root"]).resolve():
        raise ValueError("--run-dir does not match config run_root.")
    payload = recover(config, run_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
