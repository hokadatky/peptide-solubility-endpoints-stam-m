"""Strict-I/O V4 external retrospective application.

The frozen model/statistics contract is unchanged.  This runtime path avoids
parsing the external outcome column until development-only model selection,
refit, calibration, external prediction, checkpointing, and hash sealing have
completed.  Historical viewing of this retrospective dataset is recorded
explicitly; this is not a prospective locked benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1] / "vendor"
sys.path.insert(0, str(PAPER))

from analysis_addon.src import run_soluprot_reanalysis as frozen  # noqa: E402
from analysis_addon.src.io_contracts import (  # noqa: E402
    assert_finite_matrix,
    assert_forbidden_columns_absent,
    atomic_write_json,
    sha256_file,
    validate_binary_labels,
    validate_canonical_sequence,
    validate_unique_ids,
)
from analysis_addon.src.resumable_models import _array_hash  # noqa: E402
from analysis_addon.src.sequence_features import build_b0_b1, primary_feature_names  # noqa: E402
from analysis_addon.src.split_contracts import fold_hash  # noqa: E402
from analysis_addon.src.stage_utils import write_stage  # noqa: E402


EXPECTED_RUN_ID = "addon_20260810_v4"


def read_external_without_outcome(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if "label" not in columns:
        raise ValueError("External source is missing the expected outcome column.")
    selected = [column for column in columns if column != "label"]
    frame = pd.read_csv(path, usecols=selected, dtype={"record_id": str, "source_id": str})
    if "label" in frame.columns:
        raise RuntimeError("External outcome entered the pre-seal frame.")
    return frame


def load_label_blind_inputs(config: dict) -> dict[str, object]:
    data_root = Path(config["paths"]["data_root"])
    train = frozen._read_csv(data_root / "soluprot_train_partitions.csv")
    external = read_external_without_outcome(data_root / "soluprot_external_test_clustered.csv")
    train_features = frozen._read_csv(
        data_root / "solubility_transparent_features_development.csv"
    )
    external_features = frozen._read_csv(
        data_root / "solubility_transparent_features_external.csv"
    )
    validate_unique_ids(train, "record_id")
    validate_unique_ids(external, "record_id")
    validate_binary_labels(train, "label")
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
    b0_names = primary_feature_names(
        manifest_path, expected["transparent_manifest_group"]
    )
    dev_b0 = frozen._align_features(train, train_features, b0_names)
    ext_b0 = frozen._align_features(external, external_features, b0_names)
    all_features = pd.concat([dev_b0, ext_b0], ignore_index=True)
    all_sequences = pd.concat([train["sequence"], external["sequence"]], ignore_index=True).tolist()
    blocks = build_b0_b1(
        all_features,
        all_sequences,
        manifest_path,
        expected["ordered_dimer_alphabet"],
    )
    b1 = np.asarray(blocks["B1"][0], dtype=np.float64)
    assert_finite_matrix(b1, "strict external B1")

    index = pd.read_csv(
        data_root / "embeddings/soluprot_embedding_index.csv", dtype={"record_id": str}
    )
    validate_unique_ids(index, "record_id")
    expected_ids = pd.concat(
        [train["record_id"].astype(str), external["record_id"].astype(str)],
        ignore_index=True,
    ).to_numpy()
    if not np.array_equal(index["record_id"].astype(str).to_numpy(), expected_ids):
        raise ValueError("Embedding index order does not equal development-then-external order.")
    embeddings = np.load(data_root / "embeddings/soluprot_prott5.npy", mmap_mode="r")
    if embeddings.shape != (len(expected_ids), int(expected["prott5_dimension"])):
        raise ValueError(f"ProtT5 array shape mismatch: {embeddings.shape}")
    embeddings = np.asarray(embeddings, dtype=np.float64)
    assert_finite_matrix(embeddings, "strict external ProtT5 embeddings")
    for name in blocks["B1"][1]:
        assert_forbidden_columns_absent([name])
    return {
        "train": train,
        "external_without_outcome": external,
        "b1": b1,
        "embeddings": embeddings,
        "preseal_hashes": {
            "development_b1": _array_hash(b1[: len(train)]),
            "external_b1": _array_hash(b1[len(train) :]),
            "development_embedding": _array_hash(embeddings[: len(train)]),
            "external_embedding": _array_hash(embeddings[len(train) :]),
            "external_record_order": string_array_hash(
                external["record_id"].astype(str).to_numpy()
            ),
        },
    }


def probability_hash(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def string_array_hash(values: np.ndarray) -> str:
    """Hash ordered strings without object-pointer-dependent NumPy bytes."""
    values = np.asarray(values).astype(str).reshape(-1)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def build_development_only_predictions(
    config: dict, run_dir: Path, data: dict[str, object]
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    train = data["train"]
    external = data["external_without_outcome"]
    n_dev = len(train)
    y_dev = train["label"].to_numpy().astype(int)
    dev_groups = train["cluster_id"].astype(str).to_numpy()
    dev_base = data["b1"][:n_dev]
    ext_base = data["b1"][n_dev:]
    dev_emb = data["embeddings"][:n_dev]
    ext_emb = data["embeddings"][n_dev:]
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
    results: dict[str, dict[str, object]] = {}
    for arm, mode in (
        ("B1_ORDER_LOGIT", "none"),
        ("B1_ORDER_PLUS_PROTT5_PLS_LOGIT", "pls"),
    ):
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
            arm_result = frozen._load_external_checkpoint(checkpoint, contract, len(external))
            print(f"[strict-external] validated existing {checkpoint.name}", flush=True)
        else:
            embedding = dev_emb if mode == "pls" else None
            ext_embedding = ext_emb if mode == "pls" else None
            head, calibrator, selected, selection_info, diagnostics = (
                frozen.fit_external_development_seal(
                    dev_base,
                    embedding,
                    y_dev,
                    dev_groups,
                    fit_indices,
                    model_selection_indices,
                    calibration_indices,
                    mode,
                    c_grid,
                    k_grid,
                )
            )
            raw_ext = frozen.predict_head(head, ext_base, ext_embedding)
            calibrated_ext = calibrator.predict(raw_ext)
            frozen._save_external_checkpoint(
                checkpoint,
                contract,
                raw_ext,
                calibrated_ext,
                selected,
                selection_info,
                int(len(calibration_indices)),
                diagnostics,
            )
            arm_result = {
                "raw": raw_ext,
                "calibrated": calibrated_ext,
                "selected": list(selected),
                "selection_info": selection_info,
                "calibration_n": int(len(calibration_indices)),
                "development_seal_diagnostics": diagnostics,
            }
            print(f"[strict-external] created {checkpoint.name}", flush=True)
        results[arm] = arm_result

    seal_payload = {
        "schema": "external_development_only_seal_v2_strict_io",
        "run_id": config["run_id"],
        "analysis_scope": "retrospective_data_informed",
        "historical_external_outcomes_previously_viewed": True,
        "external_outcome_values_parsed_or_used_by_this_runtime_before_seal": False,
        "development_partition_group_hash": partition_group_hash,
        "partition_rows": {
            "fit": int(len(fit_indices)),
            "model_selection": int(len(model_selection_indices)),
            "calibration": int(len(calibration_indices)),
        },
        "preseal_input_hashes": data["preseal_hashes"],
        "arms": {
            arm: {
                "selected": values["selected"],
                "calibration_n": values["calibration_n"],
                "raw_external_prediction_sha256": probability_hash(values["raw"]),
                "calibrated_external_prediction_sha256": probability_hash(values["calibrated"]),
                "checkpoint_sha256": sha256_file(
                    run_dir / "checkpoints/external" / f"{arm}.npz"
                ),
                "development_seal_diagnostics": values["development_seal_diagnostics"],
            }
            for arm, values in results.items()
        },
    }
    return results, seal_payload


def write_development_only_seal(run_dir: Path, seal_payload: dict[str, object]) -> Path:
    seal_path = run_dir / "manifests/external_development_only_seal.json"
    if seal_path.is_file():
        observed = json.loads(seal_path.read_text(encoding="utf-8"))
        if observed != seal_payload:
            raise ValueError("Existing strict development-only seal differs.")
    else:
        atomic_write_json(seal_payload, seal_path)
    return seal_path


def load_external_outcomes_after_seal(
    config: dict, external_without_outcome: pd.DataFrame, seal_path: Path
) -> np.ndarray:
    if not seal_path.is_file():
        raise FileNotFoundError("External outcomes cannot be parsed before the seal exists.")
    path = Path(config["paths"]["data_root"]) / "soluprot_external_test_clustered.csv"
    outcomes = pd.read_csv(path, usecols=["record_id", "label"], dtype={"record_id": str})
    validate_unique_ids(outcomes, "record_id")
    validate_binary_labels(outcomes, "label")
    if not np.array_equal(
        outcomes["record_id"].astype(str).to_numpy(),
        external_without_outcome["record_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Post-seal external outcome order mismatch.")
    return outcomes["label"].to_numpy().astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = frozen.load_config(config_path)
    if config.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError(f"Runtime path is restricted to {EXPECTED_RUN_ID}.")
    run_dir = args.run_dir.resolve()
    if run_dir != Path(config["paths"]["run_root"]).resolve():
        raise ValueError("--run-dir does not match config run_root.")
    if not (run_dir / "manifests/internal-primary.json").is_file() and not args.preflight_only:
        raise FileNotFoundError("Internal primary stage is required before external evaluation.")

    data = load_label_blind_inputs(config)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_only",
                    "run_id": config["run_id"],
                    "development_rows": int(len(data["train"])),
                    "external_rows": int(len(data["external_without_outcome"])),
                    "external_groups": int(
                        data["external_without_outcome"]["external_cluster_id"].nunique()
                    ),
                    "external_outcome_column_in_preseal_frame": False,
                    "preseal_hashes": data["preseal_hashes"],
                    "writes_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    results, seal_payload = build_development_only_predictions(config, run_dir, data)
    seal_path = write_development_only_seal(run_dir, seal_payload)

    external = data["external_without_outcome"]
    y_ext = load_external_outcomes_after_seal(config, external, seal_path)
    ext_groups = external["external_cluster_id"].astype(str).to_numpy()
    private_paths = []
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
            str(
                frozen._save_predictions_resumable(
                    run_dir, f"soluprot_external_{arm}.parquet", frame
                )
            )
        )
    reference = results["B1_ORDER_LOGIT"]["calibrated"]
    candidate = results["B1_ORDER_PLUS_PROTT5_PLS_LOGIT"]["calibrated"]
    contrast = frozen._primary_contrast(
        y_ext,
        reference,
        candidate,
        ext_groups,
        float(config["statistics"]["auc_gain_margin"]),
        int(config["statistics"]["bootstrap_replicates_final"]),
        int(config["statistics"]["bootstrap_seed"]) + 100,
    )
    result = {
        "status": "completed",
        "analysis_scope": "retrospective_data_informed",
        "endpoint": "recombinant soluble-expression status in Escherichia coli; not direct aqueous solubility",
        "split": {
            "group_definition": "external_cluster_id joined one-to-one by record_id",
            "n_rows": int(len(external)),
            "n_groups": int(len(np.unique(ext_groups))),
            "largest_group_rows": int(pd.Series(ext_groups).value_counts().max()),
            "singleton_group_fraction": float(
                np.mean(pd.Series(ext_groups).value_counts().to_numpy() == 1)
            ),
            "external_tuning": "none",
            "external_join": "3100/3100 rows and 2723/2723 groups",
        },
        "primary_contrast": {
            "reference": "B1_ORDER_LOGIT",
            "candidate": "B1_ORDER_PLUS_PROTT5_PLS_LOGIT",
            "metric": "roc_auc",
            "gain_definition": "AUC(candidate) - AUC(reference)",
            **contrast,
        },
        "arms": {
            arm: {
                "raw": frozen._arm_metrics(y_ext, values["raw"], ext_groups),
                "calibrated": frozen._arm_metrics(y_ext, values["calibrated"], ext_groups),
                "selected_hyperparameter": values["selected"],
                "calibration_n": values["calibration_n"],
            }
            for arm, values in results.items()
        },
        "calibration_contract": "development fit-only candidate training; model-selection-only k/C selection; refit on fit plus model_selection; Platt on calibration only; external outcome values not parsed or used by this runtime until development-only external predictions and their hashes were sealed",
        "retrospective_boundary": "External outcomes and earlier model results had historically been viewed; this procedural seal is not a prospective locked benchmark.",
        "development_only_seal": str(seal_path),
        "private_prediction_files": private_paths,
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
    }
    frozen._save_raw(run_dir, "external_retrospective.json", result)
    write_stage(
        run_dir,
        "external-retrospective",
        {
            "result_path": str(run_dir / "raw_results/external_retrospective.json"),
            "status_detail": result["status"],
            "strict_io_seal_sha256": sha256_file(seal_path),
        },
        requires=["historical-bridge", "internal-primary"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
