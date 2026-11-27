"""Quantify split severity, sequence similarity, and external source shift."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from .clustered_stats import metric_bundle, paired_cluster_bootstrap
    from .io_contracts import atomic_write_json, sha256_file, validate_unique_ids
    from .resumable_models import checkpointed_nested_oof
    from .sequence_features import build_b0_b1, primary_feature_names
    from .split_contracts import balanced_group_folds, fold_hash
    from .stage_utils import refuse_existing, write_stage
except ImportError:  # pragma: no cover - direct execution path
    from clustered_stats import metric_bundle, paired_cluster_bootstrap
    from io_contracts import atomic_write_json, sha256_file, validate_unique_ids
    from resumable_models import checkpointed_nested_oof
    from sequence_features import build_b0_b1, primary_feature_names
    from split_contracts import balanced_group_folds, fold_hash
    from stage_utils import refuse_existing, write_stage


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"record_id": str, "source_id": str})


def _write_fasta(path: Path, sequences: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing an already-used MMseqs attempt FASTA: {path}")
    with path.open("w", encoding="utf-8") as handle:
        for index, sequence in enumerate(sequences):
            handle.write(f">r{index:05d}\n{sequence}\n")


def _run_mmseqs(run_dir: Path, name: str, sequences: list[str], threshold: float, config: dict) -> np.ndarray:
    sequence_digest = hashlib.sha256()
    for sequence in sequences:
        sequence_digest.update(sequence.encode("ascii") + b"\0")
    contract = json.dumps(
        {
            "schema": "mmseqs_group_checkpoint_v1",
            "run_id": config["run_id"],
            "name": name,
            "threshold": float(threshold),
            "n_sequences": int(len(sequences)),
            "sequence_sha256": sequence_digest.hexdigest(),
            "coverage": float(config["shift"]["pooled_recluster_coverage"]),
            "cov_mode": 0,
            "cluster_mode": int(config["shift"]["pooled_recluster_cluster_mode"]),
            "sensitivity": float(config["shift"]["pooled_recluster_sensitivity"]),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    checkpoint = run_dir / "checkpoints/mmseqs" / f"{name}_{threshold:.2f}.npz"
    if checkpoint.is_file():
        with np.load(checkpoint, allow_pickle=False) as payload:
            observed_contract = str(payload["contract_json"][0])
            groups = np.asarray(payload["groups"]).astype(str)
        if observed_contract != contract or groups.shape != (len(sequences),) or np.any(groups == ""):
            raise ValueError(f"MMseqs checkpoint contract failed: {checkpoint}")
        print(f"[checkpoint] reused {checkpoint.name}", flush=True)
        return groups

    attempt = run_dir / "splits/mmseqs_attempts" / f"{name}_{threshold:.2f}_{os.getpid()}"
    fasta = attempt / "input.fasta"
    _write_fasta(fasta, sequences)
    prefix = attempt / "cluster"
    tmp = attempt / "tmp"
    cluster_tsv = Path(str(prefix) + "_cluster.tsv")
    command = [
        "mmseqs", "easy-cluster", str(fasta), str(prefix), str(tmp),
        "--min-seq-id", str(threshold),
        "-c", str(config["shift"]["pooled_recluster_coverage"]),
        "--cov-mode", "0",
        "--cluster-mode", str(config["shift"]["pooled_recluster_cluster_mode"]),
        "-s", str(config["shift"]["pooled_recluster_sensitivity"]),
        "--threads", "8",
    ]
    log = run_dir / "logs" / f"mmseqs_{name}_{threshold:.2f}_{os.getpid()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    if not cluster_tsv.is_file():
        raise FileNotFoundError(cluster_tsv)
    mapping: dict[str, str] = {}
    with cluster_tsv.open(encoding="utf-8") as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")[:2]
            mapping[member] = representative
    expected = {f"r{index:05d}" for index in range(len(sequences))}
    if set(mapping) != expected:
        raise ValueError(f"MMseqs cluster membership is incomplete for {name} {threshold:.2f}.")
    groups = np.asarray([mapping[f"r{index:05d}"] for index in range(len(sequences))], dtype=str)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, contract_json=np.asarray([contract]), groups=groups)
    os.replace(temporary, checkpoint)
    print(f"[checkpoint] created {checkpoint.name}", flush=True)
    return groups


def _cluster_diagnostics(y: np.ndarray, groups: np.ndarray, folds: np.ndarray | None = None) -> dict[str, object]:
    sizes = pd.Series(groups).value_counts().to_numpy()
    mixed = 0
    for group in np.unique(groups):
        if len(np.unique(y[groups == group])) > 1:
            mixed += 1
    result = {
        "n_rows": int(len(y)),
        "n_groups": int(len(sizes)),
        "singleton_groups": int(np.sum(sizes == 1)),
        "singleton_group_fraction": float(np.mean(sizes == 1)),
        "singleton_row_fraction": float(np.sum(sizes[sizes == 1]) / len(y)),
        "min_group_rows": int(sizes.min()),
        "median_group_rows": float(np.median(sizes)),
        "p90_group_rows": float(np.percentile(sizes, 90)),
        "max_group_rows": int(sizes.max()),
        "largest_group_row_fraction": float(sizes.max() / len(y)),
        "mixed_label_groups": int(mixed),
    }
    if folds is not None:
        result["fold_class_counts"] = {
            str(int(fold)): {str(int(label)): int(np.sum((folds == fold) & (y == label))) for label in (0, 1)}
            for fold in sorted(np.unique(folds).tolist())
        }
        result["fold_hash"] = fold_hash(groups, folds)
    return result


def _load_core(config: dict) -> dict[str, object]:
    data_root = Path(config["paths"]["data_root"])
    train = _read(data_root / "soluprot_train_partitions.csv")
    external = _read(data_root / "soluprot_external_test_clustered.csv")
    dev_features = _read(data_root / "solubility_transparent_features_development.csv")
    ext_features = _read(data_root / "solubility_transparent_features_external.csv")
    names = primary_feature_names(data_root / "solubility_transparent_features.json")
    dev = train[["record_id", "sequence", "label", "cluster_id", "development_fold"]].copy()
    ext = external[["record_id", "sequence", "label", "external_cluster_id"]].copy()
    dev_features = dev_features.set_index("record_id").loc[dev["record_id"], names].reset_index(drop=True)
    ext_features = ext_features.set_index("record_id").loc[ext["record_id"], names].reset_index(drop=True)
    all_features = pd.concat([dev_features, ext_features], ignore_index=True)
    all_source = pd.concat([dev, ext], ignore_index=True)
    blocks = build_b0_b1(all_features, all_source["sequence"].tolist(), data_root / "solubility_transparent_features.json")
    embeddings = np.asarray(np.load(data_root / "embeddings/soluprot_prott5.npy", mmap_mode="r"), dtype=np.float64)
    if embeddings.shape[0] != len(all_source):
        raise ValueError("Shift audit embedding alignment mismatch.")
    return {"train": train, "external": external, "all_source": all_source, "b0": blocks["B0"][0], "b1": blocks["B1"][0], "embeddings": embeddings}


def _distance_bins(config: dict, run_dir: Path, core: dict[str, object]) -> dict[str, object]:
    similarity_path = Path(config["shift"]["similarity_audit"])
    similarity = _read(similarity_path)
    external = core["external"]
    validate_unique_ids(similarity, "record_id")
    if not np.array_equal(similarity["record_id"].to_numpy(), external["record_id"].to_numpy()):
        raise ValueError("Similarity audit is not aligned to external rows.")
    ref = pd.read_parquet(run_dir / "predictions/soluprot_external_B1_ORDER_LOGIT.parquet")
    cand = pd.read_parquet(run_dir / "predictions/soluprot_external_B1_ORDER_PLUS_PROTT5_PLS_LOGIT.parquet")
    if not np.array_equal(ref["record_id"].to_numpy(), external["record_id"].to_numpy()) or not np.array_equal(cand["record_id"].to_numpy(), external["record_id"].to_numpy()):
        raise ValueError("External prediction rows are not aligned.")
    identity = pd.to_numeric(similarity["global_identity_matches_over_max_length"], errors="coerce")
    if identity.isna().any():
        raise ValueError("Similarity audit has missing global identity values.")
    rank = identity.rank(method="first")
    bins = pd.qcut(rank, 4, labels=False) + 1
    rows = []
    y = external["label"].to_numpy().astype(int)
    groups = external["external_cluster_id"].astype(str).to_numpy()
    ref_score = ref["calibrated_probability"].to_numpy(float)
    cand_score = cand["calibrated_probability"].to_numpy(float)
    for bin_id in range(1, 5):
        mask = bins.to_numpy() == bin_id
        subset_groups = groups[mask]
        n_groups = len(np.unique(subset_groups))
        row = {"identity_quartile": int(bin_id), "n_rows": int(mask.sum()), "n_groups": int(n_groups), "median_global_identity": float(identity[mask].median()), "prevalence": float(y[mask].mean())}
        row["inference_allowed"] = bool(mask.sum() >= 200 and n_groups >= 100)
        if row["inference_allowed"]:
            boot = paired_cluster_bootstrap(y[mask], ref_score[mask], cand_score[mask], subset_groups, "roc_auc", 10000, int(config["statistics"]["bootstrap_seed"]) + bin_id)
            row["reference_auc"] = float(metric_bundle(y[mask], ref_score[mask])["roc_auc"])
            row["candidate_auc"] = float(metric_bundle(y[mask], cand_score[mask])["roc_auc"])
            row["auc_gain_candidate_minus_reference"] = boot["point_gain"]
            row["auc_gain_ci95"] = boot["ci95"]
            row["reference_brier"] = float(metric_bundle(y[mask], ref_score[mask])["brier"])
            row["candidate_brier"] = float(metric_bundle(y[mask], cand_score[mask])["brier"])
        rows.append(row)
    return {"similarity_audit_sha256": sha256_file(similarity_path), "identity_definition": "global_identity_matches_over_max_length", "cutpoints_label_blind": True, "bins": rows, "causal_interpretation_prohibited": True}


def _source_classifier(config: dict, run_dir: Path, core: dict[str, object], pooled_groups: np.ndarray) -> dict[str, object]:
    source_label = np.asarray([0] * len(core["train"]) + [1] * len(core["external"]), dtype=int)
    folds = balanced_group_folds(source_label, pooled_groups, int(config["shift"]["source_classifier_folds"]), int(config["global_seed"]))
    c_grid = [float(value) for value in config["soluprot"]["logistic_c_grid"]]
    k_grid = [int(value) for value in config["soluprot"]["pls_k_grid"]]
    outputs = {}
    for name, base, embedding, mode in (("B0_transparent", core["b0"], None, "none"), ("B1_order_aware", core["b1"], None, "none"), ("ProtT5_PLS", np.zeros((len(source_label), 0)), core["embeddings"], "pls")):
        score, _ = checkpointed_nested_oof(
            base,
            embedding,
            source_label,
            pooled_groups,
            folds,
            mode,
            c_grid,
            k_grid,
            5,
            int(config["global_seed"]),
            run_dir / "checkpoints/shift/source_classifier" / name,
            name,
            f"{config['run_id']}:shift:source-classifier:{name}",
        )
        outputs[name] = {"metrics": metric_bundle(source_label, score), "n_rows": int(len(score)), "n_groups": int(len(np.unique(pooled_groups)))}
    outputs["pooled_group_diagnostics"] = _cluster_diagnostics(source_label, pooled_groups, folds)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    if not (run_dir / "manifests/external-retrospective.json").is_file():
        raise FileNotFoundError("External retrospective stage is required before shift audit.")
    core = _load_core(config)
    train = core["train"]
    y_train = train["label"].to_numpy().astype(int)
    groups_train = train["cluster_id"].astype(str).to_numpy()
    ladder = {}
    for threshold in config["soluprot"]["threshold_ladder"]:
        groups = _run_mmseqs(run_dir, "development", train["sequence"].tolist(), float(threshold), config)
        folds = balanced_group_folds(y_train, groups, int(config["soluprot"]["primary_outer_folds"]), int(config["global_seed"]))
        threshold_name = f"threshold_{float(threshold):.2f}"
        c_grid = [float(value) for value in config["soluprot"]["logistic_c_grid"]]
        k_grid = [int(value) for value in config["soluprot"]["pls_k_grid"]]
        reference, _ = checkpointed_nested_oof(
            core["b1"][: len(train)],
            None,
            y_train,
            groups,
            folds,
            "none",
            c_grid,
            k_grid,
            int(config["soluprot"]["inner_folds"]),
            int(config["global_seed"]),
            run_dir / "checkpoints/shift/threshold_ladder" / threshold_name / "reference",
            "B1_ORDER_LOGIT",
            f"{config['run_id']}:shift:{threshold_name}:reference",
        )
        candidate, _ = checkpointed_nested_oof(
            core["b1"][: len(train)],
            core["embeddings"][: len(train)],
            y_train,
            groups,
            folds,
            "pls",
            c_grid,
            k_grid,
            int(config["soluprot"]["inner_folds"]),
            int(config["global_seed"]),
            run_dir / "checkpoints/shift/threshold_ladder" / threshold_name / "candidate",
            "B1_ORDER_PLUS_PROTT5_PLS_LOGIT",
            f"{config['run_id']}:shift:{threshold_name}:candidate",
        )
        boot = paired_cluster_bootstrap(y_train, reference, candidate, groups, "roc_auc", 10000, int(config["statistics"]["bootstrap_seed"]) + int(round(float(threshold) * 100)))
        ladder[f"{float(threshold):.2f}"] = {"cluster_diagnostics": _cluster_diagnostics(y_train, groups, folds), "reference_auc": float(metric_bundle(y_train, reference)["roc_auc"]), "candidate_auc": float(metric_bundle(y_train, candidate)["roc_auc"]), "auc_gain_candidate_minus_reference": boot["point_gain"], "auc_gain_ci90": boot["ci90"], "auc_gain_ci95": boot["ci95"], "bootstrap_invalid_fraction": boot["invalid_fraction"], "group_definition": "label-blind MMseqs2 sequence clustering", "threshold_not_selected_by_performance": True}
    pooled_sequences = core["all_source"]["sequence"].tolist()
    pooled_groups = _run_mmseqs(run_dir, "pooled", pooled_sequences, 0.30, config)
    result = {
        "status": "completed",
        "analysis_scope": "retrospective_data_informed",
        "endpoint": "recombinant soluble-expression status in Escherichia coli; external shift diagnostic is label-blind",
        "split_severity_ladder": ladder,
        "source_classifier": _source_classifier(config, run_dir, core, pooled_groups),
        "external_shift": _distance_bins(config, run_dir, core),
        "interpretation_boundary": "Performance attenuation cannot be causally assigned separately to sequence similarity, source database, operator, or label construction in this retrospective design.",
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
    }
    path = run_dir / "raw_results/shift_audit.json"
    refuse_existing(path)
    atomic_write_json(result, path)
    write_stage(run_dir, "shift-audit", {"result_path": str(path), "status_detail": result["status"]}, requires=["external-retrospective"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
