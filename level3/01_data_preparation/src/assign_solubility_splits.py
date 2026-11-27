#!/usr/bin/env python3
"""Assign deterministic cluster-level SoluProt development partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold


SEED = 20260804
N_FOLDS = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/solubility/soluprot_train.csv"),
    )
    parser.add_argument(
        "--cluster-tsv",
        type=Path,
        default=Path("data/solubility/soluprot_train_mmseqs30_cluster.tsv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/solubility/soluprot_train_partitions.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/solubility/soluprot_partition_manifest.json"),
    )
    parser.add_argument(
        "--external-csv",
        type=Path,
        default=Path("data/solubility/soluprot_external_test.csv"),
    )
    parser.add_argument(
        "--external-cluster-tsv",
        type=Path,
        default=Path("data/solubility/soluprot_external_mmseqs30_cluster.tsv"),
    )
    parser.add_argument(
        "--external-output-csv",
        type=Path,
        default=Path("data/solubility/soluprot_external_test_clustered.csv"),
    )
    return parser.parse_args()


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    role_stats: dict[str, object] = {}
    for role, subset in frame.groupby("partition", sort=True):
        role_stats[str(role)] = {
            "rows": int(len(subset)),
            "clusters": int(subset["cluster_id"].nunique()),
            "label_counts": {
                str(int(key)): int(value)
                for key, value in subset["label"].value_counts().sort_index().items()
            },
            "length": {
                "min": int(subset["length"].min()),
                "median": float(subset["length"].median()),
                "mean": float(subset["length"].mean()),
                "max": int(subset["length"].max()),
            },
        }
    return role_stats


def main() -> None:
    args = parse_args()
    train_path = args.train_csv.resolve()
    cluster_path = args.cluster_tsv.resolve()
    output_path = args.output_csv.resolve()
    manifest_path = args.manifest.resolve()
    external_path = args.external_csv.resolve()
    external_cluster_path = args.external_cluster_tsv.resolve()
    external_output_path = args.external_output_csv.resolve()

    frame = pd.read_csv(train_path, dtype={"record_id": str, "source_id": str})
    clusters = pd.read_csv(
        cluster_path,
        sep="\t",
        header=None,
        names=["cluster_id", "record_id"],
        dtype=str,
    )
    if clusters["record_id"].duplicated().any():
        raise ValueError("A record occurs more than once in the MMseqs2 cluster table.")
    if set(clusters["record_id"]) != set(frame["record_id"]):
        raise ValueError("MMseqs2 cluster membership does not match the analysis-ready training set.")

    frame = frame.merge(clusters, on="record_id", validate="one_to_one")
    cluster_sizes = frame.groupby("cluster_id").size().rename("cluster_size")
    frame = frame.join(cluster_sizes, on="cluster_id")

    # Length strata are derived from training data only and used solely to improve
    # partition balance. The cut points are fixed here before model fitting.
    edges = [-np.inf, 150, 300, 500, np.inf]
    labels = ["le150", "151to300", "301to500", "gt500"]
    frame["length_stratum"] = pd.cut(frame["length"], bins=edges, labels=labels).astype(str)
    frame["stratification_key"] = frame["label"].astype(str) + "_" + frame["length_stratum"]

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    frame["development_fold"] = -1
    for fold, (_, held_out_index) in enumerate(
        splitter.split(frame, y=frame["stratification_key"], groups=frame["cluster_id"])
    ):
        frame.loc[frame.index[held_out_index], "development_fold"] = fold
    if (frame["development_fold"] < 0).any():
        raise RuntimeError("Some training records were not assigned to a development fold.")

    # Fold roles are fixed a priori. The official 3,100-case NESG test remains
    # separate and untouched.
    frame["partition"] = "fit"
    frame.loc[frame["development_fold"] == 0, "partition"] = "model_selection"
    frame.loc[frame["development_fold"] == 1, "partition"] = "calibration"

    cluster_role_counts = frame.groupby("cluster_id")["partition"].nunique()
    if int(cluster_role_counts.max()) != 1:
        raise RuntimeError("A similarity cluster crossed a development partition.")

    ordered_columns = [
        "record_id",
        "source_id",
        "sequence",
        "length",
        "label",
        "source_dataset",
        "source_split",
        "endpoint",
        "cluster_id",
        "cluster_size",
        "length_stratum",
        "stratification_key",
        "development_fold",
        "partition",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame[ordered_columns].to_csv(output_path, index=False)

    external = pd.read_csv(external_path, dtype={"record_id": str, "source_id": str})
    external_clusters = pd.read_csv(
        external_cluster_path,
        sep="\t",
        header=None,
        names=["external_cluster_id", "record_id"],
        dtype=str,
    )
    if external_clusters["record_id"].duplicated().any():
        raise ValueError("An external-test record occurs more than once in its cluster table.")
    if set(external_clusters["record_id"]) != set(external["record_id"]):
        raise ValueError("External-test cluster membership does not match the analysis-ready data.")
    external = external.merge(external_clusters, on="record_id", validate="one_to_one")
    external_sizes = external.groupby("external_cluster_id").size().rename("external_cluster_size")
    external = external.join(external_sizes, on="external_cluster_id")
    external_output_path.parent.mkdir(parents=True, exist_ok=True)
    external.to_csv(external_output_path, index=False)

    cluster_label_counts = frame.groupby("cluster_id")["label"].nunique()
    cluster_sizes_array = cluster_sizes.to_numpy()
    manifest = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "fold_roles": {
            "0": "model_selection",
            "1": "calibration",
            "2-6": "fit",
        },
        "clustering": {
            "tool": "MMseqs2",
            "version": "17-b804f",
            "command": (
                "mmseqs easy-cluster soluprot_train.fasta soluprot_train_mmseqs30 "
                "mmseqs_tmp30 --min-seq-id 0.30 -c 0.80 --cov-mode 0 "
                "--cluster-mode 1 -s 7.5 --threads 8"
            ),
            "identity_threshold": 0.30,
            "coverage_threshold": 0.80,
            "coverage_definition": "both query and target",
            "cluster_mode": "connected component through MMseqs2 cascaded clustering",
            "cluster_table_sha256": sha256(cluster_path),
            "n_clusters": int(frame["cluster_id"].nunique()),
            "singleton_clusters": int((cluster_sizes_array == 1).sum()),
            "largest_cluster": int(cluster_sizes_array.max()),
            "mixed_label_clusters": int((cluster_label_counts > 1).sum()),
        },
        "external_clustering": {
            "cluster_table_sha256": sha256(external_cluster_path),
            "n_clusters": int(external["external_cluster_id"].nunique()),
            "singleton_clusters": int((external_sizes.to_numpy() == 1).sum()),
            "largest_cluster": int(external_sizes.max()),
            "mixed_label_clusters": int(
                (external.groupby("external_cluster_id")["label"].nunique() > 1).sum()
            ),
            "output_sha256": sha256(external_output_path),
        },
        "stratification": {
            "variables": ["binary label", "fixed sequence-length stratum"],
            "length_strata": {
                "le150": "length <= 150",
                "151to300": "150 < length <= 300",
                "301to500": "300 < length <= 500",
                "gt500": "length > 500",
            },
        },
        "partition_summary": summarize(frame),
        "input_sha256": sha256(train_path),
        "output_sha256": sha256(output_path),
        "external_input_sha256": sha256(external_path),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
