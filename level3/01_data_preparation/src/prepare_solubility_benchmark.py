#!/usr/bin/env python3
"""Prepare and audit the official SoluProt benchmark files.

This script is intentionally conservative. It preserves the downloaded source
archives, joins labels to sequences by the official identifiers, rejects
non-canonical sequences, and writes derived analysis tables separately. Exact
duplicates, label conflicts, and train--test overlap are reported before any
model fitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from Bio import SeqIO


CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
ENDPOINT = "recombinant-expression soluble status in Escherichia coli"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize(sequence: str) -> str:
    """Upper-case and remove whitespace; do not alter tags or residues."""
    return "".join(sequence.split()).upper()


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for record in SeqIO.parse(path, "fasta"):
        identifier = str(record.id)
        if identifier in records:
            raise ValueError(f"Duplicate FASTA identifier in {path.name}: {identifier}")
        records[identifier] = canonicalize(str(record.seq))
    return records


def length_summary(lengths: Iterable[int]) -> dict[str, float | int]:
    values = np.asarray(list(lengths), dtype=float)
    if values.size == 0:
        return {"n": 0}
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(values.size),
        "min": int(values.min()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "mean": float(values.mean()),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "max": int(values.max()),
        "n_gt_900": int((values > 900).sum()),
        "n_gt_1000": int((values > 1000).sum()),
        "n_gt_1022": int((values > 1022).sum()),
    }


def sequence_audit(frame: pd.DataFrame) -> dict[str, object]:
    grouped = frame.groupby("sequence", sort=False)["label"]
    duplicate_groups = grouped.size()
    label_cardinality = grouped.nunique()
    noncanonical = frame["sequence"].map(lambda value: sorted(set(value) - CANONICAL_AA))
    return {
        "rows": int(len(frame)),
        "unique_sequences": int(frame["sequence"].nunique()),
        "duplicate_sequence_groups": int((duplicate_groups > 1).sum()),
        "duplicate_rows_beyond_first": int((duplicate_groups - 1).clip(lower=0).sum()),
        "conflicting_label_groups": int((label_cardinality > 1).sum()),
        "label_counts": {str(int(k)): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
        "noncanonical_sequences": int(noncanonical.map(bool).sum()),
        "missing_sequences": int(frame["sequence"].isna().sum()),
        "missing_labels": int(frame["label"].isna().sum()),
        "length": length_summary(frame["length"]),
    }


def join_official_split(csv_path: Path, fasta_path: Path, split: str) -> pd.DataFrame:
    labels = pd.read_csv(csv_path)
    expected_columns = {"sid", "solubility"}
    if set(labels.columns) != expected_columns:
        raise ValueError(f"Unexpected columns in {csv_path.name}: {labels.columns.tolist()}")
    if labels["sid"].duplicated().any():
        raise ValueError(f"Duplicate official identifiers in {csv_path.name}")
    if not set(labels["solubility"].unique()).issubset({0, 1}):
        raise ValueError(f"Non-binary labels in {csv_path.name}")

    sequences = read_fasta(fasta_path)
    label_ids = set(labels["sid"].astype(str))
    fasta_ids = set(sequences)
    if label_ids != fasta_ids:
        raise ValueError(
            f"CSV/FASTA identifier mismatch for {split}: "
            f"CSV-only={len(label_ids - fasta_ids)}, FASTA-only={len(fasta_ids - label_ids)}"
        )

    frame = labels.rename(columns={"sid": "source_id", "solubility": "label"}).copy()
    frame["source_id"] = frame["source_id"].astype(str)
    frame["sequence"] = frame["source_id"].map(sequences)
    frame["length"] = frame["sequence"].str.len().astype(int)
    frame["source_dataset"] = "SoluProt"
    frame["source_split"] = split
    frame["endpoint"] = ENDPOINT
    frame["record_id"] = [f"soluprot_{split}_{i:05d}" for i in range(len(frame))]
    return frame[
        [
            "record_id",
            "source_id",
            "sequence",
            "length",
            "label",
            "source_dataset",
            "source_split",
            "endpoint",
        ]
    ]


def audit_original_nesg(csv_path: Path, fasta_path: Path, official_test: pd.DataFrame) -> dict[str, object]:
    metadata = pd.read_csv(csv_path)
    sequences = read_fasta(fasta_path)
    metadata_ids = set(metadata["id"].astype(str))
    fasta_ids = set(sequences)
    if metadata_ids != fasta_ids:
        raise ValueError(
            "Original NESG CSV/FASTA identifier mismatch: "
            f"CSV-only={len(metadata_ids - fasta_ids)}, FASTA-only={len(fasta_ids - metadata_ids)}"
        )
    frame = metadata.copy()
    frame["sequence"] = frame["id"].astype(str).map(sequences)
    sizes = frame.groupby("sequence", sort=False).size()
    conflicts = frame.groupby("sequence", sort=False)["sol"].nunique()
    official_sequences = set(official_test["sequence"])
    official_matches = frame[frame["sequence"].isin(official_sequences)]
    match_counts = official_matches.groupby("sequence", sort=False).size()
    return {
        "rows": int(len(frame)),
        "unique_sequences": int(frame["sequence"].nunique()),
        "duplicate_sequence_groups": int((sizes > 1).sum()),
        "duplicate_rows_beyond_first": int((sizes - 1).clip(lower=0).sum()),
        "conflicting_sol_score_groups": int((conflicts > 1).sum()),
        "official_test_sequences_found": int(official_test["sequence"].isin(set(frame["sequence"])).sum()),
        "official_test_sequences_missing": int((~official_test["sequence"].isin(set(frame["sequence"]))).sum()),
        "official_test_matching_nesg_rows": int(len(official_matches)),
        "official_test_sequences_with_multiple_nesg_rows": int((match_counts > 1).sum()),
        "note": (
            "The official 3,100-case SoluProt test set is retained as distributed. "
            "The larger NESG archive is audited for provenance only and is not used to reconstruct labels."
        ),
    }


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in frame.itertuples(index=False):
            handle.write(f">{row.record_id}\n{row.sequence}\n")


def write_summary_csv(audits: dict[str, dict[str, object]], path: Path) -> None:
    rows: list[dict[str, object]] = []
    for split, audit in audits.items():
        length = audit["length"]
        rows.append(
            {
                "split": split,
                "rows": audit["rows"],
                "unique_sequences": audit["unique_sequences"],
                "label_0": audit["label_counts"].get("0", 0),
                "label_1": audit["label_counts"].get("1", 0),
                "duplicate_sequence_groups": audit["duplicate_sequence_groups"],
                "conflicting_label_groups": audit["conflicting_label_groups"],
                "noncanonical_sequences": audit["noncanonical_sequences"],
                "min_length": length["min"],
                "median_length": length["median"],
                "mean_length": length["mean"],
                "max_length": length["max"],
                "n_gt_900": length["n_gt_900"],
                "n_gt_1000": length["n_gt_1000"],
                "n_gt_1022": length["n_gt_1022"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/external/soluprot"),
        help="Directory containing the official SoluProt archives and extracted files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/solubility"),
        help="Directory for derived, analysis-ready files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = source_dir / "soluprot_data"
    paths = {
        "soluprot_archive": source_dir / "soluprot_data.zip",
        "nesg_archive": source_dir / "nesg.zip",
        "training_csv": extracted / "training_set.csv",
        "training_fasta": extracted / "training_set.fasta",
        "test_csv": extracted / "test_set.csv",
        "test_fasta": extracted / "test_set.fasta",
        "nesg_csv": source_dir / "nesg.csv",
        "nesg_fasta": source_dir / "nesg.fasta",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required source files: {missing}")

    train = join_official_split(paths["training_csv"], paths["training_fasta"], "train")
    external = join_official_split(paths["test_csv"], paths["test_fasta"], "external_test")

    train_audit = sequence_audit(train)
    external_audit = sequence_audit(external)
    if train_audit["noncanonical_sequences"] or external_audit["noncanonical_sequences"]:
        raise ValueError("Non-canonical sequences detected; the primary analysis is canonical-residue only.")
    if train_audit["conflicting_label_groups"] or external_audit["conflicting_label_groups"]:
        raise ValueError("Conflicting labels detected in an official split; resolve before modelling.")

    exact_overlap = set(train["sequence"]) & set(external["sequence"])
    if exact_overlap:
        raise ValueError(
            f"Detected {len(exact_overlap)} exact train--external overlaps. "
            "The locked policy requires removing overlaps from training before modelling."
        )

    train.to_csv(output_dir / "soluprot_train.csv", index=False)
    external.to_csv(output_dir / "soluprot_external_test.csv", index=False)
    write_fasta(train, output_dir / "soluprot_train.fasta")
    write_fasta(external, output_dir / "soluprot_external_test.fasta")

    audits = {"train": train_audit, "external_test": external_audit}
    write_summary_csv(audits, output_dir / "soluprot_split_audit.csv")
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint_name": ENDPOINT,
        "endpoint_scope_note": (
            "Operational binary status after recombinant production in Escherichia coli; "
            "not intrinsic aqueous or thermodynamic solubility."
        ),
        "canonicalization": {
            "rule": "remove whitespace and convert letters to uppercase",
            "tag_handling": "no tag or cloning-scar removal",
            "allowed_alphabet": "ACDEFGHIKLMNPQRSTVWY",
            "noncanonical_policy": "exclude before modelling; none detected in official splits",
        },
        "duplicate_policy": {
            "identical_label": "collapse to one sequence before splitting",
            "conflicting_label": "exclude from the primary analysis",
            "cross_split": "retain the locked external test and remove exact matches from training",
            "observed_action": "no action required for the official training and test files",
        },
        "source_files": {
            name: {"basename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "split_audits": audits,
        "exact_train_external_sequence_overlap": len(exact_overlap),
        "original_nesg_audit": audit_original_nesg(paths["nesg_csv"], paths["nesg_fasta"], external),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    with (output_dir / "soluprot_provenance_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({
        "train": train_audit,
        "external_test": external_audit,
        "exact_overlap": len(exact_overlap),
        "output_dir": str(output_dir),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
