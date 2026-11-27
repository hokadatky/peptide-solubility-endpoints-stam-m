#!/usr/bin/env python3
"""Build transparent sequence features for the SoluProt benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import Bio
import numpy as np
import pandas as pd
from Bio.SeqUtils import ProtParamData
from Bio.SeqUtils.ProtParam import ProteinAnalysis


AA_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")
KD_SCALE = ProtParamData.kd

AAC_FEATURES = [f"aac_{aa}" for aa in AA_ORDER]
AAC_REFERENCE_RESIDUE = "A"
INDEPENDENT_AAC_FEATURES = [
    feature for feature in AAC_FEATURES if feature != f"aac_{AAC_REFERENCE_RESIDUE}"
]
PRIMARY_PHYSICOCHEMICAL_FEATURES = [
    "log1p_length",
    "isoelectric_point",
    "charge_density_ph7",
    "instability_index",
    "normalized_sequence_entropy",
    "max_kd_window9",
]
DESCRIPTIVE_ONLY_FEATURES = [
    "length",
    "molecular_weight_da",
    "charge_at_ph7",
    "helix_propensity_fraction",
    "turn_propensity_fraction",
    "sheet_propensity_fraction",
    "whole_sequence_gravy",
    "aromaticity_fraction",
    "aliphatic_index",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_entropy(fractions: np.ndarray) -> float:
    positive = fractions[fractions > 0]
    if positive.size <= 1:
        return 0.0
    return float(-(positive * np.log(positive)).sum() / math.log(len(AA_ORDER)))


def maximum_window_hydropathy(sequence: str, window: int = 9) -> float:
    values = np.asarray([KD_SCALE[aa] for aa in sequence], dtype=float)
    if len(values) <= window:
        return float(values.mean())
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    means = (cumulative[window:] - cumulative[:-window]) / window
    return float(means.max())


def aliphatic_index(fractions: dict[str, float]) -> float:
    return 100.0 * (
        fractions["A"]
        + 2.9 * fractions["V"]
        + 3.9 * (fractions["I"] + fractions["L"])
    )


def extract_features(sequence: str) -> dict[str, float]:
    analysis = ProteinAnalysis(sequence)
    counts = analysis.count_amino_acids()
    length = len(sequence)
    fractions = {aa: counts[aa] / length for aa in AA_ORDER}
    fraction_array = np.asarray([fractions[aa] for aa in AA_ORDER], dtype=float)
    charge = float(analysis.charge_at_pH(7.0))
    helix, turn, sheet = analysis.secondary_structure_fraction()

    features: dict[str, float] = {
        **{f"aac_{aa}": fractions[aa] for aa in AA_ORDER},
        "length": float(length),
        "log1p_length": float(math.log1p(length)),
        "molecular_weight_da": float(analysis.molecular_weight()),
        "isoelectric_point": float(analysis.isoelectric_point()),
        "charge_at_ph7": charge,
        "charge_density_ph7": charge / length,
        "instability_index": float(analysis.instability_index()),
        "normalized_sequence_entropy": sequence_entropy(fraction_array),
        "max_kd_window9": maximum_window_hydropathy(sequence, window=9),
        "helix_propensity_fraction": float(helix),
        "turn_propensity_fraction": float(turn),
        "sheet_propensity_fraction": float(sheet),
        "whole_sequence_gravy": float(analysis.gravy()),
        "aromaticity_fraction": float(analysis.aromaticity()),
        "aliphatic_index": aliphatic_index(fractions),
    }
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/solubility/soluprot_train_partitions.csv"),
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("data/solubility/soluprot_external_test_clustered.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/solubility/solubility_transparent_features.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/solubility/solubility_transparent_features.json"),
    )
    parser.add_argument(
        "--development-output",
        type=Path,
        default=Path("data/solubility/solubility_transparent_features_development.csv"),
    )
    parser.add_argument(
        "--external-output",
        type=Path,
        default=Path("data/solubility/solubility_transparent_features_external.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.train.resolve()
    external_path = args.external.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    development_output_path = args.development_output.resolve()
    external_output_path = args.external_output.resolve()

    train = pd.read_csv(train_path, dtype={"record_id": str, "source_id": str})
    external = pd.read_csv(external_path, dtype={"record_id": str, "source_id": str})
    train["analysis_split"] = "development"
    external["analysis_split"] = "external_test"
    combined = pd.concat([train, external], ignore_index=True, sort=False)
    if combined["record_id"].duplicated().any():
        raise ValueError("Record identifiers are not unique across the combined data.")

    extraction_started = time.perf_counter()
    rows = [extract_features(sequence) for sequence in combined["sequence"]]
    extraction_seconds = time.perf_counter() - extraction_started
    features = pd.DataFrame(rows)
    if not np.allclose(features[AAC_FEATURES].sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("Amino-acid fractions do not sum to one.")
    if not np.array_equal(features["length"].astype(int), combined["length"].astype(int)):
        raise RuntimeError("Derived and source sequence lengths differ.")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite transparent feature detected.")

    metadata_columns = [
        "record_id",
        "source_id",
        "label",
        "source_split",
        "analysis_split",
        "partition",
        "cluster_id",
        "external_cluster_id",
    ]
    metadata = combined.reindex(columns=metadata_columns)
    output = pd.concat([metadata.reset_index(drop=True), features], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    development_output = output[output["analysis_split"] == "development"].copy()
    external_output = output[output["analysis_split"] == "external_test"].copy()
    development_output.to_csv(development_output_path, index=False)
    external_output.to_csv(external_output_path, index=False)

    primary_baseline = INDEPENDENT_AAC_FEATURES + PRIMARY_PHYSICOCHEMICAL_FEATURES
    descriptive_baseline = INDEPENDENT_AAC_FEATURES + ["log1p_length"]
    standardized_primary = (
        output.loc[output["analysis_split"] == "development", primary_baseline]
        .astype(float)
        .pipe(lambda frame: (frame - frame.mean()) / frame.std(ddof=0))
        .to_numpy()
    )
    primary_rank = int(np.linalg.matrix_rank(standardized_primary))
    if primary_rank != len(primary_baseline):
        raise RuntimeError(
            f"Primary transparent matrix is rank deficient: rank={primary_rank}, "
            f"columns={len(primary_baseline)}"
        )
    manifest = {
        "rows": int(len(output)),
        "development_rows": int((output["analysis_split"] == "development").sum()),
        "external_test_rows": int((output["analysis_split"] == "external_test").sum()),
        "feature_groups": {
            "aac20_all_descriptive": AAC_FEATURES,
            "independent_aac19_reference_A": INDEPENDENT_AAC_FEATURES,
            "aac_composition_plus_log_length": descriptive_baseline,
            "primary_aac_plus_physicochemical": primary_baseline,
            "descriptive_only_not_in_primary_matrix": DESCRIPTIVE_ONLY_FEATURES,
        },
        "definitions": {
            "aac": "count of each canonical amino acid divided by sequence length",
            "independent_aac_parameterization": (
                "19 amino-acid fractions with alanine omitted as a fixed reference; "
                "alanine fraction is recoverable as one minus the other fractions"
            ),
            "molecular_weight_da": "Biopython ProteinAnalysis.molecular_weight; average isotopic masses",
            "isoelectric_point": "Biopython ProteinAnalysis.isoelectric_point",
            "charge_at_ph7": "Biopython ProteinAnalysis.charge_at_pH(7.0)",
            "charge_density_ph7": "charge_at_ph7 divided by sequence length",
            "instability_index": "Biopython ProtParam instability index based on dipeptide weights",
            "normalized_sequence_entropy": "Shannon entropy of AAC divided by log(20)",
            "max_kd_window9": "maximum mean Kyte-Doolittle hydropathy over contiguous nine-residue windows",
            "secondary_structure_propensities": "Biopython ProtParam helix, turn and sheet propensity fractions; not measured structure",
            "aliphatic_index": "100 * [fraction(A) + 2.9 fraction(V) + 3.9(fraction(I)+fraction(L))]",
        },
        "primary_feature_count": len(primary_baseline),
        "primary_feature_matrix_rank_on_development": primary_rank,
        "compute_cost": {
            "feature_extraction_seconds": extraction_seconds,
            "sequences_per_second": len(combined) / extraction_seconds,
            "residues_per_second": float(combined["length"].sum()) / extraction_seconds,
            "output_bytes": output_path.stat().st_size,
        },
        "input_sha256": {
            "development": sha256(train_path),
            "external_test": sha256(external_path),
        },
        "output_sha256": sha256(output_path),
        "split_output_sha256": {
            "development": sha256(development_output_path),
            "external_test": sha256(external_output_path),
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "biopython": Bio.__version__,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
