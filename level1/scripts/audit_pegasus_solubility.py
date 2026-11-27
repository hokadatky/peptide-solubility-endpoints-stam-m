#!/usr/bin/env python3
"""Aggregate audit of PEGASUS public in-house aqueous-solubility records."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from math import isfinite
from pathlib import Path

from audit_common import (
    CANONICAL_SEQUENCE_RE,
    sha256_file,
    validated_input_file,
    write_json_exclusive,
)


CANONICAL_RE = CANONICAL_SEQUENCE_RE
sha256 = sha256_file


def finite_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def validated_input(path: Path, root: Path) -> Path:
    return validated_input_file(path, root, allowed_suffixes={".csv"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--repository-commit",
        required=True,
        help="Upstream commit of the source repository, recorded as provenance",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.repository_commit):
        raise ValueError("repository commit must be a full 40-character SHA-1")
    source = validated_input(args.input, args.root)

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        required = {
            "Sequence",
            "IsomericSmiles",
            "Cyc",
            "Group",
            "Solubility (um)",
            "LogS",
            "NMethyl",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("required columns are missing")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("duplicate columns are not allowed")
        rows = list(reader)

    for row_number, row in enumerate(rows, start=2):
        for field in ("Sequence", "IsomericSmiles", "Cyc", "Group"):
            value = row.get(field)
            if value is None or not value.strip():
                raise ValueError(f"blank {field} at CSV row {row_number}")
        for field in ("Solubility (um)", "LogS", "NMethyl"):
            value = row.get(field)
            if value is None or finite_float(value.strip()) is None:
                raise ValueError(f"non-finite {field} at CSV row {row_number}")

    sequences = [row["Sequence"].strip() for row in rows]
    structures = [row["IsomericSmiles"].strip() for row in rows]
    solubility = [finite_float(row["Solubility (um)"].strip()) for row in rows]
    logs = [finite_float(row["LogS"].strip()) for row in rows]
    nmethyl = [finite_float(row["NMethyl"].strip()) for row in rows]
    groups = Counter(row["Group"].strip() for row in rows)
    cyclizations = Counter(row["Cyc"].strip() for row in rows)

    numeric_solubility = [value for value in solubility if value is not None]
    if not numeric_solubility:
        raise ValueError("no finite numeric solubility values were found")
    sequence_to_structures: dict[str, set[str]] = {}
    for sequence, structure in zip(sequences, structures, strict=True):
        sequence_to_structures.setdefault(sequence, set()).add(structure)
    collision_sizes = sorted(
        (len(values) for values in sequence_to_structures.values() if len(values) > 1),
        reverse=True,
    )
    report = {
        "source_basename": source.name,
        "source_sha256": sha256(source),
        "repository_commit": args.repository_commit,
        "data_rows": len(rows),
        "unique_sequence_string_count": len(set(sequences)),
        "unique_isomeric_SMILES_count": len(set(structures)),
        "sequence_to_structure_collision_group_count": len(collision_sizes),
        "sequence_to_structure_collision_group_sizes": collision_sizes,
        "chemical_distinctions_lost_by_sequence_projection": sum(
            size - 1 for size in collision_sizes
        ),
        "canonical_one_letter_sequence_count": sum(
            bool(CANONICAL_RE.fullmatch(sequence)) for sequence in sequences
        ),
        "rows_with_numeric_solubility": len(numeric_solubility),
        "rows_with_numeric_logS": sum(value is not None for value in logs),
        # Prespecified coarse bins only; exact extrema are withheld because
        # each reproduces a single third-party row (see THIRD_PARTY.md).
        "solubility_micromolar_summary": {
            "below_10": sum(value < 10 for value in numeric_solubility),
            "10_to_100": sum(10 <= value <= 100 for value in numeric_solubility),
            "above_100": sum(value > 100 for value in numeric_solubility),
        },
        "rows_with_N_methylation_count_greater_than_zero": sum(
            value is not None and value > 0 for value in nmethyl
        ),
        "distinct_group_count": len(groups),
        "group_sizes_without_labels": sorted(groups.values(), reverse=True),
        "distinct_cyclization_label_count": len(cyclizations),
        "cyclization_group_sizes_without_labels": sorted(
            cyclizations.values(), reverse=True
        ),
        "endpoint": (
            "aqueous solubility in HBSS at pH 7.4 after 18 hours, quantified "
            "by LC-MS/MS, as described in the primary article"
        ),
        "sampling_interpretation": (
            "The data are a small, deliberately designed cyclic-peptide set and "
            f"commercial controls; {sum(value > 100 for value in numeric_solubility)} "
            f"of the {len(numeric_solubility)} reported values exceed 100 micromolar."
        ),
        "representation_interpretation": (
            "Repeated sequence strings can correspond to distinct stereochemical "
            "structures, and cyclization/N-methylation are not faithfully represented "
            "by canonical ProtT5 or ESM-2 tokens."
        ),
        "limitations": [
            "The massively parallel PEGASUS training signals are permeability proxies, not direct solubility measurements.",
            "The 40 direct-solubility rows are too small and too narrowly designed for a general pLM superiority benchmark.",
            "The repository license is CC BY-NC-ND 4.0; redistribution of transformed data requires careful rights review.",
        ],
    }
    write_json_exclusive(args.output, report)


if __name__ == "__main__":
    main()
