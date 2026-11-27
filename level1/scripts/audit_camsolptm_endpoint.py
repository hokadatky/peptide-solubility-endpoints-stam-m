#!/usr/bin/env python3
"""Aggregate endpoint audit for the public CamSol-PTM workbooks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from audit_common import (
    CANONICAL_SEQUENCE_RE,
    is_finite_number,
    sha256_file,
    validated_input_file,
    write_json_exclusive,
)


GROUP_ROWS = {
    "PYY_PEG_precipitant_midpoint": (4, 18),
    "18A_ammonium_sulfate_midpoint": (25, 33),
    "GLP1_ultracentrifugation_supernatant": (40, 49),
}
EXPECTED_METADATA_ROWS = {1, 3, 22, 24, 37, 39}
CANONICAL_RE = CANONICAL_SEQUENCE_RE
sha256 = sha256_file


def validated_input(path: Path, root: Path) -> Path:
    return validated_input_file(path, root, allowed_suffixes={".xlsx"})


def audited_figure_rows(figure_sheet: Any) -> dict[int, list[Any]]:
    """Read the bounded Figure 3 region once and reject undeclared content."""
    required_max_row = max(
        max(end for _, end in GROUP_ROWS.values()), max(EXPECTED_METADATA_ROWS)
    )
    if figure_sheet.max_row < required_max_row:
        raise ValueError(
            "Figure3 declared worksheet range ends before the audited row boundary "
            f"({figure_sheet.max_row} < {required_max_row})"
        )
    if figure_sheet.max_column < 5:
        raise ValueError(
            "Figure3 declared worksheet range ends before audited column E"
        )

    rows = {
        row_index: list(values)
        for row_index, values in enumerate(
            figure_sheet.iter_rows(
                min_row=1,
                max_row=figure_sheet.max_row,
                min_col=1,
                max_col=figure_sheet.max_column,
                values_only=True,
            ),
            start=1,
        )
    }
    actual_max_row = max(rows, default=0)
    if actual_max_row < required_max_row:
        raise ValueError(
            "Figure3 actual worksheet rows end before the audited row boundary "
            f"({actual_max_row} < {required_max_row})"
        )
    covered_rows = {
        row_index
        for start, end in GROUP_ROWS.values()
        for row_index in range(start, end + 1)
    }
    unexpected_rows = []
    for row_index, values in rows.items():
        if row_index in EXPECTED_METADATA_ROWS:
            if any(is_finite_number(value) for value in values):
                raise ValueError(
                    f"numeric value found in expected metadata row {row_index}"
                )
            continue
        if row_index not in covered_rows and any(
            value not in (None, "") for value in values
        ):
            unexpected_rows.append(row_index)
    if unexpected_rows:
        raise ValueError(
            f"non-empty Figure3 rows fall outside audited groups: {unexpected_rows}"
        )
    return rows


def audit_inputs(supp_path: Path, source_path: Path, root: Path) -> dict[str, Any]:
    supp_path = validated_input(supp_path, root)
    source_path = validated_input(source_path, root)

    supp_book = load_workbook(
        supp_path, read_only=True, data_only=True, keep_links=False
    )
    source_book = load_workbook(
        source_path, read_only=True, data_only=True, keep_links=False
    )
    try:
        sequence_sheet = supp_book["Table for SI"]
        sequence_values: list[str] = []
        for row in sequence_sheet.iter_rows(min_row=2, values_only=True):
            value = row[1] if len(row) > 1 else None
            if value not in (None, ""):
                sequence_values.append(str(value).strip())

        figure_sheet = source_book["Figure3"]
        figure_rows = audited_figure_rows(figure_sheet)

        group_summary: dict[str, Any] = {}
        measured_rows = 0
        for group, (start, end) in GROUP_ROWS.items():
            rows = 0
            rows_with_prediction = 0
            rows_with_experiment = 0
            for row_index in range(start, end + 1):
                row = figure_rows[row_index][:5]
                if not any(value not in (None, "") for value in row):
                    continue
                rows += 1
                # In every Figure 3 block, column B is the CamSol prediction
                # and column C is the reported experimental endpoint.
                if is_finite_number(row[1]):
                    rows_with_prediction += 1
                if is_finite_number(row[2]):
                    rows_with_experiment += 1
                    measured_rows += 1
            group_summary[group] = {
                "figure_rows": rows,
                "rows_with_CamSol_prediction": rows_with_prediction,
                "rows_with_experimental_endpoint": rows_with_experiment,
            }

        return {
            "supplementary_data_basename": supp_path.name,
            "supplementary_data_sha256": sha256(supp_path),
            "source_data_basename": source_path.name,
            "source_data_sha256": sha256(source_path),
            "sequence_table_rows": len(sequence_values),
            "unique_sequence_strings": len(set(sequence_values)),
            "canonical_one_letter_sequence_strings": sum(
                bool(CANONICAL_RE.fullmatch(value)) for value in sequence_values
            ),
            "group_row_ranges_verified": True,
            "expected_metadata_rows": sorted(EXPECTED_METADATA_ROWS),
            "Figure3_groups": group_summary,
            "total_rows_with_experimental_endpoint": measured_rows,
            "endpoint_interpretation": (
                "The three scaffolds use two precipitant-midpoint assays and one "
                "ultracentrifugation-supernatant assay. Their numerical outcomes "
                "must be analysed by scaffold/assay and are not a single pooled logS target."
            ),
            "representation_interpretation": (
                "Most variants contain non-natural residues whose chemical identity "
                "is not represented by canonical ProtT5 or ESM-2 residue tokens."
            ),
            "limitations": [
                "Only three related peptide scaffolds were experimentally studied.",
                "The 40,000 and 50,000 scale values in the paper are computational predictions, not experimental solubility records.",
                "The audit reports aggregate counts and does not certify chemical-entity canonicalization.",
            ],
        }
    finally:
        supp_book.close()
        source_book.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("supplementary_data", type=Path)
    parser.add_argument("source_data", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = audit_inputs(args.supplementary_data, args.source_data, args.root)
    write_json_exclusive(args.output, report)


if __name__ == "__main__":
    main()
