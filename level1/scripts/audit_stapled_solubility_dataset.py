#!/usr/bin/env python3
"""Aggregate audit of the public stapled-peptide solubility workbook.

The workbook contains three repeated header blocks.  This script reads only
the public processed-data sheet with formulas disabled and reports counts,
censoring patterns, and representation-support flags.  It never prints raw
peptide identifiers or chemical names.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from audit_common import (
    CANONICAL_SEQUENCE_RE,
    sha256_file,
    validated_input_file,
    write_json_exclusive,
)


SECTION_ROWS = {
    "ATSP_analogues": (4, 367),
    "C_terminal_tail_analogues": (371, 413),
    "D_peptide_analogues": (417, 430),
}
EXPECTED_METADATA_ROWS = {1, 2, 3, 368, 369, 370, 414, 415, 416}
ID_COLUMN = 1
CHEMICAL_NAME_COLUMN = 2
SOLUBILITY_COLUMN = 36  # AJ: kinetic solubility in PBS at pH 7, micromolar


COMPARATOR_RE = re.compile(
    r"^\s*(?P<op>>=|<=|>|<|~)?\s*(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)
CANONICAL_RE = CANONICAL_SEQUENCE_RE
REPRESENTATION_FLAGS = (
    "canonical_linear_one_letter_only",
    "contains_parentheses",
    "contains_digits",
    "contains_lowercase",
    "explicit_terminal_modification",
)


sha256 = sha256_file


def parse_solubility(value: Any) -> tuple[str, float | None]:
    if value is None or value == "":
        return "missing", None
    if isinstance(value, bool):
        return "unparsed_text", None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return "exact_numeric", number
        return "nonfinite_numeric", None
    match = COMPARATOR_RE.match(str(value))
    if not match:
        return "unparsed_text", None
    operator = match.group("op") or "="
    labels = {
        "=": "exact_numeric_string",
        ">": "right_censored_lower_bound",
        ">=": "right_censored_lower_bound",
        "<": "left_censored_upper_bound",
        "<=": "left_censored_upper_bound",
        "~": "approximate_numeric",
    }
    return labels[operator], float(match.group("number"))


def support_flags(chemical_name: str) -> dict[str, bool]:
    compact = chemical_name.replace("-", "").replace(" ", "")
    return {
        "canonical_linear_one_letter_only": bool(CANONICAL_RE.fullmatch(compact)),
        "contains_parentheses": "(" in chemical_name or ")" in chemical_name,
        "contains_digits": any(character.isdigit() for character in chemical_name),
        "contains_lowercase": any(character.islower() for character in chemical_name),
        "explicit_terminal_modification": any(
            token in chemical_name.upper()
            for token in ("AC-", "NH2", "COOH", "N3")
        ),
    }


def validated_input(path: Path, root: Path) -> Path:
    return validated_input_file(path, root, allowed_suffixes={".xlsx"})


def audited_section_rows(sheet: Any) -> dict[int, tuple[Any, Any, Any]]:
    """Read relevant cells once and reject truncated or undeclared table rows."""
    required_max_row = max(
        max(end for _, end in SECTION_ROWS.values()), max(EXPECTED_METADATA_ROWS)
    )
    if sheet.max_row < required_max_row:
        raise ValueError(
            "Supp Data 1 declared worksheet range ends before the audited row boundary "
            f"({sheet.max_row} < {required_max_row})"
        )
    if sheet.max_column < SOLUBILITY_COLUMN:
        raise ValueError(
            "Supp Data 1 declared worksheet range ends before audited column AJ"
        )

    full_rows = {
        row_index: tuple(values)
        for row_index, values in enumerate(
            sheet.iter_rows(
                min_row=1,
                max_row=sheet.max_row,
                min_col=1,
                max_col=sheet.max_column,
                values_only=True,
            ),
            start=1,
        )
    }
    actual_max_row = max(full_rows, default=0)
    if actual_max_row < required_max_row:
        raise ValueError(
            "Supp Data 1 actual worksheet rows end before the audited row boundary "
            f"({actual_max_row} < {required_max_row})"
        )
    covered_rows = {
        row_index
        for start, end in SECTION_ROWS.values()
        for row_index in range(start, end + 1)
    }
    unexpected_rows = []
    for row_index, values in full_rows.items():
        if row_index in EXPECTED_METADATA_ROWS:
            if any(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            ):
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
            f"non-empty rows fall outside audited sections: {unexpected_rows}"
        )
    return {
        row_index: (
            values[ID_COLUMN - 1],
            values[CHEMICAL_NAME_COLUMN - 1],
            values[SOLUBILITY_COLUMN - 1],
        )
        for row_index, values in full_rows.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = validated_input(args.input, args.root)

    workbook = load_workbook(
        source,
        read_only=True,
        data_only=True,
        keep_links=False,
    )
    try:
        sheet = workbook["Supp Data 1"]
        all_ids: list[str] = []
        all_chemical_names: list[str] = []
        censoring = Counter()
        censor_values: dict[str, list[float]] = {}
        textual_categories = Counter()
        support = Counter()
        sections: dict[str, Any] = {}

        sheet_rows = audited_section_rows(sheet)
        rows_without_identifier = 0
        rows_without_chemical_name = 0

        for section, (start, end) in SECTION_ROWS.items():
            section_rows = 0
            section_quantitative_or_bounded = 0
            section_censoring = Counter()
            for row_index in range(start, end + 1):
                identifier, chemical_name, solubility_value = sheet_rows[row_index]
                if identifier in (None, "") and chemical_name in (None, ""):
                    if solubility_value not in (None, ""):
                        raise ValueError(
                            "row inside an audited section carries a solubility value "
                            f"but no identifier or chemical name: row {row_index}"
                        )
                    continue
                section_rows += 1
                if identifier not in (None, ""):
                    all_ids.append(str(identifier).strip())
                else:
                    rows_without_identifier += 1
                if chemical_name not in (None, ""):
                    name = str(chemical_name).strip()
                    all_chemical_names.append(name)
                    for flag, present in support_flags(name).items():
                        if present:
                            support[flag] += 1
                else:
                    rows_without_chemical_name += 1

                category, numeric = parse_solubility(solubility_value)
                censoring[category] += 1
                section_censoring[category] += 1
                if category == "unparsed_text":
                    textual_categories[str(solubility_value).strip()] += 1
                if category not in {"missing", "unparsed_text", "nonfinite_numeric"}:
                    section_quantitative_or_bounded += 1
                if numeric is not None:
                    censor_values.setdefault(category, []).append(numeric)

            sections[section] = {
                "included_rows": section_rows,
                "rows_with_quantitative_or_bounded_solubility": section_quantitative_or_bounded,
                "solubility_value_types": dict(section_censoring),
            }

        # Only counts are published. Exact extrema reproduce a single
        # third-party row verbatim and are withheld; see THIRD_PARTY.md.
        numeric_ranges = {
            category: {"count": len(values)}
            for category, values in censor_values.items()
        }
        report = {
            "source_basename": source.name,
            "source_sha256": sha256(source),
            "sheet": "Supp Data 1",
            "endpoint": "kinetic solubility after dilution from DMSO into PBS at pH 7, filtered and quantified by UPLC/DAD",
            "reported_unit": "micromolar",
            "section_row_ranges_verified": True,
            "expected_metadata_rows": sorted(EXPECTED_METADATA_ROWS),
            "sections": sections,
            "total_included_rows": sum(
                section["included_rows"] for section in sections.values()
            ),
            "rows_without_identifier": rows_without_identifier,
            "rows_without_chemical_name": rows_without_chemical_name,
            "unique_identifier_count": len(set(all_ids)),
            "duplicate_identifier_count": len(all_ids) - len(set(all_ids)),
            "unique_chemical_name_count": len(set(all_chemical_names)),
            "duplicate_chemical_name_count": len(all_chemical_names)
            - len(set(all_chemical_names)),
            "solubility_value_types": dict(censoring),
            "textual_category_counts": dict(textual_categories),
            "numeric_value_counts_by_type": numeric_ranges,
            "representation_support_counts": {
                flag: support.get(flag, 0) for flag in REPRESENTATION_FLAGS
            },
            "representation_interpretation": (
                "Canonical protein pLMs do not natively encode the documented "
                "staples, terminal chemistry, D-residues, and other non-natural "
                "features; lossy sequence projection would alter the scientific input."
            ),
            "limitations": [
                "Rows are close analogues from three related development sections, not a broad general-peptide sample.",
                "Textual outcome categories are retained as aggregate counts and are not converted to numeric values.",
                "The audit counts rows carrying an identifier or a chemical name and applies no peptide-versus-control eligibility rule.",
                "The audit does not infer chemical equivalence from the name strings.",
                "Exact scaffold clustering requires chemically explicit structure processing.",
            ],
        }
        write_json_exclusive(args.output, report)
    finally:
        workbook.close()


if __name__ == "__main__":
    main()
