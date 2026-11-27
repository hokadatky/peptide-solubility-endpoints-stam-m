#!/usr/bin/env python3
"""Create a strict derived CSV by removing verified export-index columns.

Raw public files are never modified.  A column is removed only when its header
is empty or starts with ``Unnamed:`` and its values form an integer row index
(0..n-1 or 1..n).  All other cells are preserved byte-for-value after CSV
round-tripping.  A JSON manifest records the transformation and checksums.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from audit_common import sha256_file, validated_input_file, write_json_exclusive


sha256 = sha256_file


def is_index(values: list[str]) -> bool:
    if not values:
        return False
    try:
        numbers = [int(value.strip()) for value in values]
    except (TypeError, ValueError):
        return False
    return numbers == list(range(len(numbers))) or numbers == list(
        range(1, len(numbers) + 1)
    )


def normalize(source: Path, destination: Path, root: Path) -> dict[str, object]:
    if source.suffix.lower() != ".csv" or destination.suffix.lower() != ".csv":
        raise ValueError("source and destination must be CSV files")
    try:
        source = validated_input_file(source, root, allowed_suffixes={".csv"})
    except ValueError as error:
        raise ValueError(str(error).replace("input", "source", 1)) from error
    if destination.exists():
        raise FileExistsError(destination)

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, strict=True))
    if not rows:
        raise ValueError("empty CSV")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("non-rectangular CSV")

    headers = rows[0]
    data = rows[1:]
    removable: list[int] = []
    reasons: list[dict[str, object]] = []
    for index, header in enumerate(headers):
        candidate = not header.strip() or header.strip().lower().startswith("unnamed:")
        if not candidate:
            continue
        values = [row[index] for row in data]
        if all(not value.strip() for value in values):
            removable.append(index)
            reasons.append({"column_index_1based": index + 1, "reason": "all-empty export column"})
        elif is_index(values):
            removable.append(index)
            reasons.append({"column_index_1based": index + 1, "reason": "verified sequential export index"})

    if not removable:
        raise ValueError("no safely removable export-index column was found")
    if len(removable) == width:
        raise ValueError("every column was classified as removable")
    keep = [index for index in range(width) if index not in removable]
    if any(not headers[index].strip() for index in keep):
        raise ValueError("an empty non-index header remains")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows([[row[index] for index in keep] for row in rows])

    return {
        "source_basename": source.name,
        "source_sha256": sha256(source),
        "derived_basename": destination.name,
        "derived_sha256": sha256(destination),
        "data_rows": len(data),
        "original_columns": width,
        "derived_columns": len(keep),
        "removed_columns": reasons,
        "cell_values_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    report = normalize(args.source, args.destination, args.root)
    write_json_exclusive(args.manifest, report)


if __name__ == "__main__":
    main()
