#!/usr/bin/env python3
"""Shared fail-closed helpers for the public endpoint-audit scripts.

The audit commands intentionally accept only regular files contained by an
explicitly approved source root.  Keeping those checks here prevents the four
command-line entry points from drifting apart as the audit is revised.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Collection, Mapping


CANONICAL_SEQUENCE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{2,100}$")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_finite_number(value: Any) -> bool:
    """Return True only for finite, non-boolean Python numeric values."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validated_input_file(
    path: Path,
    root: Path,
    *,
    allowed_suffixes: Collection[str],
) -> Path:
    """Resolve and validate an input against an explicitly approved root.

    Symlinks are rejected before resolution, and the resolved file must remain
    below ``root``.  Suffixes are compared case-insensitively.
    """
    normalized_suffixes = {suffix.lower() for suffix in allowed_suffixes}
    if not normalized_suffixes or any(
        not suffix.startswith(".") for suffix in normalized_suffixes
    ):
        raise ValueError("allowed_suffixes must contain dot-prefixed extensions")
    if path.is_symlink():
        raise ValueError("input must not be a symbolic link")
    if not path.is_file():
        suffix_label = "/".join(sorted(normalized_suffixes))
        raise ValueError(f"input must be a regular {suffix_label} file")
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("input must be inside the approved source root") from error
    suffix_label = "/".join(sorted(normalized_suffixes))
    if resolved_path.suffix.lower() not in normalized_suffixes:
        raise ValueError(f"input must be a regular {suffix_label} file")
    return resolved_path


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable UTF-8 JSON and refuse to overwrite an existing result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
