"""Private-run stage manifests and immutable artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .io_contracts import atomic_write_json, sha256_file
except ImportError:  # pragma: no cover - direct execution path
    from io_contracts import atomic_write_json, sha256_file


def manifest_path(run_dir: Path, stage: str) -> Path:
    return run_dir / "manifests" / f"{stage}.json"


def require_stage(run_dir: Path, stage: str) -> dict:
    path = manifest_path(run_dir, stage)
    if not path.is_file():
        raise FileNotFoundError(f"Required stage manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_stage(run_dir: Path, stage: str, payload: dict, requires: list[str] | None = None) -> Path:
    path = manifest_path(run_dir, stage)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite stage manifest: {path}")
    prior = {}
    for required in requires or []:
        required_path = manifest_path(run_dir, required)
        if not required_path.is_file():
            raise FileNotFoundError(f"Required stage is missing: {required_path}")
        prior[required] = {
            "path": str(required_path),
            "sha256": sha256_file(required_path),
        }
    output = {
        "stage": stage,
        "status": "completed",
        "requires": prior,
        **payload,
    }
    atomic_write_json(output, path)
    return path


def refuse_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis artifact: {path}")
