"""Fail-closed aggregate privacy and frozen-provenance audit."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import yaml

try:
    from .io_contracts import atomic_write_json, sha256_file
    from .preflight import EXPECTED_HASH_FILES, source_tree_hash
    from .stage_utils import require_stage, refuse_existing, write_stage
except ImportError:  # pragma: no cover - direct execution path
    from io_contracts import atomic_write_json, sha256_file
    from preflight import EXPECTED_HASH_FILES, source_tree_hash
    from stage_utils import require_stage, refuse_existing, write_stage


FORBIDDEN_RELEASE_SUFFIXES = {".csv", ".tsv", ".npy", ".npz", ".parquet", ".xlsx", ".xls"}
ROW_LEVEL_KEYS = {"record_id", "source_id", "peptide_id", "sequence", "sequences"}


def _walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    else:
        yield None, value


def _validate_release_json(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contains_row_level_data") is not False:
        raise ValueError(f"Missing row-level release guard: {path}")
    if payload.get("contains_sequences") is not False:
        raise ValueError(f"Missing sequence release guard: {path}")
    if payload.get("contains_record_ids") is not False:
        raise ValueError(f"Missing record-ID release guard: {path}")
    for key, value in _walk(payload):
        if key is not None and key.lower() in ROW_LEVEL_KEYS:
            raise ValueError(f"Row-level key in aggregate JSON: {path.name}:{key}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite value in aggregate JSON: {path}")
        if isinstance(value, str) and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{50,}", value):
            raise ValueError(f"Sequence-like string in aggregate JSON: {path}")


def _denylist_match_count(denylist: Path, roots: list[Path]) -> tuple[int, int]:
    if not denylist.is_file():
        raise FileNotFoundError(f"Private denylist is missing: {denylist}")
    tokens = [line.strip().casefold() for line in denylist.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not tokens:
        raise ValueError("Private denylist is empty.")
    matches = 0
    scanned = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            scanned += 1
            content = path.read_bytes().lower()
            matches += sum(1 for token in tokens if token.encode("utf-8") in content)
    return matches, scanned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    release_dir = args.release_dir.resolve()
    if run_dir != Path(config["paths"]["run_root"]).resolve():
        raise ValueError("--run-dir does not match config run_root.")
    require_stage(run_dir, "aggregate-release")

    release_files = sorted(path for path in release_dir.rglob("*") if path.is_file())
    if not release_files:
        raise FileNotFoundError(f"No aggregate release files found: {release_dir}")
    bad_suffixes = sorted(str(path) for path in release_files if path.suffix.lower() in FORBIDDEN_RELEASE_SUFFIXES)
    if bad_suffixes:
        raise ValueError(f"Row-level-capable file types in release directory: {bad_suffixes}")
    json_files = [path for path in release_files if path.suffix.lower() == ".json"]
    if len(json_files) != len(release_files):
        raise ValueError("The aggregate release directory must contain JSON only.")
    for path in json_files:
        _validate_release_json(path)

    paper_root = Path(config["paths"]["paper_root"]).resolve()
    figures_root = paper_root / "figures_addon"
    denylist = run_dir / "privacy/private_denylist.txt"
    denylist_matches, files_scanned = _denylist_match_count(
        denylist, [paper_root / "analysis_addon", figures_root]
    )
    if denylist_matches:
        raise ValueError("Private denylist matched one or more tracked add-on artifacts.")

    preflight_path = run_dir / "manifests/preflight_manifest.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    current_code_hash = source_tree_hash(paper_root / "analysis_addon")
    if current_code_hash != preflight["code_hash"]:
        raise ValueError("Analysis source tree changed after the v3 freeze.")
    if sha256_file(config_path) != preflight["config_sha256"]:
        raise ValueError("Analysis config changed after preflight.")
    observed_inputs = {
        key: sha256_file(Path(config["paths"]["project_root"]) / relative)
        for key, relative in EXPECTED_HASH_FILES.items()
    }
    if observed_inputs != config["input_hashes"]:
        raise ValueError("One or more input canaries changed after preflight.")

    required_stages = [
        "recovery-v2",
        "historical-bridge",
        "internal-primary",
        "external-retrospective",
        "modified-peptide-audit",
        "shift-audit",
        "aggregate-release",
    ]
    manifests = {stage: require_stage(run_dir, stage) for stage in required_stages}
    payload = {
        "status": "completed",
        "run_id": config["run_id"],
        "aggregate_json_files": {path.name: sha256_file(path) for path in json_files},
        "aggregate_json_file_count": int(len(json_files)),
        "forbidden_release_suffix_count": 0,
        "denylist_files_scanned": int(files_scanned),
        "private_denylist_match_count": 0,
        "config_sha256_matches_preflight": True,
        "source_tree_sha256_matches_preflight": True,
        "input_canaries_match_preflight": True,
        "stage_manifest_sha256": {
            stage: sha256_file(run_dir / "manifests" / f"{stage}.json") for stage in manifests
        },
        "release_json_recursive_privacy_check": "passed",
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
    }
    result_path = run_dir / "raw_results/privacy_reproducibility_audit.json"
    refuse_existing(result_path)
    atomic_write_json(payload, result_path)
    write_stage(
        run_dir,
        "privacy-reproducibility-audit",
        {"result_path": str(result_path), "status_detail": payload["status"]},
        requires=["aggregate-release"],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
