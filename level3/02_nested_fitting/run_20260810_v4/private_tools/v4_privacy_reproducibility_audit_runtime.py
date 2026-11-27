"""V4 privacy/reproducibility audit shim without the V3-only recovery stage.

The frozen audit unconditionally requires ``recovery-v2``.  V4 deliberately
contains no ``recovery`` config and recomputes all affected internal arms, so
that stage cannot legitimately exist.  This shim preserves every privacy,
canary, source-hash, and manifest-hash check while using the actual V4 stage
graph.  ``--preflight-only`` performs no writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
PAPER = ROOT / "paper/peptide_solubility_review"
sys.path.insert(0, str(PAPER))

from analysis_addon.src.io_contracts import atomic_write_json, sha256_file  # noqa: E402
from analysis_addon.src.preflight import EXPECTED_HASH_FILES, source_tree_hash  # noqa: E402
from analysis_addon.src.privacy_reproducibility_audit import (  # noqa: E402
    FORBIDDEN_RELEASE_SUFFIXES,
    _denylist_match_count,
    _validate_release_json,
)
from analysis_addon.src.stage_utils import refuse_existing, require_stage, write_stage  # noqa: E402


EXPECTED_RUN_ID = "addon_20260810_v4"
REQUIRED_STAGES = [
    "historical-bridge",
    "internal-primary",
    "external-retrospective",
    "modified-peptide-audit",
    "shift-audit",
    "aggregate-release",
]


def base_checks(config_path: Path, run_dir: Path) -> tuple[dict, dict, dict[str, str]]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError(f"Runtime shim is restricted to {EXPECTED_RUN_ID}.")
    if "recovery" in config:
        raise ValueError("V4 privacy shim refuses configs containing a recovery stage.")
    run_dir = run_dir.resolve()
    if run_dir != Path(config["paths"]["run_root"]).resolve():
        raise ValueError("--run-dir does not match config run_root.")
    preflight_path = run_dir / "manifests/preflight_manifest.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    paper_root = Path(config["paths"]["paper_root"]).resolve()
    current_code_hash = source_tree_hash(paper_root / "analysis_addon")
    if current_code_hash != preflight["code_hash"]:
        raise ValueError("Analysis source tree changed after the V4 freeze.")
    if sha256_file(config_path) != preflight["config_sha256"]:
        raise ValueError("Analysis config changed after preflight.")
    observed_inputs = {
        key: sha256_file(Path(config["paths"]["project_root"]) / relative)
        for key, relative in EXPECTED_HASH_FILES.items()
    }
    if observed_inputs != config["input_hashes"]:
        raise ValueError("One or more input canaries changed after preflight.")
    return config, preflight, observed_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    release_dir = args.release_dir.resolve()
    config, preflight, observed_inputs = base_checks(config_path, run_dir)
    missing_stages = [
        stage for stage in REQUIRED_STAGES if not (run_dir / "manifests" / f"{stage}.json").is_file()
    ]
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_only",
                    "run_id": config["run_id"],
                    "recovery_stage_required": False,
                    "source_tree_sha256": preflight["code_hash"],
                    "config_sha256": preflight["config_sha256"],
                    "input_canaries_checked": len(observed_inputs),
                    "missing_required_stages": missing_stages,
                    "writes_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if missing_stages:
        raise FileNotFoundError(f"Required V4 stages missing: {missing_stages}")

    require_stage(run_dir, "aggregate-release")
    release_files = sorted(path for path in release_dir.rglob("*") if path.is_file())
    if not release_files:
        raise FileNotFoundError(f"No aggregate release files found: {release_dir}")
    bad_suffixes = sorted(
        str(path) for path in release_files if path.suffix.lower() in FORBIDDEN_RELEASE_SUFFIXES
    )
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

    manifests = {stage: require_stage(run_dir, stage) for stage in REQUIRED_STAGES}
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
        "runtime_stage_graph_correction": {
            "recovery_v2_required": False,
            "reason": "V4 has no recovery config and recomputes all affected internal arms",
            "scientific_contract_changed": False,
        },
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
