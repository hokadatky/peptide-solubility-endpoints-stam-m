"""V4 aggregate-release runtime shim with the corrected freeze filename.

The frozen central ``aggregate_release.py`` derives a 20260809 filename for all
non-v1 runs, while the V4 freeze was intentionally created on 20260810.  This
shim changes no scientific result, threshold, or payload field.  It reuses the
frozen release guards and emits the same aggregate schema, but binds provenance
to the actual V4 freeze file.  ``--preflight-only`` performs no writes.
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

from analysis_addon.src.aggregate_release import _write_once, load_private  # noqa: E402
from analysis_addon.src.io_contracts import sha256_file  # noqa: E402
from analysis_addon.src.stage_utils import write_stage  # noqa: E402


EXPECTED_RUN_ID = "addon_20260810_v4"
FREEZE_NAME = "RETROSPECTIVE_ANALYSIS_FREEZE_20260810_V4.md"
REQUIRED_MANIFESTS = [
    "historical-bridge.json",
    "internal-primary.json",
    "external-retrospective.json",
    "modified-peptide-audit.json",
    "shift-audit.json",
]


def resolve_contract(config_path: Path, run_dir: Path) -> tuple[dict, Path]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError(f"Runtime shim is restricted to {EXPECTED_RUN_ID}.")
    run_dir = run_dir.resolve()
    if run_dir != Path(config["paths"]["run_root"]).resolve():
        raise ValueError("--run-dir does not match config run_root.")
    freeze_path = Path(config["paths"]["paper_root"]) / "analysis_addon/protocol" / FREEZE_NAME
    if not freeze_path.is_file():
        raise FileNotFoundError(freeze_path)
    return config, freeze_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    config, freeze_path = resolve_contract(config_path, run_dir)
    missing = [name for name in REQUIRED_MANIFESTS if not (run_dir / "manifests" / name).is_file()]
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_only",
                    "run_id": config["run_id"],
                    "freeze_file": FREEZE_NAME,
                    "freeze_sha256": sha256_file(freeze_path),
                    "missing_required_manifests": missing,
                    "writes_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if missing:
        raise FileNotFoundError(f"Required manifests missing: {missing}")

    historical = load_private(run_dir / "raw_results/historical_bridge.json")
    internal = load_private(run_dir / "raw_results/internal_primary.json")
    external = load_private(run_dir / "raw_results/external_retrospective.json")
    modified = load_private(run_dir / "raw_results/modified_peptide_audit.json")
    shift = load_private(run_dir / "raw_results/shift_audit.json")
    if historical["status"] not in {"completed", "completed_recovered_byte_identical_predictions"}:
        raise ValueError("Historical bridge is not complete.")
    if internal["status"] != "completed" or external["status"] != "completed" or shift["status"] != "completed":
        raise ValueError("One or more required model/audit results are not complete.")
    permutation = internal.get("negative_controls", {}).get("training_label_permutation", {})
    if permutation.get("status") != "PASSED":
        raise ValueError("Training-label negative-control gate did not pass.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "analysis_scope": "retrospective_data_informed",
        "run_id": config["run_id"],
        "endpoint_boundary": "SoluProt values are recombinant soluble-expression status in Escherichia coli, not direct aqueous solubility.",
        "manuscript_integration_authorized": False,
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
    }
    aggregate_soluprot = {
        **common,
        "status": "completed",
        "historical_bridge": historical["historical_bridge"],
        "internal_primary": {
            "split": internal["split"],
            "primary_contrast": internal["primary_contrast"],
            "arms": internal["arms"],
            "fold_stability": internal["fold_stability"],
            "secondary": internal["secondary"],
            "negative_controls": internal["negative_controls"],
            "recovery": internal.get("recovery"),
            "limitations": internal.get("limitations", []),
        },
        "external_retrospective": {
            "split": external["split"],
            "primary_contrast": external["primary_contrast"],
            "arms": external["arms"],
            "calibration_contract": external["calibration_contract"],
        },
        "limitations": [
            "Retrospective and data-informed; historical external labels and prior model results were already viewed.",
            "The endpoint is production-stage soluble-expression status, not direct aqueous peptide solubility.",
            "External performance reflects sequence and source/operator/label-process shift jointly.",
        ],
    }
    aggregate_shift = {
        **common,
        "status": "completed",
        **{
            key: shift[key]
            for key in (
                "split_severity_ladder",
                "source_classifier",
                "external_shift",
                "interpretation_boundary",
            )
        },
    }
    aggregate_modified = {
        **common,
        "status": modified["status"],
        "endpoint": modified["endpoint"],
        "input": modified["input"],
        "collision_audit": modified["collision_audit"],
        "family_feasibility": modified["family_feasibility"],
        "family_model_comparison": modified["family_model_comparison"],
        "status_interpretation": modified["status_interpretation"],
        "limitations": modified["limitations"],
    }
    provenance = {
        **common,
        "status": "completed",
        "config_sha256": sha256_file(config_path),
        "freeze_file": FREEZE_NAME,
        "freeze_sha256": sha256_file(freeze_path),
        "author_decisions_sha256": sha256_file(
            Path(config["paths"]["paper_root"])
            / "analysis_addon/protocol/AUTHOR_DECISIONS_20260809.md"
        ),
        "stage_manifests": {
            name: sha256_file(run_dir / "manifests" / name) for name in REQUIRED_MANIFESTS
        },
        "input_hashes": config["input_hashes"],
        "environment": {
            "python_environment": "sc8",
            "statistics_contract": "paired cluster bootstrap; 10,000 final replicates; seed 20260809",
        },
        "release_policy": config["release"],
        "runtime_release_shim": {
            "reason": "correct frozen 20260809-derived filename to actual V4 freeze dated 20260810",
            "scientific_contract_changed": False,
        },
    }

    _write_once(aggregate_soluprot, output_dir / "aggregate_soluprot.json")
    _write_once(aggregate_shift, output_dir / "aggregate_shift.json")
    _write_once(aggregate_modified, output_dir / "aggregate_modified_peptide.json")
    _write_once(provenance, output_dir / "provenance.json")
    release_files = sorted(output_dir.glob("*.json"))
    write_stage(
        run_dir,
        "aggregate-release",
        {
            "output_dir": str(output_dir),
            "aggregate_sha256": {path.name: sha256_file(path) for path in release_files},
            "status_detail": "aggregate_only_release_completed",
        },
        requires=[
            "historical-bridge",
            "internal-primary",
            "external-retrospective",
            "modified-peptide-audit",
            "shift-audit",
        ],
    )
    print(
        json.dumps(
            {
                "status": "aggregate_release_completed",
                "output_dir": str(output_dir),
                "files": [path.name for path in release_files],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
