#!/usr/bin/env python3
"""Fail-closed static validation of the distillation handoff package."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
PLAN_ROOT = Path(__file__).resolve().parents[1]


REQUIRED = [
    "README.md",
    "config/DISTILLATION_CONFIG_20260813.json",
    "protocol/DISTILLATION_ANALYSIS_FREEZE_20260813.md",
    "protocol/ENDPOINT_PORTABLE_DISTILLATION_BLUEPRINT_20260813.md",
    "protocol/CLAIM_EVIDENCE_LEDGER_20260813.tsv",
    "protocol/MANUSCRIPT_METHOD_AND_POSITIONING_DRAFT_20260813.md",
    "protocol/LUNA_MAX_EXECUTION_INSTRUCTIONS_20260813.md",
    "dry_run/TRIPEPLITE_DRY_RUN_RESULT_20260813.json",
    "dry_run/ESM_EXISTING_CPU_PARITY_DRY_RUN_20260813.json",
    "internal_governance/PLAN_VALIDATION_REPORT_20260813.md",
    "internal_governance/PLAN_SHA256_20260813.txt",
    "src/tripeplite.py",
    "src/teacher_crossfit.py",
    "src/run_distillation_stage.py",
    "src/run_dry_run.py",
    "src/benchmark_student.py",
    "src/hash_protected_roots.py",
    "src/audit_esm_existing_cpu_parity.py",
    "tests/test_tripeplite.py",
    "tests/test_teacher_contracts.py",
    "tests/test_stage_contracts.py",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(relative: str) -> dict[str, object]:
    return json.loads((PLAN_ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == encoded:
            return
        raise FileExistsError(f"Refusing to overwrite a different validation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks: list[str] = []

    for relative in REQUIRED:
        require((PLAN_ROOT / relative).is_file(), f"Required file absent: {relative}")
    checks.append(f"required_files_present:{len(REQUIRED)}")

    project_root = PLAN_ROOT.parents[1]
    manifest = PLAN_ROOT / "internal_governance/PLAN_SHA256_20260813.txt"
    manifest_entries = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected_hash, relative = line.split("  ", 1)
        path = project_root / relative
        require(path.is_file(), f"Manifest file absent: {relative}")
        require(sha256(path) == expected_hash, f"Manifest hash mismatch: {relative}")
        manifest_entries += 1
    require(manifest_entries >= 20, "Manifest is unexpectedly short")
    checks.append(f"frozen_sha256_manifest:{manifest_entries}")

    for path in sorted(PLAN_ROOT.rglob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))
    checks.append("all_tracked_json_parse")

    for path in sorted((PLAN_ROOT / "src").glob("*.py")) + sorted(
        (PLAN_ROOT / "tests").glob("*.py")
    ):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    checks.append("all_python_ast_parse")

    config = load_json("config/DISTILLATION_CONFIG_20260813.json")
    require(config["schema_version"] == "soluprot-distillation-contract-v1", "schema")
    environment = config["environment"]
    require(environment["embedding_resume_device"] == "cpu", "ESM resume device")
    require(
        environment["embedding_effective_dtype_on_cpu"] == "float32",
        "ESM CPU dtype",
    )
    require(environment["student_primary_device"] == "cpu", "student device")
    roles = config["data_contract"]
    design_train = set(roles["design_train_folds"])
    design_gate = {roles["design_gate_fold"]}
    confirmation = {roles["confirmation_fold"]}
    require(not design_train & design_gate & confirmation, "fold roles")
    require(not design_train & design_gate, "design train/gate overlap")
    require(not design_train & confirmation, "design train/confirmation overlap")
    require(not design_gate & confirmation, "gate/confirmation overlap")
    require(
        set(roles["confirmation_train_folds"]) == design_train | design_gate,
        "confirmation refit folds",
    )
    require(config["student"]["artifact_size_limit_bytes"] == 5 * 1024 * 1024, "size gate")
    require(config["external"]["status"].startswith("optional_"), "external boundary")
    require(config["release"]["app_edit_allowed"] is False, "app protection")
    require(config["release"]["manuscript_edit_allowed"] is False, "manuscript protection")
    checks.append("frozen_config_contract")

    dry = load_json("dry_run/TRIPEPLITE_DRY_RUN_RESULT_20260813.json")
    require(dry["status"] == "PASS", "TriPepLite dry run")
    require(dry["student"]["parameter_count"] == 265633, "parameter count")
    require(dry["student"]["teacher_required_at_inference"] is False, "teacher-free")
    require(
        dry["student"]["torch_numpy_probability_max_abs_difference"] < 1e-6,
        "inference parity",
    )
    require(dry["tiny_real"]["performance_metrics_computed"] is False, "dry metrics")
    require(not any(dry["privacy"].values()), "dry-run privacy")
    checks.append("bounded_student_dry_run")

    esm = load_json("dry_run/ESM_EXISTING_CPU_PARITY_DRY_RUN_20260813.json")
    require(esm["status"] == "PASS", "ESM parity")
    require(
        esm["historical_per_shard_device_provenance"] == "not recorded",
        "historical device provenance boundary",
    )
    require([row["shard_index"] for row in esm["shards"]] == [0, 776], "ESM shards")
    require(all(row["status"] == "PASS" for row in esm["shards"]), "ESM checks")
    require(not any(esm["privacy"].values()), "ESM audit privacy")
    checks.append("bounded_esm_cpu_parity")

    runbook = (
        PLAN_ROOT / "protocol/LUNA_MAX_EXECUTION_INSTRUCTIONS_20260813.md"
    ).read_text(encoding="utf-8")
    cpu_command_lines = sum(
        line.strip() == "--device cpu \\" for line in runbook.splitlines()
    )
    require(cpu_command_lines == 2, "two CPU extraction commands")
    require("--device auto" not in runbook, "auto device remains in runbook")
    require("--device mps" not in runbook, "MPS device remains in runbook")
    require("Expected: `16 passed`" in runbook, "test-count contract")
    require("Phase I — external transfer: HARD STOP" in runbook, "external stop")
    require("Phase J — manuscript and app: HARD STOP" in runbook, "promotion stop")
    checks.append("luna_runbook_static_contract")

    with (PLAN_ROOT / "protocol/CLAIM_EVIDENCE_LEDGER_20260813.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        ledger = {row["claim_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    require(ledger["D-006"]["evidence_status"] == "UNTESTED", "distillation claim")
    require(
        ledger["D-009"]["evidence_status"]
        == "METHOD_GENERALIZATION_NOT_EMPIRICAL_TRANSFER",
        "portability claim",
    )
    require(ledger["D-010"]["evidence_status"] == "FALSE", "endpoint boundary")
    checks.append("claim_evidence_boundaries")

    payload = {
        "status": "PASS",
        "scope": "static contract and bounded dry-run evidence; no scientific model fit",
        "checks": checks,
        "required_file_count": len(REQUIRED),
        "scientific_training_started": False,
        "external_evaluation_started": False,
        "manuscript_or_app_modified": False,
    }
    atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
