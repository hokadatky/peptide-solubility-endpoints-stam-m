"""Phase 0 preflight and retrospective analysis freeze writer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:  # support both ``python -m`` and the execution-plan direct CLI
    from .io_contracts import atomic_write_json, atomic_write_text, sha256_file
except ImportError:  # pragma: no cover - exercised by the direct CLI
    from io_contracts import atomic_write_json, atomic_write_text, sha256_file


REQUIRED_TOP_KEYS = {"run_id", "analysis_scope", "global_seed", "paths", "input_hashes", "statistics", "soluprot", "historical_bridge", "shift", "modified", "release"}
OPTIONAL_TOP_KEYS = {"recovery", "execution"}
EXPECTED_HASH_FILES = {
    "train_partitions": "data/solubility/soluprot_train_partitions.csv",
    "external_clustered": "data/solubility/soluprot_external_test_clustered.csv",
    "development_features": "data/solubility/solubility_transparent_features_development.csv",
    "external_features": "data/solubility/solubility_transparent_features_external.csv",
    "feature_manifest": "data/solubility/solubility_transparent_features.json",
    "embedding_index": "data/solubility/embeddings/soluprot_embedding_index.csv",
    "prott5_array": "data/solubility/embeddings/soluprot_prott5.npy",
}


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Analysis config must be a mapping.")
    unknown = sorted(set(config) - REQUIRED_TOP_KEYS - OPTIONAL_TOP_KEYS)
    missing = sorted(REQUIRED_TOP_KEYS - set(config))
    if unknown or missing:
        raise ValueError(f"Config key contract failed: unknown={unknown}, missing={missing}")
    if config["analysis_scope"] != "retrospective_data_informed":
        raise ValueError("analysis_scope must be retrospective_data_informed.")
    recovery = config.get("recovery")
    if recovery is not None:
        required_recovery = {
            "source_run_id",
            "source_run_root",
            "import_historical_bridge",
            "import_secondary_oof",
            "verify_recomputed_reference_against_v2",
            "deferred_secondary_arms",
            "planned_secondary_family_size",
            "deferred_arm_multiplicity_p_value",
            "reason",
        }
        unknown_recovery = sorted(set(recovery) - required_recovery)
        missing_recovery = sorted(required_recovery - set(recovery))
        if unknown_recovery or missing_recovery:
            raise ValueError(
                "Recovery config key contract failed: "
                f"unknown={unknown_recovery}, missing={missing_recovery}"
            )
        if recovery["source_run_id"] == config["run_id"]:
            raise ValueError("Recovery source_run_id must differ from the active run_id.")
        if float(recovery["deferred_arm_multiplicity_p_value"]) != 1.0:
            raise ValueError("A deferred secondary arm must enter multiplicity correction with p=1.0.")
    execution = config.get("execution")
    if execution is not None:
        if int(execution["logistic_max_iter"]) != 20_000:
            raise ValueError("V4 logistic_max_iter must be frozen at 20,000.")
        if execution["inner_nonconverged_candidate"] != "ineligible" or execution["outer_nonconvergence"] != "hard_stop":
            raise ValueError("V4 convergence policy is not fail-closed.")
        if int(execution["holm_family_size"]) != 4 or float(execution["deferred_arm_p_value_for_multiplicity"]) != 1.0:
            raise ValueError("V4 deferred-arm multiplicity contract failed.")
    return config


def validate_author_decisions(path: Path, config: dict) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if "APPROVED FOR RETROSPECTIVE REANALYSIS" not in text:
        raise ValueError("Author decisions are not approved for retrospective reanalysis.")
    if text.count("[x] Permit retrospective reanalysis") != 1:
        raise ValueError("Governance decision is not uniquely approved.")
    if text.count("[x] No manuscript integration before author review") != 1:
        raise ValueError("Manuscript integration decision is not locked to no integration.")
    required_values = {
        "auc_gain_delta = 0.02": 0.02,
        "log10_rmse_gain_delta = 0.10": 0.10,
        "spearman_gain_delta = 0.05": 0.05,
    }
    for marker in required_values:
        if marker not in text:
            raise ValueError(f"Author-approved margin missing: {marker}")
    if "- encoder: `ProtT5`" not in text or "- primary learner: `L2 logistic regression`" not in text or "- sensitivity learner: fixed `HistGradientBoosting`" not in text:
        raise ValueError("Primary model contract is not confirmed.")
    if "Author name: __________________" in text or "Author name:" not in text:
        raise ValueError("Author identity is missing.")
    if "Hiroyuki Okada" not in text:
        raise ValueError("Expected explicit author authorization is missing.")
    return {
        "status": "approved",
        "governance": "retrospective_reanalysis_of_recombinant_expression_worked_example",
        "manuscript_integration": "forbidden_until_author_review",
        "margins": {key: value for key, value in required_values.items()},
        "primary_contract": {
            "encoder": "ProtT5",
            "learner": "L2 logistic regression",
            "sensitivity_learner": "fixed HistGradientBoosting",
        },
        "author_decisions_sha256": sha256_file(path),
    }


def source_tree_hash(source_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path for path in source_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and "results" not in path.relative_to(source_root).parts
        and path.suffix in {".py", ".yaml", ".json"}
    )
    for path in paths:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_summary() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "PyYAML", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def write_freeze(config: dict, author: dict[str, object], config_path: Path, author_path: Path) -> tuple[Path, str]:
    paper_root = Path(config["paths"]["paper_root"]).resolve()
    is_v4 = str(config["run_id"]).endswith("_v4")
    freeze_date = "2026-08-10" if is_v4 else "2026-08-09"
    date_token = "20260810" if is_v4 else "20260809"
    freeze_name = f"RETROSPECTIVE_ANALYSIS_FREEZE_{date_token}.md" if str(config["run_id"]).endswith("_v1") else f"RETROSPECTIVE_ANALYSIS_FREEZE_{date_token}_{str(config['run_id']).split('_')[-1].upper()}.md"
    freeze_path = paper_root / "analysis_addon/protocol" / freeze_name
    code_hash = source_tree_hash(paper_root / "analysis_addon")
    input_hashes = config["input_hashes"]
    lines = [
        f"# Retrospective analysis freeze — {freeze_date}",
        "",
        "Status: **retrospective, data-informed analysis freeze**.",
        "",
        "This freeze records an already-viewed and already-developed worked-example audit. It is not a preregistration and does not establish a locked external benchmark.",
        "",
        "## Author authorization",
        "",
        "- Governance: retrospective reanalysis of the recombinant soluble-expression worked example is authorized.",
        "- Manuscript integration: prohibited until author review.",
        f"- Author-decision SHA-256: `{author['author_decisions_sha256']}`.",
        "",
        "## Primary contract",
        "",
        "- Endpoint: recombinant soluble-expression status in *E. coli*; not direct aqueous solubility.",
        "- Reference: B1_ORDER_LOGIT (25 transparent AAC/physicochemical variables + 400 ordered dimers).",
        "- Candidate: B1_ORDER_PLUS_PROTT5_PLS_LOGIT (fold-local supervised ProtT5 PLS appended to B1).",
        f"- Learner: nested L2 logistic regression, lbfgs, max_iter={int(config.get('execution', {}).get('logistic_max_iter', 5000))}, class_weight=None.",
        "- C grid: `[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]`.",
        "- PLS component grid: `[1, 2, 3, 5, 8, 16]`.",
        "- Outer split: fixed existing seven-fold `development_fold` assignment; grouped and retrospective.",
        f"- Inner split: `{config.get('execution', {}).get('split_contract_version', 'seeded row-order-invariant grouped allocator')}`; five folds with fail-closed balance gates.",
        "- Primary uncertainty: paired cluster bootstrap, 10,000 replicates, seed 20260809.",
        "- Primary margin: AUC gain 0.02; log10 RMSE gain 0.10; Spearman gain 0.05.",
        "- External procedure: fit candidates on `fit`, select on `model_selection`, refit on their union, fit Platt mapping on `calibration`, seal, then apply once retrospectively.",
        "",
        "## Prohibited changes",
        "",
        "- No manuscript, caption, cover letter, existing audit JSON, or existing figure modification.",
        "- No ESM-2 completion, amyloid, learning curve, LoRA, attention pooling, synthesis performance, or app work before the author checkpoint.",
        "- No external-label-driven tuning, threshold selection, feature selection, or model replacement.",
        "- No row-level data, sequences, or record identifiers in tracked outputs.",
        "",
        "## Provenance",
        "",
        f"- Config SHA-256: `{sha256_file(config_path)}`.",
        f"- Code/config source-tree SHA-256: `{code_hash}`.",
        f"- Author decision SHA-256: `{sha256_file(author_path)}`.",
        "- Input canary hashes:",
    ]
    lines.extend([f"  - `{key}`: `{value}`" for key, value in sorted(input_hashes.items())])
    lines.extend(["", "Any contract change requires an append-only amendment and a new run_id; this run must not be overwritten.", ""])
    if freeze_path.exists():
        existing = freeze_path.read_text(encoding="utf-8")
        if f"Code/config source-tree SHA-256: `{code_hash}`." not in existing:
            raise FileExistsError(f"Existing freeze has a different code hash: {freeze_path}")
    else:
        atomic_write_text("\n".join(lines), freeze_path)
    return freeze_path, code_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--author-decisions", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    author_path = args.author_decisions.resolve()
    config = load_config(config_path)
    if str(args.run_dir.resolve()) != str(Path(config["paths"]["run_root"]).resolve()):
        raise ValueError("--run-dir does not match the frozen run_root.")
    if not author_path.is_file():
        raise FileNotFoundError(author_path)
    author = validate_author_decisions(author_path, config)
    project_root = Path(config["paths"]["project_root"]).resolve()
    observed_hashes = {}
    for key, relative in EXPECTED_HASH_FILES.items():
        path = project_root / relative
        observed_hashes[key] = sha256_file(path)
        if observed_hashes[key] != config["input_hashes"][key]:
            raise ValueError(f"Input canary mismatch for {key}: {observed_hashes[key]}")
    run_dir = Path(config["paths"]["run_root"]).resolve()
    manifest_dir = run_dir / "manifests"
    manifest_path = manifest_dir / "preflight_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing preflight manifest: {manifest_path}")
    freeze_path, code_hash = write_freeze(config, author, config_path, author_path)
    payload = {
        "status": "phase_0_preflight_passed",
        "run_id": config["run_id"],
        "analysis_scope": config["analysis_scope"],
        "config_sha256": sha256_file(config_path),
        "author_decisions": author,
        "input_hashes": observed_hashes,
        "code_hash": code_hash,
        "freeze_path": str(freeze_path),
        "freeze_sha256": sha256_file(freeze_path),
        "environment": environment_summary(),
        "manuscript_integration_authorized": False,
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
    }
    atomic_write_json(payload, manifest_path)
    atomic_write_json(payload["environment"], run_dir / "environment" / "sc8_environment.json")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
