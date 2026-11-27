#!/usr/bin/env python3
"""Release frozen, aggregate-only S1 diagnostics without scientific recomputation.

The CLI accepts only a frozen V4 run directory and an S1 output path.  It reads
the canonical result, canonical/preflight manifests, and the frozen config hash;
it never opens OOF parquet files, checkpoints, prediction files, or model code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "s1_privacy_safe_internal_diagnostics_v1"
ENDPOINT = (
    "recombinant soluble-expression status in Escherichia coli; "
    "not direct aqueous solubility"
)
EXPECTED = {
    "result_sha256": "e5ee9773ee56ce0c1815fe2434468907c16138e09583624e981f6708ad1e0c6e",
    "canonical_manifest_sha256": "1a87741fdc7d49077d67c58327256ed169f611fb9a419491f229f1581c676b82",
    "preflight_manifest_sha256": "c031d443cc84b9d0100a695860dcd025696f0731527ff6aadfa26b19e7666f4a",
    "config_sha256": "f5474cb07f86cb6a7dc8d789a8a8cb2a062df740a2901314ddcb62a2423dfe60",
}
REQUIRED_PRIVACY_FLAGS = (
    "contains_row_level_data",
    "contains_sequences",
    "contains_record_ids",
)
FORBIDDEN_KEYS = {
    "record_id",
    "record_ids",
    "cluster_id",
    "cluster_ids",
    "sequence",
    "sequences",
    "label",
    "labels",
    "score",
    "scores",
    "probability",
    "probabilities",
    "prediction",
    "predictions",
    "fold_assignment",
    "fold_assignments",
    "checkpoint",
    "checkpoints",
    "path",
    "paths",
    "username",
}
PRIVATE_MARKERS = (
    "/users/",
    "/private/",
    "analysis_addon_run",
    "raw_results",
    "checkpoint",
    "private_tools",
)

# These are frozen protocol facts, not recomputed results.  They are included
# only because the release contract requires the already-documented compute
# disclosure in the aggregate-only diagnostic record.
FROZEN_COMPUTE = {
    "mmseqs2": {
        "version": "17-b804",
        "arguments": "--min-seq-id 0.30 -c 0.80 --cov-mode 0 --cluster-mode 1 -s 7.5",
    },
    "prott5": {
        "model": "Rostlab/prot_t5_xl_half_uniref50-enc",
        "revision": "94a6abc029ae13029317b140b7424e012bf8dfbf",
        "dimension": 1024,
        "pooling": "arithmetic mean over residue embeddings; special tokens excluded",
        "chunking": "non-overlapping 900-residue chunks",
        "affected_sequences": 379,
        "hardware": "CPU",
        "embedding_wall_seconds": 9241.855,
        "peak_rss_bytes": 7888699392,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def _resolve_inputs(run_dir: Path) -> dict[str, Path]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    result = run_dir / "raw_results" / "internal_primary.json"
    canonical = run_dir / "manifests" / "internal-primary.json"
    preflight = run_dir / "manifests" / "preflight_manifest.json"
    config = run_dir.parents[1] / "peptide_solubility_review" / "analysis_addon" / "config" / "analysis_config_v4.yaml"
    return {"result": result, "canonical": canonical, "preflight": preflight, "config": config}


def _verify_inputs(paths: dict[str, Path], strict_contract: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = load_json(paths["result"])
    canonical = load_json(paths["canonical"])
    preflight = load_json(paths["preflight"])
    if canonical.get("status") != "completed":
        raise ValueError("Canonical manifest is not completed.")
    if canonical.get("result_path", "").split("/")[-1] != paths["result"].name:
        raise ValueError("Canonical result basename mismatch.")
    config_hash = sha256_file(paths["config"])
    if preflight.get("config_sha256") != config_hash:
        raise ValueError("Config hash differs from preflight manifest.")
    if strict_contract:
        checks = {
            "result_sha256": sha256_file(paths["result"]),
            "canonical_manifest_sha256": sha256_file(paths["canonical"]),
            "preflight_manifest_sha256": sha256_file(paths["preflight"]),
            "config_sha256": config_hash,
        }
        for key, expected in EXPECTED.items():
            if checks[key] != expected:
                raise ValueError(f"Frozen input hash mismatch: {key}")
    return result, canonical, preflight


def _require_finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value: {label}")
    return number


def _selected_summary(selected: dict[str, Any], arm: str, strict_contract: bool) -> dict[str, Any]:
    rows = selected.get(arm)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Missing selected hyperparameters: {arm}")
    summary: dict[str, Any] = {"outer_folds": len(rows), "selected_c_counts": {}, "selected_k_counts": {}}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Malformed selected hyperparameter row: {arm}")
        c = _require_finite(row.get("selected_c"), f"{arm}.selected_c")
        c_key = str(int(c)) if c.is_integer() else str(c)
        summary["selected_c_counts"][c_key] = summary["selected_c_counts"].get(c_key, 0) + 1
        k = row.get("selected_pls_k")
        if k is not None:
            k_key = str(int(k))
            summary["selected_k_counts"][k_key] = summary["selected_k_counts"].get(k_key, 0) + 1
    if strict_contract:
        if summary["outer_folds"] != 7:
            raise ValueError(f"Unexpected outer-fold count: {arm}")
    return summary


def _convergence_summary(selected: Any) -> dict[str, int]:
    counts = {"flags_checked": 0, "failures": 0, "warnings": 0}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "converged" and isinstance(item, bool):
                    counts["flags_checked"] += 1
                    counts["failures"] += int(not item)
                elif key == "convergence_warning" and isinstance(item, bool):
                    counts["warnings"] += int(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(selected)
    if counts["flags_checked"] == 0:
        raise ValueError("No convergence flags found in canonical result.")
    return counts


def _metric_block(result: dict[str, Any], arm: str) -> dict[str, float]:
    source = result.get("arms", {}).get(arm, {})
    metrics = source.get("metrics", {})
    calibration = source.get("calibration", {})
    return {
        "pr_auc": _require_finite(metrics["auprc"], f"{arm}.auprc"),
        "brier": _require_finite(metrics["brier"], f"{arm}.brier"),
        "log_loss": _require_finite(metrics["log_loss"], f"{arm}.log_loss"),
        "calibration_intercept": _require_finite(calibration["intercept"], f"{arm}.calibration.intercept"),
        "calibration_slope": _require_finite(calibration["slope"], f"{arm}.calibration.slope"),
    }


def validate_public_payload(payload: dict[str, Any]) -> None:
    for key in REQUIRED_PRIVACY_FLAGS:
        if payload.get(key) is not False:
            raise ValueError(f"Privacy flag is not false: {key}")

    def visit(value: Any, location: str = "root") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = str(key).lower()
                if lowered in FORBIDDEN_KEYS:
                    raise ValueError(f"Forbidden public key: {location}.{key}")
                visit(item, f"{location}.{key}")
        elif isinstance(value, list):
            if len(value) > 256 or len(value) >= 1000:
                raise ValueError(f"Public array too large: {location}")
            for index, item in enumerate(value):
                visit(item, f"{location}[{index}]")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"Non-finite public number: {location}")
        elif isinstance(value, str):
            lowered = value.lower()
            if any(marker in lowered for marker in PRIVATE_MARKERS):
                raise ValueError(f"Private marker in public value: {location}")
            if value.startswith("/") or "\\" in value:
                raise ValueError(f"Path-like public value: {location}")

    visit(payload)
    if payload.get("schema") != SCHEMA or payload.get("status") != "completed":
        raise ValueError("Diagnostics schema/status mismatch.")


def _write_once(payload: dict[str, Any], output: Path) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"Existing diagnostics aggregate differs: {output.name}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)


def build_release(run_dir: Path, output: Path, strict_contract: bool = True) -> dict[str, Any]:
    paths = _resolve_inputs(run_dir)
    result, canonical, preflight = _verify_inputs(paths, strict_contract)
    if result.get("status") != "completed" or result.get("analysis_scope") != "retrospective_data_informed":
        raise ValueError("Canonical result status/scope is not frozen.")
    if result.get("endpoint") != ENDPOINT:
        raise ValueError("Canonical endpoint differs from the frozen boundary.")
    for flag in REQUIRED_PRIVACY_FLAGS:
        if result.get(flag) is not False:
            raise ValueError(f"Canonical privacy flag is not false: {flag}")
    split = result.get("split", {})
    if not isinstance(split, dict):
        raise ValueError("Canonical split block is missing.")
    n_rows = int(split["n_rows"])
    n_groups = int(split["n_groups"])
    outer_folds = int(split["outer_folds"])
    if strict_contract and (n_rows, n_groups, outer_folds) != (11436, 10715, 7):
        raise ValueError("Canonical split counts differ from frozen contract.")

    selected = result.get("selected_hyperparameters")
    if not isinstance(selected, dict):
        raise ValueError("Canonical selected hyperparameters are missing.")
    b1_summary = _selected_summary(selected, "B1_ORDER_LOGIT", strict_contract)
    candidate_summary = _selected_summary(selected, "B1_ORDER_PLUS_PROTT5_PLS_LOGIT", strict_contract)
    convergence = _convergence_summary(selected)
    if strict_contract and (convergence["flags_checked"], convergence["failures"]) != (12026, 0):
        raise ValueError("Canonical convergence aggregate differs from frozen audit.")

    negative_source = result.get("negative_controls", {}).get("training_label_permutation", {})
    if negative_source.get("status") != "PASSED":
        raise ValueError("Training-label negative-control gate did not pass.")
    negative = {
        "status": str(negative_source["status"]),
        "n_replicates": int(negative_source["n_replicates"]),
        "role": "gross leakage/alignment gate; not a p-value",
    }
    if strict_contract and negative["n_replicates"] != 50:
        raise ValueError("Negative-control replicate count differs from frozen contract.")

    metrics = {
        "B1_ORDER_LOGIT": _metric_block(result, "B1_ORDER_LOGIT"),
        "B1_ORDER_PLUS_PROTT5_PLS_LOGIT": _metric_block(result, "B1_ORDER_PLUS_PROTT5_PLS_LOGIT"),
    }
    if strict_contract:
        expected_metrics = {
            "B1_ORDER_LOGIT": {"pr_auc": 0.7423662792916608, "brier": 0.21303168877322712, "log_loss": 0.6115954943534385, "calibration_slope": 0.7126595977131277},
            "B1_ORDER_PLUS_PROTT5_PLS_LOGIT": {"pr_auc": 0.8013552163792494, "brier": 0.1904001256817954, "log_loss": 0.5635515651002654, "calibration_slope": 0.69547895458183},
        }
        for arm, values in expected_metrics.items():
            for key, expected in values.items():
                if abs(metrics[arm][key] - expected) > 1e-12:
                    raise ValueError(f"Frozen diagnostic mismatch: {arm}.{key}")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "completed",
        "analysis_scope": str(result["analysis_scope"]),
        "endpoint": ENDPOINT,
        "split": {
            "n_rows": n_rows,
            "n_groups": n_groups,
            "class_counts": {"negative": 5718, "positive": 5718},
            "outer_folds": outer_folds,
            "inner_folds": int(preflight.get("author_decisions", {}).get("primary_contract", {}).get("inner_folds", 5) or 5),
            "singleton_group_fraction": _require_finite(split["singleton_group_fraction"], "singleton_group_fraction"),
        },
        "arms": metrics,
        "selected_hyperparameters": {
            "B1_ORDER_LOGIT": b1_summary,
            "B1_ORDER_PLUS_PROTT5_PLS_LOGIT": candidate_summary,
        },
        "convergence": convergence,
        "compute": FROZEN_COMPUTE,
        "negative_control": negative,
        "provenance": {
            "canonical_result": {"basename": paths["result"].name, "sha256": sha256_file(paths["result"])},
            "canonical_manifest": {"basename": paths["canonical"].name, "sha256": sha256_file(paths["canonical"])},
            "preflight_manifest": {"basename": paths["preflight"].name, "sha256": sha256_file(paths["preflight"])},
            "config": {"basename": paths["config"].name, "sha256": sha256_file(paths["config"])},
            "release_script": {"basename": Path(__file__).name, "sha256": sha256_file(Path(__file__))},
        },
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
    }
    validate_public_payload(payload)
    _write_once(payload, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_release(args.run_dir, args.output, strict_contract=True)
    print(json.dumps({"status": payload["status"], "output": args.output.name}, indent=2))


if __name__ == "__main__":
    main()
