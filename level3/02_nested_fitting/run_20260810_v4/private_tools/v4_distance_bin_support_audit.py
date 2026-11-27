"""Audit class-bearing group support for the frozen external identity quartiles.

The frozen shift implementation gates distance-bin inference on row/group counts
but the external audit additionally requested at least 20 positive-bearing and
20 negative-bearing groups per bin.  This post-stage audit does not change the
bins, predictions, model selection, or frozen result.  It only records whether
the already-defined inference gate is scientifically usable.  External labels
are read only after the sealed external and shift stages exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[4]
PAPER = ROOT / "paper/peptide_solubility_review"
RUN_DIR = ROOT / "paper/general_peptide_solubility_scoping/analysis_addon_run_20260810_v4"
sys.path.insert(0, str(PAPER))

from analysis_addon.src.io_contracts import atomic_write_json, sha256_file  # noqa: E402
from analysis_addon.src.stage_utils import refuse_existing, require_stage, write_stage  # noqa: E402


MIN_CLASS_BEARING_GROUPS = 20


def assign_identity_quartiles(identity: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(identity), errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Identity values contain missing or non-finite entries.")
    ranked = values.rank(method="first")
    bins = pd.qcut(ranked, 4, labels=False)
    if bins.isna().any():
        raise ValueError("Identity quartile assignment produced missing bins.")
    return bins.to_numpy(dtype=int) + 1


def class_bearing_summary(labels: np.ndarray, groups: np.ndarray, bins: np.ndarray) -> list[dict[str, object]]:
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=str)
    bins = np.asarray(bins, dtype=int)
    if not (labels.shape == groups.shape == bins.shape):
        raise ValueError("Labels, groups, and bins must have identical shapes.")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("External labels must contain both binary classes.")
    output: list[dict[str, object]] = []
    for quartile in range(1, 5):
        mask = bins == quartile
        subset_groups = groups[mask]
        subset_labels = labels[mask]
        unique_groups = np.unique(subset_groups)
        positive_groups = sum(bool(np.any(subset_labels[subset_groups == group] == 1)) for group in unique_groups)
        negative_groups = sum(bool(np.any(subset_labels[subset_groups == group] == 0)) for group in unique_groups)
        output.append(
            {
                "identity_quartile": quartile,
                "n_rows": int(mask.sum()),
                "n_groups": int(len(unique_groups)),
                "positive_bearing_groups": int(positive_groups),
                "negative_bearing_groups": int(negative_groups),
                "class_bearing_support_gate": bool(
                    positive_groups >= MIN_CLASS_BEARING_GROUPS
                    and negative_groups >= MIN_CLASS_BEARING_GROUPS
                ),
            }
        )
    return output


def self_test() -> None:
    labels = np.asarray([0, 1] * 12, dtype=int)
    groups = np.asarray([f"g{i // 2}" for i in range(24)], dtype=str)
    identity = np.arange(24, dtype=float)
    bins = assign_identity_quartiles(identity)
    rows = class_bearing_summary(labels, groups, bins)
    assert [row["n_rows"] for row in rows] == [6, 6, 6, 6]
    assert all(row["class_bearing_support_gate"] is False for row in rows)
    labels = np.tile(np.asarray([0, 1], dtype=int), 80)
    groups = np.asarray([f"g{i // 2}" for i in range(160)], dtype=str)
    bins = assign_identity_quartiles(np.arange(160, dtype=float))
    rows = class_bearing_summary(labels, groups, bins)
    assert all(row["class_bearing_support_gate"] is True for row in rows)
    print("distance-bin support self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=False)
    parser.add_argument("--run-dir", type=Path, required=False)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.config is None or args.run_dir is None:
        parser.error("--config and --run-dir are required unless --self-test is used")
    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("run_id") != "addon_20260810_v4":
        raise ValueError("This audit is restricted to addon_20260810_v4.")
    if run_dir != Path(config["paths"]["run_root"]).resolve():
        raise ValueError("Run directory does not match the frozen config.")
    required = ["external-retrospective", "shift-audit"]
    missing = [stage for stage in required if not (run_dir / "manifests" / f"{stage}.json").is_file()]
    if args.preflight_only:
        print(json.dumps({"status": "preflight_only", "missing_required_stages": missing, "writes_performed": False}, indent=2))
        return
    if missing:
        raise FileNotFoundError(f"Required stages are missing: {missing}")
    external_path = Path(config["paths"]["data_root"]) / "soluprot_external_test_clustered.csv"
    similarity_path = Path(config["shift"]["similarity_audit"])
    external = pd.read_csv(external_path, dtype={"record_id": str, "external_cluster_id": str})
    similarity = pd.read_csv(similarity_path, dtype={"record_id": str})
    for frame, name in ((external, "external"), (similarity, "similarity")):
        if frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
            raise ValueError(f"{name} record IDs are not unique and complete.")
    if not np.array_equal(external["record_id"].to_numpy(), similarity["record_id"].to_numpy()):
        raise ValueError("External and similarity rows are not in identical record order.")
    raw_path = run_dir / "raw_results/shift_audit.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    identity = pd.to_numeric(similarity["global_identity_matches_over_max_length"], errors="coerce").to_numpy(float)
    bins = assign_identity_quartiles(identity)
    rows = class_bearing_summary(
        external["label"].to_numpy(dtype=int),
        external["external_cluster_id"].astype(str).to_numpy(),
        bins,
    )
    raw_bins = raw.get("external_shift", {}).get("bins", [])
    if len(raw_bins) != 4:
        raise ValueError("Frozen shift result does not contain four identity bins.")
    for row, raw_row in zip(rows, raw_bins, strict=True):
        if int(raw_row["identity_quartile"]) != row["identity_quartile"]:
            raise ValueError("Frozen result bin order does not match the audit.")
        row["raw_row_group_gate"] = bool(raw_row.get("inference_allowed", False))
        row["safe_inference_allowed"] = bool(
            row["raw_row_group_gate"] and row["class_bearing_support_gate"]
        )
        row["suppression_reason"] = (
            None
            if row["safe_inference_allowed"]
            else "class-bearing group support below 20/20 or frozen row/group gate failed"
        )
    payload = {
        "status": "completed",
        "analysis_scope": "retrospective_data_informed",
        "identity_definition": "global_identity_matches_over_max_length",
        "cutpoints_label_blind": True,
        "min_positive_bearing_groups": MIN_CLASS_BEARING_GROUPS,
        "min_negative_bearing_groups": MIN_CLASS_BEARING_GROUPS,
        "bins": rows,
        "all_bins_safe_for_inference": bool(all(row["safe_inference_allowed"] for row in rows)),
        "external_sha256": sha256_file(external_path),
        "similarity_audit_sha256": sha256_file(similarity_path),
        "shift_result_sha256": sha256_file(raw_path),
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
        "interpretation_boundary": "This audit can suppress sparse-bin inference; it cannot make a retrospective distance analysis causal or prospective.",
    }
    result_path = run_dir / "raw_results/distance_bin_support_audit.json"
    refuse_existing(result_path)
    atomic_write_json(payload, result_path)
    write_stage(
        run_dir,
        "distance-bin-support-audit",
        {"result_path": str(result_path), "status_detail": payload["status"]},
        requires=required,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
