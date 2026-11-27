#!/usr/bin/env python3
"""Release only a privacy-safe aggregate for the frozen S1 internal AUC result.

The real-data CLI is deliberately strict: it accepts one canonical result,
three private OOF parquet files, frozen manifests/configuration, and writes one
aggregate JSON only after every alignment, hash, contract, and privacy check
passes.  The public JSON never contains row-level labels, scores, identifiers,
sequences, cluster identifiers, paths, or bootstrap arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


ENDPOINT = (
    "recombinant soluble-expression status in Escherichia coli; "
    "not direct aqueous solubility"
)
REQUIRED_OOF_COLUMNS = ("record_id", "cluster_id", "fold", "label", "score")
PUBLIC_FORBIDDEN_KEYS = {
    "record_id",
    "record_ids",
    "source_id",
    "source_ids",
    "cluster_id",
    "cluster_ids",
    "sequence",
    "sequences",
    "label",
    "labels",
    "score",
    "scores",
    "prediction",
    "predictions",
    "row_level_label",
    "row_level_labels",
    "row_level_score",
    "row_level_scores",
}
PUBLIC_PRIVATE_MARKERS = (
    "/users/",
    "/private/",
    "analysis_addon_run",
    "checkpoint",
    "raw_results",
    "private_tools",
)
ARM_SPECS = {
    "B0_TRANSPARENT_LOGIT": "Transparent baseline",
    "B1_ORDER_LOGIT": "Order-aware transparent baseline",
    "B1_ORDER_PLUS_PROTT5_PLS_LOGIT": "Order-aware baseline + fold-local ProtT5",
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _canonical_oof_hash(frame: pd.DataFrame) -> str:
    """Hash canonical row content after sorting, so joint row permutation is inert."""
    ordered = frame.loc[:, list(REQUIRED_OOF_COLUMNS)].copy()
    ordered["record_id"] = ordered["record_id"].astype(str)
    ordered["cluster_id"] = ordered["cluster_id"].astype(str)
    ordered["fold"] = ordered["fold"].astype(int)
    ordered["label"] = ordered["label"].astype(int)
    ordered["score"] = ordered["score"].astype(float)
    ordered = ordered.sort_values("record_id", kind="mergesort").reset_index(drop=True)
    serialized = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_oof(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    observed = tuple(str(column) for column in frame.columns)
    if set(observed) != set(REQUIRED_OOF_COLUMNS):
        raise ValueError(f"OOF columns differ from the frozen contract: {path.name}")
    frame = frame.loc[:, list(REQUIRED_OOF_COLUMNS)].copy()
    if frame.empty:
        raise ValueError(f"OOF file is empty: {path.name}")
    if frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise ValueError(f"OOF record alignment is not one-to-one: {path.name}")
    if frame["cluster_id"].isna().any() or frame["fold"].isna().any():
        raise ValueError(f"OOF grouping/fold values are incomplete: {path.name}")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or set(labels.astype(int).unique()) != {0, 1}:
        raise ValueError(f"OOF labels are not binary: {path.name}")
    if not np.allclose(labels.to_numpy(), labels.astype(int).to_numpy()):
        raise ValueError(f"OOF labels are non-integral: {path.name}")
    scores = pd.to_numeric(frame["score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy()).all():
        raise ValueError(f"OOF scores are non-finite: {path.name}")
    if np.any((scores.to_numpy() < 0.0) | (scores.to_numpy() > 1.0)):
        raise ValueError(f"OOF scores are outside [0, 1]: {path.name}")
    frame["label"] = labels.astype(int)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["score"] = scores.astype(float)
    return frame, _canonical_oof_hash(frame)


def _validate_hash_manifest(
    manifest_path: Path,
    paths_and_frames: dict[str, tuple[Path, pd.DataFrame, str]],
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Hash manifest must contain a files object.")
    for role, (path, frame, actual_hash) in paths_and_frames.items():
        entry = files.get(path.name)
        if not isinstance(entry, dict):
            raise ValueError(f"Hash manifest lacks {role}: {path.name}")
        if entry.get("canonical_sha256") != actual_hash:
            raise ValueError(f"OOF canonical hash mismatch: {path.name}")
        if entry.get("n_rows") != len(frame):
            raise ValueError(f"OOF row-count hash-manifest mismatch: {path.name}")
        if entry.get("columns") != list(REQUIRED_OOF_COLUMNS):
            raise ValueError(f"OOF column hash-manifest mismatch: {path.name}")
    return manifest


def _validate_alignment(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    for column in ("record_id", "cluster_id", "fold", "label"):
        left = reference[column].astype(str).to_numpy()
        right = other[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"OOF {label} is not aligned on {column}.")


def _validate_canonical(
    result: dict[str, Any],
    strict_contract: bool,
) -> tuple[str, dict[str, Any]]:
    if result.get("status") != "completed":
        raise ValueError("Canonical result is not completed.")
    if result.get("analysis_scope") != "retrospective_data_informed":
        raise ValueError("Canonical result has an unexpected analysis scope.")
    if ENDPOINT not in str(result.get("endpoint", "")):
        raise ValueError("Canonical endpoint boundary is missing.")
    split = result.get("split")
    if not isinstance(split, dict):
        raise ValueError("Canonical split block is missing.")
    if strict_contract:
        expected_split = {"n_rows": 11436, "n_groups": 10715, "outer_folds": 7}
        for key, expected in expected_split.items():
            if split.get(key) != expected:
                raise ValueError(f"Canonical split.{key} is not frozen at {expected}.")
    contrast = result.get("primary_contrast")
    if not isinstance(contrast, dict):
        raise ValueError("Canonical primary contrast is missing.")
    expected_contrast = {
        "reference": "B1_ORDER_LOGIT",
        "candidate": "B1_ORDER_PLUS_PROTT5_PLS_LOGIT",
        "metric": "roc_auc",
        "practical_margin": 0.02,
        "n_replicates_valid": 10000,
    }
    if strict_contract:
        for key, expected in expected_contrast.items():
            if contrast.get(key) != expected:
                raise ValueError(f"Canonical primary_contrast.{key} is not frozen.")
    stability = result.get("fold_stability")
    if not isinstance(stability, list) or len(stability) != 5:
        raise ValueError("Canonical five-seed stability block is missing.")
    gains = [float(row["auc_gain_candidate_minus_reference"]) for row in stability]
    ci95 = [float(value) for value in contrast.get("ci95", [])]
    margin = float(contrast.get("practical_margin", 0.02))
    if len(ci95) != 2:
        raise ValueError("Canonical 95% CI is malformed.")
    c1 = ci95[0] > 0.0
    c2 = c1 and ci95[0] > margin
    c3 = c1 and all(gain > 0.0 for gain in gains)
    if not c1:
        raise ValueError("C1 failed: the canonical 95% CI does not exclude zero.")
    claim_tier = "C3" if c3 else "C2" if c2 else "C1"
    negative = result.get("negative_controls", {}).get("training_label_permutation", {})
    if negative.get("status") != "PASSED":
        raise ValueError("Training-label negative-control gate did not pass.")
    if strict_contract and negative.get("n_replicates") != 50:
        raise ValueError("Negative-control replicate count is not frozen at 50.")
    for key in ("contains_row_level_data", "contains_sequences", "contains_record_ids"):
        if result.get(key) is not False:
            raise ValueError(f"Canonical privacy flag is not false: {key}")
    arms = result.get("arms")
    if not isinstance(arms, dict) or any(name not in arms for name in ARM_SPECS):
        raise ValueError("Canonical primary arms are incomplete.")
    return claim_tier, {"split": split, "contrast": contrast, "stability": stability, "negative": negative, "arms": arms}


def _interpolated_roc(labels: np.ndarray, scores: np.ndarray) -> list[list[float]]:
    fpr, tpr, _ = roc_curve(labels, scores)
    unique_fpr, first = np.unique(fpr, return_index=True)
    max_tpr = np.maximum.reduceat(tpr, first)
    grid = np.linspace(0.0, 1.0, 256, dtype=float)
    interpolated = np.interp(grid, unique_fpr, max_tpr)
    interpolated[0] = 0.0
    interpolated[-1] = 1.0
    if len(interpolated) != 256 or not np.isfinite(interpolated).all():
        raise ValueError("Interpolated ROC output failed the 256-point contract.")
    return [grid.tolist(), interpolated.tolist()]


def _finite_public(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in PUBLIC_FORBIDDEN_KEYS:
                raise ValueError(f"Forbidden public key at {path}.{key}")
            _finite_public(item, f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) > 256:
            raise ValueError(f"Public array exceeds 256 elements at {path}")
        for index, item in enumerate(value):
            _finite_public(item, f"{path}[{index}]")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite public number at {path}")
    elif isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in PUBLIC_PRIVATE_MARKERS):
            raise ValueError(f"Private path/artifact marker in public value at {path}")


def validate_public_payload(payload: dict[str, Any]) -> None:
    for key in ("contains_row_level_data", "contains_sequences", "contains_record_ids"):
        if payload.get(key) is not False:
            raise ValueError(f"Public privacy flag is not false: {key}")
    _finite_public(payload)
    roc = payload.get("roc")
    if not isinstance(roc, dict) or len(roc) != len(ARM_SPECS) + 1:
        raise ValueError("Public ROC block must contain fpr_grid plus exactly three arms.")
    grid = roc.get("fpr_grid")
    if not isinstance(grid, list) or len(grid) != 256 or grid[0] != 0.0 or grid[-1] != 1.0:
        raise ValueError("Public FPR grid failed the 256-point endpoint contract.")
    for arm, arrays in roc.items():
        if arm == "fpr_grid":
            continue
        if not isinstance(arrays, list) or len(arrays) != 256:
            raise ValueError(f"Public TPR array failed the 256-point contract: {arm}")
        if arrays[0] != 0.0 or arrays[-1] != 1.0:
            raise ValueError(f"Public TPR endpoints failed: {arm}")


def _write_once(payload: dict[str, Any], output: Path) -> None:
    if output.exists():
        existing = load_json(output)
        if existing != payload:
            raise FileExistsError(f"Existing S1 aggregate differs: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def build_release(
    *,
    result_path: Path,
    transparent_oof: Path,
    reference_oof: Path,
    candidate_oof: Path,
    config_path: Path,
    canonical_manifest_path: Path,
    hash_manifest_path: Path,
    preflight_manifest_path: Path,
    scope_manifest_path: Path,
    output_path: Path,
    strict_contract: bool = True,
) -> dict[str, Any]:
    result = load_json(result_path)
    canonical_manifest = load_json(canonical_manifest_path)
    hash_manifest = load_json(hash_manifest_path)
    preflight = load_json(preflight_manifest_path)
    scope = load_json(scope_manifest_path)
    if canonical_manifest.get("status") != "completed":
        raise ValueError("Canonical manifest is not completed.")
    if canonical_manifest.get("result_path", "").split("/")[-1] != result_path.name:
        raise ValueError("Canonical result/manifest basename mismatch.")
    if preflight.get("config_sha256") != sha256_file(config_path):
        raise ValueError("Frozen config hash does not match preflight manifest.")
    if scope.get("s0_fallback_must_remain_immutable") is not True:
        raise ValueError("S1 scope manifest does not preserve the S0 fallback.")
    claim_tier, canonical = _validate_canonical(result, strict_contract)

    paths = {
        "transparent": transparent_oof,
        "reference": reference_oof,
        "candidate": candidate_oof,
    }
    loaded = {role: _load_oof(path) for role, path in paths.items()}
    reference_frame = loaded["reference"][0]
    for role in ("transparent", "candidate"):
        _validate_alignment(reference_frame, loaded[role][0], role)
    n_rows = len(reference_frame)
    n_groups = int(reference_frame["cluster_id"].nunique())
    if n_rows != int(canonical["split"]["n_rows"]):
        raise ValueError("OOF row count differs from canonical result.")
    if n_groups != int(canonical["split"]["n_groups"]):
        raise ValueError("OOF group count differs from canonical result.")
    _validate_hash_manifest(
        hash_manifest_path,
        {role: (paths[role], loaded[role][0], loaded[role][1]) for role in paths},
    )

    labels = reference_frame["label"].to_numpy(dtype=int)
    computed_auc: dict[str, float] = {}
    frame_by_arm = {
        "B0_TRANSPARENT_LOGIT": loaded["transparent"][0],
        "B1_ORDER_LOGIT": loaded["reference"][0],
        "B1_ORDER_PLUS_PROTT5_PLS_LOGIT": loaded["candidate"][0],
    }
    for arm, frame in frame_by_arm.items():
        score = frame["score"].to_numpy(dtype=float)
        value = float(roc_auc_score(labels, score))
        canonical_value = float(canonical["arms"][arm]["metrics"]["roc_auc"])
        if abs(value - canonical_value) > 1e-9:
            raise ValueError(f"OOF AUC differs from canonical result: {arm}")
        computed_auc[arm] = canonical_value

    fold_gains = [
        float(candidate["roc_auc"] - reference["roc_auc"])
        for reference, candidate in zip(
            canonical["arms"]["B1_ORDER_LOGIT"]["fold_metrics"],
            canonical["arms"]["B1_ORDER_PLUS_PROTT5_PLS_LOGIT"]["fold_metrics"],
        )
    ]
    if claim_tier == "C3" and not all(gain > 0.0 for gain in fold_gains):
        raise ValueError("C3 fold-direction gate failed.")

    contrast = canonical["contrast"]
    stability = [
        {
            "seed": int(row["seed"]),
            "reference_auc": float(row["reference_auc"]),
            "candidate_auc": float(row["candidate_auc"]),
            "gain": float(row["auc_gain_candidate_minus_reference"]),
        }
        for row in canonical["stability"]
    ]
    negative_source = canonical["negative"]
    negative = {
        "status": str(negative_source["status"]),
        "n_replicates": int(negative_source["n_replicates"]),
        "reference_auc_null": {
            key: negative_source["reference_auc_null"][key]
            for key in ("mean", "median", "range95_percentile", "sd")
        },
        "candidate_auc_null": {
            key: negative_source["candidate_auc_null"][key]
            for key in ("mean", "median", "range95_percentile", "sd")
        },
        "candidate_minus_reference_auc_gain_null": {
            key: negative_source["candidate_minus_reference_auc_gain_null"][key]
            for key in ("mean", "median", "range95_percentile", "sd")
        },
    }
    public_arms = {}
    public_roc = {"fpr_grid": None}
    for arm, name in ARM_SPECS.items():
        public_arms[arm] = {
            "name": name,
            "roc_auc": computed_auc[arm],
            "fold_metrics": [
                {"fold": int(row["fold"]), "roc_auc": float(row["roc_auc"])}
                for row in canonical["arms"][arm]["fold_metrics"]
            ],
        }
        public_roc[arm] = _interpolated_roc(
            labels,
            frame_by_arm[arm]["score"].to_numpy(dtype=float),
        )[1]
    public_roc["fpr_grid"] = _interpolated_roc(labels, reference_frame["score"].to_numpy(dtype=float))[0]

    input_artifacts = []
    for role in ("transparent", "reference", "candidate"):
        input_artifacts.append(
            {
                "role": role,
                "basename": paths[role].name,
                "canonical_sha256": loaded[role][1],
            }
        )
    provenance = {
        "input_artifacts": input_artifacts,
        "canonical_result": {"basename": result_path.name, "sha256": sha256_file(result_path)},
        "canonical_manifest": {
            "basename": canonical_manifest_path.name,
            "sha256": sha256_file(canonical_manifest_path),
        },
        "hash_manifest": {"basename": hash_manifest_path.name, "sha256": sha256_file(hash_manifest_path)},
        "preflight_manifest": {
            "basename": preflight_manifest_path.name,
            "sha256": sha256_file(preflight_manifest_path),
        },
        "scope_manifest": {"basename": scope_manifest_path.name, "sha256": sha256_file(scope_manifest_path)},
        "config": {"basename": config_path.name, "sha256": sha256_file(config_path)},
        "release_script": {"basename": Path(__file__).name, "sha256": sha256_file(Path(__file__))},
    }
    payload: dict[str, Any] = {
        "schema": "s1_privacy_safe_internal_auc_v1",
        "status": "completed",
        "claim_tier": claim_tier,
        "analysis_scope": str(result["analysis_scope"]),
        "endpoint": ENDPOINT,
        "split": {
            "n_rows": n_rows,
            "n_groups": n_groups,
            "group_definition": "existing 30%-identity groups",
            "outer_folds": int(canonical["split"]["outer_folds"]),
        },
        "arms": public_arms,
        "primary_contrast": {
            "reference": str(contrast["reference"]),
            "candidate": str(contrast["candidate"]),
            "metric": str(contrast["metric"]),
            "point_gain": float(contrast["point_gain"]),
            "ci90": [float(value) for value in contrast["ci90"]],
            "ci95": [float(value) for value in contrast["ci95"]],
            "practical_margin": float(contrast["practical_margin"]),
            "n_replicates_valid": int(contrast["n_replicates_valid"]),
            "decision_flags": dict(contrast["decision_flags"]),
        },
        "stability": stability,
        "negative_control": negative,
        "roc": public_roc,
        "provenance": provenance,
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
    }
    validate_public_payload(payload)
    _write_once(payload, output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--transparent-oof", type=Path, required=True)
    parser.add_argument("--reference-oof", type=Path, required=True)
    parser.add_argument("--candidate-oof", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--hash-manifest", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--scope-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_release(
        result_path=args.result_json.resolve(),
        transparent_oof=args.transparent_oof.resolve(),
        reference_oof=args.reference_oof.resolve(),
        candidate_oof=args.candidate_oof.resolve(),
        config_path=args.config.resolve(),
        canonical_manifest_path=args.canonical_manifest.resolve(),
        hash_manifest_path=args.hash_manifest.resolve(),
        preflight_manifest_path=args.preflight_manifest.resolve(),
        scope_manifest_path=args.scope_manifest.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps({"status": payload["status"], "claim_tier": payload["claim_tier"], "output": args.output.name}, indent=2))


if __name__ == "__main__":
    main()
