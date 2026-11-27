#!/usr/bin/env python3
"""Derive privacy-safe S6 diagnostics from the package's frozen S5 aggregates.

This script performs no model fitting, embedding extraction, resampling,
permutation, threshold selection, or access to row-level predictions.  It checks
the exact released JSON hashes, computes descriptive differences and stability
summaries, and generates the Supplementary Table S3 TeX source from JSON.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import io
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "analysis_addon" / "results"

SOURCES = {
    "auc": (
        SOURCE_ROOT / "aggregate_internal_auc_s1.json",
        "ac69b08ce85e2cb8d02ef7633a6e98e59dd1976377384269d5fa78c78c982ee6",
    ),
    "diagnostics": (
        SOURCE_ROOT / "aggregate_internal_diagnostics_s1.json",
        "6c6f79db0b7f1500926a6a4e4f19cf9fd9173f92215a47e64abd5219596f6378",
    ),
    "distillation": (
        SOURCE_ROOT / "distillation_feasibility_s2.json",
        "1b9816bbf34b029c693bdad5845574ff19783f07ed89dec32ab4654c92be7549",
    ),
}

REF = "B1_ORDER_LOGIT"
CAND = "B1_ORDER_PLUS_PROTT5_PLS_LOGIT"
B0 = "B0_TRANSPARENT_LOGIT"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sources() -> tuple[dict, dict, dict, dict]:
    loaded: dict[str, dict] = {}
    provenance: dict[str, dict[str, str]] = {}
    for label, (path, expected) in SOURCES.items():
        observed = sha256(path)
        if observed != expected:
            raise ValueError(
                f"Canonical source hash mismatch for {path.name}: {observed} != {expected}"
            )
        loaded[label] = json.loads(path.read_text(encoding="utf-8"))
        provenance[label] = {
            "basename": path.name,
            "sha256": observed,
            "relative_path_from_package": path.relative_to(PACKAGE_ROOT).as_posix(),
        }

    auc = loaded["auc"]
    diagnostics = loaded["diagnostics"]
    distillation = loaded["distillation"]
    if auc.get("schema") != "s1_privacy_safe_internal_auc_v1":
        raise ValueError("Unexpected AUC schema")
    if diagnostics.get("schema") != "s1_privacy_safe_internal_diagnostics_v1":
        raise ValueError("Unexpected diagnostics schema")
    if distillation.get("schema") != "privacy-safe-fixed-distillation-feasibility-v1":
        raise ValueError("Unexpected distillation schema")
    for payload in (auc, diagnostics):
        for key in ("contains_record_ids", "contains_row_level_data", "contains_sequences"):
            if payload.get(key) is not False:
                raise ValueError(f"Privacy gate failed: {key}")
    if "not direct aqueous solubility" not in auc.get("endpoint", ""):
        raise ValueError("Endpoint boundary is missing")
    if distillation.get("status") != "scientific_null_engineering_pass_not_promoted":
        raise ValueError("Distillation status is not the frozen non-promotion state")
    return auc, diagnostics, distillation, provenance


DECIMAL_QUANTUM = Decimal("0.000000000000001")


def stable_mean(values: list[float]) -> float:
    """Return a cross-Python-version stable descriptive mean."""
    decimals = [Decimal(str(value)) for value in values]
    with localcontext() as context:
        context.prec = 50
        value = sum(decimals, Decimal(0)) / Decimal(len(decimals))
        return float(value.quantize(DECIMAL_QUANTUM, rounding=ROUND_HALF_EVEN))


def sample_sd(values: list[float]) -> float:
    """Return a fixed-precision sample SD without statistics-module drift."""
    if len(values) <= 1:
        return 0.0
    decimals = [Decimal(str(value)) for value in values]
    with localcontext() as context:
        context.prec = 50
        mean = sum(decimals, Decimal(0)) / Decimal(len(decimals))
        variance = sum((value - mean) ** 2 for value in decimals) / Decimal(len(decimals) - 1)
        value = variance.sqrt()
        return float(value.quantize(DECIMAL_QUANTUM, rounding=ROUND_HALF_EVEN))


def build_analysis(auc: dict, diagnostics: dict, distillation: dict, provenance: dict) -> dict:
    ref_folds = auc["arms"][REF]["fold_metrics"]
    cand_folds = auc["arms"][CAND]["fold_metrics"]
    if [x["fold"] for x in ref_folds] != [x["fold"] for x in cand_folds]:
        raise ValueError("Reference/candidate fold alignment failed")
    fold_rows = [
        {
            "fold": int(reference["fold"]),
            "reference_auc": float(reference["roc_auc"]),
            "candidate_auc": float(candidate["roc_auc"]),
            "gain": float(candidate["roc_auc"] - reference["roc_auc"]),
        }
        for reference, candidate in zip(ref_folds, cand_folds, strict=True)
    ]
    split_rows = [
        {
            "seed": int(row["seed"]),
            "reference_auc": float(row["reference_auc"]),
            "candidate_auc": float(row["candidate_auc"]),
            "gain": float(row["gain"]),
        }
        for row in auc["stability"]
    ]
    if not all(row["gain"] > 0 for row in fold_rows + split_rows):
        raise ValueError("Frozen positive-direction canary failed")

    fold_gains = [row["gain"] for row in fold_rows]
    split_gains = [row["gain"] for row in split_rows]
    ref_diag = diagnostics["arms"][REF]
    cand_diag = diagnostics["arms"][CAND]
    metrics = [
        ("pr_auc", "PR-AUC", "higher", "descriptive improvement"),
        ("brier", "Brier score", "lower", "descriptive improvement"),
        ("log_loss", "Log loss", "lower", "descriptive improvement"),
        (
            "calibration_slope",
            "Calibration slope",
            "closer_to_1",
            "did not improve; candidate is farther from 1",
        ),
        (
            "calibration_intercept",
            "Calibration intercept",
            "closer_to_0",
            "descriptively closer to 0; no external calibration evidence",
        ),
    ]
    diagnostic_rows = []
    for key, label, preferred, interpretation in metrics:
        reference = float(ref_diag[key])
        candidate = float(cand_diag[key])
        diagnostic_rows.append(
            {
                "metric": key,
                "label": label,
                "reference": reference,
                "candidate": candidate,
                "candidate_minus_reference": candidate - reference,
                "preferred_direction": preferred,
                "interpretation": interpretation,
            }
        )

    contrast = auc["primary_contrast"]
    analysis = {
        "schema": "stam_s6_released_aggregate_diagnostic_analysis_v1",
        "status": "completed_no_refit_no_resampling",
        "analysis_date": "2026-08-16",
        "analysis_scope": "descriptive derivation from released privacy-safe aggregates",
        "endpoint": auc["endpoint"],
        "privacy": {
            "contains_record_ids": False,
            "contains_row_level_data": False,
            "contains_sequences": False,
        },
        "scientific_computation_counts": {
            "model_fits": 0,
            "embedding_extractions": 0,
            "bootstrap_replicates_added": 0,
            "permutations_added": 0,
            "external_evaluations": 0,
        },
        "source_files": provenance,
        "absolute_auc": {
            "B0": float(auc["arms"][B0]["roc_auc"]),
            "B1": float(auc["arms"][REF]["roc_auc"]),
            "B1_plus_ProtT5": float(auc["arms"][CAND]["roc_auc"]),
        },
        "primary_contrast": {
            "metric": contrast["metric"],
            "point_gain": float(contrast["point_gain"]),
            "ci95": [float(x) for x in contrast["ci95"]],
            "practical_margin": float(contrast["practical_margin"]),
            "n_replicates_valid_existing": int(contrast["n_replicates_valid"]),
            "interpretation": (
                "Conditional group-bootstrap interval for fixed OOF predictions; "
                "not algorithm-level uncertainty including refitting."
            ),
        },
        "fold_sensitivity": {
            "rows": fold_rows,
            "n_partitions": len(fold_rows),
            "n_positive": sum(row["gain"] > 0 for row in fold_rows),
            "gain_min": min(fold_gains),
            "gain_max": max(fold_gains),
            "gain_mean_descriptive": stable_mean(fold_gains),
            "gain_sample_sd_descriptive": sample_sd(fold_gains),
            "interpretation": "evaluation partitions, not independent biological replicates",
        },
        "split_assignment_sensitivity": {
            "rows": split_rows,
            "n_assignments": len(split_rows),
            "n_positive": sum(row["gain"] > 0 for row in split_rows),
            "gain_min": min(split_gains),
            "gain_max": max(split_gains),
            "gain_mean_descriptive": stable_mean(split_gains),
            "gain_sample_sd_descriptive": sample_sd(split_gains),
            "interpretation": "frozen split sensitivity, not independent replication",
        },
        "secondary_diagnostics": diagnostic_rows,
        "compute_context": {
            "ProtT5_embedding_wall_hours": float(
                diagnostics["compute"]["prott5"]["embedding_wall_seconds"] / 3600.0
            ),
            "ProtT5_peak_rss_GiB": float(
                diagnostics["compute"]["prott5"]["peak_rss_bytes"] / (1024.0**3)
            ),
            "hardware": diagnostics["compute"]["prott5"]["hardware"],
            "model": diagnostics["compute"]["prott5"]["model"],
        },
        "negative_control": {
            "n_replicates_existing": int(auc["negative_control"]["n_replicates"]),
            "status": auc["negative_control"]["status"],
            "interpretation": "gross leakage/alignment gate; not a p-value",
        },
        "distillation_boundary": {
            "status": distillation["status"],
            "promoted": False,
            "point_gain_distilled_minus_hard": float(
                distillation["primary_auc_contrast"]["point_gain"]
            ),
            "ci95": [float(x) for x in distillation["primary_auc_contrast"]["ci95"]],
        },
        "claim_ceiling": (
            "Within-development recombinant soluble-expression discrimination in E. coli; "
            "not direct aqueous solubility, external validation, calibrated deployment utility, "
            "or state-of-the-art ranking."
        ),
    }
    return analysis


def render_diagnostic_tsv(analysis: dict) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "metric",
            "label",
            "reference",
            "candidate",
            "candidate_minus_reference",
            "preferred_direction",
            "interpretation",
        ],
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(analysis["secondary_diagnostics"])
    return handle.getvalue()


def render_table_s3(analysis: dict) -> str:
    rows = analysis["secondary_diagnostics"]
    source = analysis["source_files"]["diagnostics"]
    lines = [
        "% Generated; do not edit numeric literals by hand.",
        f"% Source: {source['basename']} SHA-256 {source['sha256']}",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Secondary within-development diagnostics for the order-aware transparent comparator (B1) and B1 plus fold-local ProtT5. The diagnostics are descriptive, are not a second headline and do not provide external calibration evidence.}",
        "\\label{tab:s3-diagnostics}",
        "\\fontsize{9.0}{11.0}\\selectfont",
        "\\renewcommand{\\arraystretch}{1.20}",
        "\\begin{tabularx}{0.86\\textwidth}{>{\\raggedright\\arraybackslash}X >{\\centering\\arraybackslash}p{0.22\\textwidth} >{\\centering\\arraybackslash}p{0.25\\textwidth}}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{B1 comparator} & \\textbf{B1 + fold-local ProtT5} \\\\",
        "\\midrule",
    ]
    def cell(value: float) -> str:
        """Render a value, setting a negative with a math minus rather than an ASCII hyphen."""
        text = f"{value:.7f}"
        return f"${text}$" if text.startswith("-") else text

    for row in rows:
        lines.append(
            f"{row['label']} & {cell(row['reference'])} & {cell(row['candidate'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabularx}",
            "",
            "\\vspace{4pt}",
            "\\begin{minipage}{0.86\\textwidth}",
            "\\footnotesize",
            # Revision 1 added a retrospective external evaluation whose calibration slope and
            # intercept are reported as diagnostics, so the note's former closing sentence
            # ("No external calibration assessment was performed.") became false. It is scoped
            # here to this within-development comparison, matching Supplementary Method 12.
            "These within-development diagnostics were secondary. PR-AUC, Brier score and log loss changed in the favorable direction. Calibration slope did not improve; the intercept was descriptively closer to zero. \\rev{No external calibration assessment forms part of this within-development comparison; the external calibration slope and intercept are reported as diagnostics of the separate retrospective external evaluation in Supplementary Methods 18 and 19, and no external recalibration was fitted anywhere in this article.}",
            "\\end{minipage}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def assert_replaceable(path: Path, kind: str) -> None:
    """Refuse to overwrite anything that is not a prior output of this generator."""
    if not path.exists():
        return
    if kind == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        valid = payload.get("schema") == "stam_s6_released_aggregate_diagnostic_analysis_v1"
    elif kind == "tsv":
        valid = path.read_text(encoding="utf-8").startswith(
            "metric\tlabel\treference\tcandidate\tcandidate_minus_reference\t"
        )
    elif kind == "tex":
        valid = path.read_text(encoding="utf-8").startswith(
            "% Generated; do not edit numeric literals by hand."
        )
    else:
        raise ValueError(f"Unknown generated-output kind: {kind}")
    if not valid:
        raise FileExistsError(f"Refusing to replace non-generated file: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", type=Path, default=PACKAGE_ROOT / "analysis_addon" / "results"
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=PACKAGE_ROOT / "tables" / "supp_table_s3_diagnostics.tex",
    )
    args = parser.parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    table_out = args.table_out.resolve()
    table_out.parent.mkdir(parents=True, exist_ok=True)
    targets = {
        "json": outdir / "stam_s6_released_aggregate_diagnostic_analysis_20260816.json",
        "tsv": outdir / "stam_s6_secondary_diagnostics_20260816.tsv",
        "tex": table_out,
    }
    for kind, path in targets.items():
        assert_replaceable(path, kind)

    auc, diagnostics, distillation, provenance = read_sources()
    analysis = build_analysis(auc, diagnostics, distillation, provenance)
    targets["json"].write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    targets["tsv"].write_text(
        render_diagnostic_tsv(analysis), encoding="utf-8", newline="\n"
    )
    targets["tex"].write_text(
        render_table_s3(analysis), encoding="utf-8", newline="\n"
    )
    for label, path in targets.items():
        print(f"WROTE {label}: {path} SHA-256 {sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
