"""Audit chemical information loss and family feasibility for modified peptides."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from openpyxl import load_workbook
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

try:
    from .io_contracts import atomic_write_json, sha256_file
    from .nested_models import PlattCalibrator
    from .split_contracts import balanced_group_folds
    from .stage_utils import refuse_existing, write_stage
except ImportError:  # pragma: no cover - direct execution path
    from io_contracts import atomic_write_json, sha256_file
    from nested_models import PlattCalibrator
    from split_contracts import balanced_group_folds
    from stage_utils import refuse_existing, write_stage


AA = "ACDEFGHIKLMNPQRSTVWY"
THREE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
PARENT = {"DALA": "A", "DLEU": "L", "DGLU": "E", "DTHR": "T", "DTYR": "Y", "DGLN": "Q", "DPHE": "F", "DSER": "S", "DTRP": "W", "DPRO": "P", "DLYS": "K", "DASN": "N", "DARG": "R", "NMELLEU": "L", "NMELTHR": "T", "NMEALA": "A", "NMETHYLLALA": "A", "NMELGLN": "Q", "NMELGLU": "E", "AMEPHE": "F", "AMELEU": "L", "AMETRP": "W", "AMELGLU": "E", "ALPHAMETHYLGLU": "E", "6FDTRP": "W", "6CLTRP": "W", "6FTRP": "W", "6CLW": "W", "4CF3DPHE": "F", "PHE4SO3H": "F", "PHE3CF3": "F", "PHE342F": "F"}
DROP = {"CBA", "N3", "BETAALA", "BA", "AHX", "AIB", "SAR", "TBA", "NAL", "DCBA", "CIT", "NLE", "HSER", "HARG", "HOMOSER", "CHA", "HEXA", "ME", "4H", "AC", "NH2", "OH", "COOH", "AMEN3OCT", "ALPHAMEN3OCT", "ALPHAMEHEPT6YNE", "N3OCT", "HEPT6YNE", "EARLYISOMER", "CPROPYLR8", "CPROPYLS5", "R8NOMETHYL", "S5NOMETHYL", "KTFR8", "ISOPROPYLSULFONYL", "3CHLOROPHENYL", "4CHLOROPHENYL", "45CHLOROPYRIDIN3YL", "3R5R6S", "1R4R", "DBU"}
STAPLE = re.compile(r"^[RSB][0-9]$")
LINKER = re.compile(r"STAPLE|LACTAM|RCM|OLEFIN|ALKYNE|COUPLING|BETWEEN|ISOMER|CPROPYL|NOMETHYL")
IUPAC = re.compile(r"ACID|AMINO-2-METHYL|YNOIC|PYRIDIN|SULFONYL|CHLOROPHENYL|METHYLOCT")
SECTIONS = {"ATSP": (4, 367), "Cterm": (371, 413), "Dpep": (417, 430)}


def classify_token(token: str) -> tuple[str, str | None]:
    token = re.sub(r"[^A-Za-z0-9]", "", token).upper()
    if not token:
        return "drop", None
    if len(token) == 1 and token in AA:
        return "keep", token
    if token in THREE:
        return "keep", THREE[token]
    if token in PARENT:
        return "keep", PARENT[token]
    if STAPLE.match(token) or LINKER.search(token) or token in DROP:
        return "drop", None
    return "unknown", token


def project_name(name: str) -> tuple[str | None, str]:
    if ";" in name:
        return None, "semicolon"
    if IUPAC.search(name.upper()):
        return None, "iupac"
    parts: list[tuple[str, str]] = []
    cursor = 0
    for match in re.finditer(r"\(([^)]*)\)", name):
        parts.append(("free", name[cursor:match.start()]))
        parts.append(("paren", match.group(1)))
        cursor = match.end()
    parts.append(("free", name[cursor:]))
    sequence: list[str] = []
    for kind, text in parts:
        if kind == "paren":
            token_kind, value = classify_token(text)
            if token_kind == "unknown":
                return None, "unknown"
            if value:
                sequence.append(value)
            continue
        for chunk in re.split(r"[\s\-]+", text):
            cleaned = re.sub(r"[^A-Za-z0-9]", "", chunk)
            if not cleaned:
                continue
            upper = cleaned.upper()
            if upper in THREE:
                sequence.append(THREE[upper])
            elif len(upper) == 1 and upper in AA:
                sequence.append(upper)
            elif re.fullmatch(r"[A-Za-z]+", cleaned) and all(char.upper() in AA for char in cleaned):
                sequence.extend(char.upper() for char in cleaned)
            else:
                token_kind, value = classify_token(cleaned)
                if token_kind == "unknown":
                    return None, "unknown"
                if value:
                    sequence.append(value)
    projected = "".join(sequence)
    if len(projected) < 2:
        return None, "too_short"
    return projected, "ok"


def load_rows(workbook_path: Path, sheet_name: str, target_column: int) -> tuple[list[dict], dict[str, int]]:
    workbook = load_workbook(workbook_path, read_only=True, data_only=True, keep_links=False)
    sheet = workbook[sheet_name]
    rows: list[dict] = []
    excluded = Counter()
    try:
        for section, (start, end) in SECTIONS.items():
            for row in range(start, end + 1):
                identifier = sheet.cell(row, 1).value
                name = sheet.cell(row, 2).value
                if identifier in (None, "") and name in (None, ""):
                    continue
                if name in (None, ""):
                    excluded["missing_name"] += 1
                    continue
                projected, reason = project_name(str(name))
                target = sheet.cell(row, target_column).value
                if projected is None:
                    excluded[reason] += 1
                    continue
                if not isinstance(target, (int, float)) or not math.isfinite(float(target)) or float(target) <= 0:
                    excluded["missing_or_nonpositive_target"] += 1
                    continue
                descriptor_values = []
                for column in (4, 5, 6):
                    value = sheet.cell(row, column).value
                    descriptor_values.append(float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else np.nan)
                rows.append({"section": section, "sequence": projected, "target_log10": math.log10(float(target)), "descriptor": descriptor_values})
    finally:
        workbook.close()
    return rows, dict(excluded)


def normalized_levenshtein(left: str, right: str) -> float:
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_left != char_right)))
        previous = current
    return float(previous[-1]) / max(len(left), len(right), 1)


def complete_linkage_groups(sequences: list[str], threshold: float) -> np.ndarray:
    unique = sorted(set(sequences))
    if len(unique) == 1:
        lookup = {unique[0]: 1}
        return np.asarray([1] * len(sequences), dtype=int)
    distances = np.zeros((len(unique), len(unique)), dtype=float)
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            distance = normalized_levenshtein(unique[i], unique[j])
            distances[i, j] = distance
            distances[j, i] = distance
    tree = linkage(squareform(distances, checks=False), method="complete")
    labels = fcluster(tree, t=float(threshold), criterion="distance")
    lookup = {sequence: int(label) for sequence, label in zip(unique, labels)}
    return np.asarray([lookup[sequence] for sequence in sequences], dtype=int)


def family_feasibility(rows: list[dict], family_ids: np.ndarray, config: dict) -> dict[str, object]:
    counts = np.bincount(family_ids)
    counts = counts[counts > 0]
    n_rows = len(rows)
    n_families = int(len(counts))
    largest = int(counts.max()) if len(counts) else 0
    y = np.asarray([row["target_log10"] for row in rows])
    # Feasibility is label-blind in construction; labels are used only for the
    # declared gate check and not to form or tune the families.
    pseudo_label = np.asarray([int(row["section"] == "ATSP") for row in rows])
    try:
        fold = balanced_group_folds(pseudo_label, family_ids.astype(str), 5, 20260809)
        test_sizes = [int(np.sum(fold == value)) for value in range(5)]
        fold_ok = min(test_sizes) >= int(config["modified"]["feasibility_min_test_rows_per_fold"])
    except ValueError:
        test_sizes = []
        fold_ok = False
    gate = {
        "n_independent_families": n_families,
        "largest_family_rows": largest,
        "largest_family_fraction": float(largest / n_rows) if n_rows else None,
        "test_rows_per_fold": test_sizes,
        "at_least_10_families": n_families >= int(config["modified"]["feasibility_min_families"]),
        "largest_family_at_most_20_percent": bool(n_rows and largest / n_rows <= float(config["modified"]["feasibility_max_family_fraction"])),
        "every_fold_at_least_10_test_rows": fold_ok,
    }
    gate["gate_passes"] = bool(
        gate["at_least_10_families"]
        and gate["largest_family_at_most_20_percent"]
        and gate["every_fold_at_least_10_test_rows"]
    )
    return gate


def collision_audit(rows: list[dict]) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence"]].append(float(row["target_log10"]))
    errors = []
    collision_sizes = []
    for values in grouped.values():
        mean = float(np.mean(values))
        errors.extend((value - mean) ** 2 for value in values)
        if len(values) > 1:
            collision_sizes.append(len(values))
    return {
        "n_rows": int(len(rows)),
        "n_unique_projected_sequences": int(len(grouped)),
        "n_collision_groups": int(len(collision_sizes)),
        "collision_rows": int(sum(collision_sizes)),
        "collision_row_fraction": float(sum(collision_sizes) / len(rows)) if rows else None,
        "collision_group_size_distribution": {str(key): int(value) for key, value in Counter(collision_sizes).items()},
        "observed_sample_projection_collision_rmse_log10": float(np.sqrt(np.mean(errors))) if errors else 0.0,
        "observed_sample_projection_collision_concentration_ratio": float(10 ** np.sqrt(np.mean(errors))) if errors else 1.0,
        "estimand_note": "observed-sample projection-collision floor for deterministic predictors using the canonical projection; not a population bound",
    }


def optional_family_model(rows: list[dict], family_ids: np.ndarray, prott5_path: Path) -> dict[str, object] | None:
    if not prott5_path.is_file():
        return None
    embedding = np.load(prott5_path)
    if embedding.shape[0] != len(rows):
        return None
    y = np.asarray([row["target_log10"] for row in rows], dtype=float)
    comp = np.asarray([[sequence.count(letter) / len(sequence) for letter in AA] for sequence in [row["sequence"] for row in rows]], dtype=float)
    phys = np.asarray([row["descriptor"] for row in rows], dtype=float)
    phys = np.where(np.isfinite(phys), phys, np.nanmedian(phys, axis=0))
    reference = np.hstack([comp, phys])
    candidate_scores = {"reference": np.full(len(rows), np.nan), "candidate": np.full(len(rows), np.nan)}
    fold = balanced_group_folds(np.asarray([int(row["section"] == "ATSP") for row in rows]), family_ids.astype(str), 5, 20260809)
    for fold_id in range(5):
        train = np.flatnonzero(fold != fold_id)
        test = np.flatnonzero(fold == fold_id)
        scaler = StandardScaler().fit(reference[train])
        ref_model = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(scaler.transform(reference[train]), y[train])
        candidate_scaler = StandardScaler().fit(embedding[train])
        pls = PLSRegression(n_components=3, max_iter=500).fit(candidate_scaler.transform(embedding[train]), y[train])
        candidate_matrix_train = np.hstack([reference[train], pls.transform(candidate_scaler.transform(embedding[train]))])
        candidate_matrix_test = np.hstack([reference[test], pls.transform(candidate_scaler.transform(embedding[test]))])
        cand_scaler = StandardScaler().fit(candidate_matrix_train)
        cand_model = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(cand_scaler.transform(candidate_matrix_train), y[train])
        candidate_scores["reference"][test] = ref_model.predict(scaler.transform(reference[test]))
        candidate_scores["candidate"][test] = cand_model.predict(cand_scaler.transform(candidate_matrix_test))
    return {"status": "exploratory_family_feasibility_passed_model_comparison", "reference_rmse": float(np.sqrt(np.mean((candidate_scores["reference"] - y) ** 2))), "candidate_rmse": float(np.sqrt(np.mean((candidate_scores["candidate"] - y) ** 2))), "log10_rmse_gain_candidate_minus_reference": float(np.sqrt(np.mean((candidate_scores["reference"] - y) ** 2)) - np.sqrt(np.mean((candidate_scores["candidate"] - y) ** 2))), "model_note": "chemistry-aware composition plus three public descriptor columns; ProtT5 candidate uses canonical projection and fixed three-coordinate sensitivity representation"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["collision-and-family-feasibility"], required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    if not (run_dir / "manifests/external-retrospective.json").is_file():
        raise FileNotFoundError("External retrospective stage is required before modified-peptide audit.")
    workbook = Path(config["modified"]["workbook"]).resolve()
    rows, excluded = load_rows(workbook, config["modified"]["sheet"], int(config["modified"]["target_column_number"]))
    sequences = [row["sequence"] for row in rows]
    exact = collision_audit(rows)
    schemes = {"curated_public_sections": family_feasibility(rows, np.asarray([{"ATSP": 0, "Cterm": 1, "Dpep": 2}[row["section"]] for row in rows]), config)}
    for threshold in config["modified"]["complete_linkage_thresholds"]:
        labels = complete_linkage_groups(sequences, float(threshold))
        schemes[f"complete_linkage_{float(threshold):.2f}"] = family_feasibility(rows, labels, config)
    passed = [key for key, value in schemes.items() if value["gate_passes"]]
    model_comparison = None
    if passed:
        selected = passed[0]
        if selected.startswith("complete_linkage_"):
            threshold = float(selected.rsplit("_", 1)[-1])
            family_ids = complete_linkage_groups(sequences, threshold)
            model_comparison = optional_family_model(rows, family_ids, Path(config["paths"]["scope_root"]) / "analysis_plm_ablation/_local_prott5.npy")
    result = {
        "status": "completed" if not passed else "completed_with_exploratory_family_model",
        "analysis_scope": "retrospective_data_informed",
        "endpoint": "direct modified-peptide kinetic solubility in the public workbook; not recombinant-expression status",
        "input": {"workbook_basename": workbook.name, "workbook_sha256": sha256_file(workbook), "n_rows_after_label_blind_projection_and_numeric_target_filter": int(len(rows)), "excluded_reason_counts": excluded},
        "collision_audit": exact,
        "family_feasibility": schemes,
        "family_model_comparison": model_comparison,
        "status_interpretation": "NOT_ESTIMABLE_WITH_AVAILABLE_INDEPENDENT_FAMILIES" if not passed else "Family gate passed for an automated sensitivity scheme; comparison is exploratory and does not establish general peptide superiority.",
        "limitations": ["Projection discards documented chemical identity.", "Exact projected-sequence grouping is optimistic and not a family-held-out claim.", "The public source contains closely related analogue sections rather than broad independent chemical families."],
        "contains_row_level_data": False,
        "contains_sequences": False,
        "contains_record_ids": False,
    }
    path = run_dir / "raw_results/modified_peptide_audit.json"
    refuse_existing(path)
    atomic_write_json(result, path)
    write_stage(run_dir, "modified-peptide-audit", {"result_path": str(path), "status_detail": result["status_interpretation"]}, requires=["external-retrospective"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
