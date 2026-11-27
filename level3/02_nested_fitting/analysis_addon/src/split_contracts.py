"""Deterministic, group-exclusive split contracts.

V4 deliberately uses scikit-learn's version-pinned StratifiedGroupKFold on a
canonical, row-order-independent sufficient-statistics representation.  Group
tie order is encoded in seeded SHA-256 tokens and ``shuffle`` remains disabled.
This matters in scikit-learn 1.2.0: its ``shuffle=True`` implementation shuffles
the group-count matrix without shuffling the corresponding group identifiers,
which can destroy stratification.  The superseded greedy allocator is also not
retained because it produced catastrophic 3-row folds on real stability seeds.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold


SPLIT_CONTRACT_VERSION = "v4-sgkf-1.2.0-seeded-token-shuffle-off-gates-v2"


class SplitNotEstimable(ValueError):
    """The requested grouped split is structurally impossible."""


class SplitBlocked(RuntimeError):
    """The frozen allocator did not produce a gate-passing feasible split."""


def _as_string_groups(groups: np.ndarray) -> np.ndarray:
    values = np.asarray(groups).astype(str)
    if np.any(values == "") or np.any(values == "nan"):
        raise ValueError("Group IDs must be non-empty and non-missing.")
    return values


def _stable_digest(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _seeded_group_token(group_digest: str, seed: int) -> str:
    """Encode a deterministic, seed-specific tie order without SGKF shuffle."""
    payload = f"{int(seed)}\x00{group_digest}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _group_mapping(groups: np.ndarray, folds: np.ndarray) -> dict[str, int]:
    groups = _as_string_groups(groups)
    folds = np.asarray(folds).astype(int)
    mapping: dict[str, int] = {}
    for group, fold in zip(groups, folds):
        prior = mapping.setdefault(str(group), int(fold))
        if prior != int(fold):
            raise ValueError("A group appears in multiple folds.")
    return mapping


def split_diagnostics(
    y: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    n_splits: int,
    *,
    row_tolerance: float = 0.05,
    positive_tolerance: float = 0.05,
    negative_tolerance: float = 0.05,
    prevalence_tolerance: float = 0.02,
    min_class_groups_per_fold: int = 1,
) -> dict[str, object]:
    """Return exact per-fold balance/support diagnostics and a hard-gate flag."""
    y = np.asarray(y).astype(int)
    groups = _as_string_groups(groups)
    folds = np.asarray(folds).astype(int)
    target_rows = float(len(y)) / int(n_splits)
    target_positive = float(y.sum()) / int(n_splits)
    target_negative = float((1 - y).sum()) / int(n_splits)
    overall_prevalence = float(y.mean())
    rows: list[dict[str, object]] = []
    for fold in range(int(n_splits)):
        mask = folds == fold
        n_rows = int(mask.sum())
        n_positive = int(y[mask].sum())
        n_negative = int(n_rows - n_positive)
        prevalence = float(n_positive / n_rows) if n_rows else float("nan")
        positive_groups = int(len(np.unique(groups[mask & (y == 1)])))
        negative_groups = int(len(np.unique(groups[mask & (y == 0)])))
        row = {
            "fold": int(fold),
            "n_rows": n_rows,
            "n_groups": int(len(np.unique(groups[mask]))),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "prevalence": prevalence,
            "positive_bearing_groups": positive_groups,
            "negative_bearing_groups": negative_groups,
            "relative_row_deviation": abs(n_rows - target_rows) / max(target_rows, 1.0),
            "relative_positive_deviation": abs(n_positive - target_positive) / max(target_positive, 1.0),
            "relative_negative_deviation": abs(n_negative - target_negative) / max(target_negative, 1.0),
            "absolute_prevalence_deviation": abs(prevalence - overall_prevalence),
        }
        row["hard_gate_passed"] = bool(
            row["relative_row_deviation"] <= row_tolerance
            and row["relative_positive_deviation"] <= positive_tolerance
            and row["relative_negative_deviation"] <= negative_tolerance
            and row["absolute_prevalence_deviation"] <= prevalence_tolerance
            and positive_groups >= int(min_class_groups_per_fold)
            and negative_groups >= int(min_class_groups_per_fold)
        )
        rows.append(row)
    return {
        "split_contract_version": SPLIT_CONTRACT_VERSION,
        "sklearn_version": sklearn.__version__,
        "n_rows": int(len(y)),
        "n_groups": int(len(np.unique(groups))),
        "n_splits": int(n_splits),
        "overall_prevalence": overall_prevalence,
        "gates": {
            "row_tolerance": float(row_tolerance),
            "positive_tolerance": float(positive_tolerance),
            "negative_tolerance": float(negative_tolerance),
            "prevalence_tolerance": float(prevalence_tolerance),
            "min_class_groups_per_fold": int(min_class_groups_per_fold),
        },
        "folds": rows,
        "all_hard_gates_passed": bool(all(bool(row["hard_gate_passed"]) for row in rows)),
    }


def validate_group_folds(y: np.ndarray, groups: np.ndarray, folds: np.ndarray, n_splits: int) -> None:
    y = np.asarray(y).astype(int)
    groups = _as_string_groups(groups)
    folds = np.asarray(folds).astype(int)
    if not (len(y) == len(groups) == len(folds)):
        raise ValueError("y, groups, and folds must have equal length.")
    if set(np.unique(folds)) != set(range(n_splits)):
        raise ValueError(f"Fold labels must be exactly 0..{n_splits - 1}.")
    for group in np.unique(groups):
        assigned = np.unique(folds[groups == group])
        if len(assigned) != 1:
            raise ValueError("A group appears in multiple folds.")
    for fold in range(n_splits):
        observed = set(np.unique(y[folds == fold]).tolist())
        if observed != {0, 1}:
            raise ValueError(f"Fold {fold} does not contain both classes: {observed}")


def fixed_group_folds(y: np.ndarray, groups: np.ndarray, supplied_folds: np.ndarray, n_splits: int) -> np.ndarray:
    supplied = np.asarray(supplied_folds)
    unique = sorted(np.unique(supplied).tolist())
    if len(unique) != n_splits:
        raise ValueError(f"Expected {n_splits} supplied folds, observed {unique}")
    mapping = {value: index for index, value in enumerate(unique)}
    normalized = np.asarray([mapping[value] for value in supplied], dtype=int)
    validate_group_folds(y, groups, normalized, n_splits)
    return normalized


def _canonical_relabel(groups: np.ndarray, folds: np.ndarray, n_splits: int) -> np.ndarray:
    """Remove arbitrary fold-label permutations using group-signature hashes."""
    mapping = _group_mapping(groups, folds)
    signatures = []
    for fold in range(int(n_splits)):
        members = sorted(_stable_digest(group) for group, value in mapping.items() if value == fold)
        signature = hashlib.sha256(json.dumps(members, separators=(",", ":")).encode("utf-8")).hexdigest()
        signatures.append((signature, fold))
    relabel = {old: new for new, (_, old) in enumerate(sorted(signatures))}
    return np.asarray([relabel[int(value)] for value in folds], dtype=int)


def balanced_group_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    *,
    row_tolerance: float = 0.05,
    positive_tolerance: float = 0.05,
    negative_tolerance: float = 0.05,
    prevalence_tolerance: float = 0.02,
    min_class_groups_per_fold: int = 1,
) -> np.ndarray:
    """Return canonical SGKF folds or fail closed.

    The sample-level table is reconstructed from per-group class counts in
    digest order.  Consequently, permuting input rows cannot alter the input
    presented to StratifiedGroupKFold.  Fold labels are canonicalized after
    allocation so label swaps do not change the partition hash.
    """
    y = np.asarray(y).astype(int)
    groups = _as_string_groups(groups)
    if len(y) != len(groups) or len(y) == 0:
        raise ValueError("y and groups must be non-empty and have equal length.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise SplitNotEstimable("Binary grouped stratification requires both classes.")
    unique_groups = sorted(np.unique(groups).tolist(), key=_stable_digest)
    if len(unique_groups) < int(n_splits):
        raise SplitNotEstimable("Not enough groups for the requested number of folds.")
    positive_group_count = len(np.unique(groups[y == 1]))
    negative_group_count = len(np.unique(groups[y == 0]))
    required = int(n_splits) * int(min_class_groups_per_fold)
    if positive_group_count < required or negative_group_count < required:
        raise SplitNotEstimable("Insufficient class-bearing groups for the requested support gate.")

    target_rows = float(len(y)) / int(n_splits)
    target_positive = float(y.sum()) / int(n_splits)
    target_negative = float((1 - y).sum()) / int(n_splits)
    canonical_y: list[int] = []
    canonical_groups: list[str] = []
    token_to_group: dict[str, str] = {}
    for group in unique_groups:
        digest = _stable_digest(group)
        token = _seeded_group_token(digest, int(seed))
        if token in token_to_group and token_to_group[token] != group:
            raise RuntimeError("Unexpected SHA-256 seeded group-token collision.")
        token_to_group[token] = group
        mask = groups == group
        negative = int(np.sum(y[mask] == 0))
        positive = int(np.sum(y[mask] == 1))
        if negative + positive > target_rows * (1.0 + float(row_tolerance)):
            raise SplitNotEstimable("An indivisible group exceeds the per-fold row hard bound.")
        if positive > target_positive * (1.0 + float(positive_tolerance)):
            raise SplitNotEstimable("An indivisible group exceeds the per-fold positive hard bound.")
        if negative > target_negative * (1.0 + float(negative_tolerance)):
            raise SplitNotEstimable("An indivisible group exceeds the per-fold negative hard bound.")
        canonical_y.extend([0] * negative + [1] * positive)
        canonical_groups.extend([token] * (negative + positive))

    canonical_y_array = np.asarray(canonical_y, dtype=int)
    canonical_group_array = np.asarray(canonical_groups, dtype=str)
    if sklearn.__version__ != "1.2.0":
        raise RuntimeError(
            "The frozen V4 split contract requires scikit-learn 1.2.0; "
            f"observed {sklearn.__version__}."
        )
    splitter = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=False)
    token_fold: dict[str, int] = {}
    for fold, (_, test) in enumerate(
        splitter.split(
            np.zeros((len(canonical_y_array), 1), dtype=np.int8),
            canonical_y_array,
            canonical_group_array,
        )
    ):
        for token in np.unique(canonical_group_array[test]):
            prior = token_fold.setdefault(str(token), int(fold))
            if prior != int(fold):
                raise RuntimeError("SGKF assigned one group to multiple folds.")
    if set(token_fold) != set(token_to_group):
        raise RuntimeError("SGKF did not assign every group exactly once.")
    group_fold = {token_to_group[token]: fold for token, fold in token_fold.items()}
    output = np.asarray([group_fold[str(group)] for group in groups], dtype=int)
    output = _canonical_relabel(groups, output, int(n_splits))
    validate_group_folds(y, groups, output, n_splits)
    diagnostics = split_diagnostics(
        y,
        groups,
        output,
        int(n_splits),
        row_tolerance=float(row_tolerance),
        positive_tolerance=float(positive_tolerance),
        negative_tolerance=float(negative_tolerance),
        prevalence_tolerance=float(prevalence_tolerance),
        min_class_groups_per_fold=int(min_class_groups_per_fold),
    )
    if not diagnostics["all_hard_gates_passed"]:
        raise SplitBlocked("Canonical SGKF assignment failed the frozen balance/support gates.")
    return output


def fold_hash(groups: np.ndarray, folds: np.ndarray) -> str:
    mapping = _group_mapping(groups, folds)
    pairs = sorted((_stable_digest(group), int(fold)) for group, fold in mapping.items())
    payload = {"split_contract_version": SPLIT_CONTRACT_VERSION, "mapping": pairs}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def assert_no_group_overlap(train_groups: np.ndarray, test_groups: np.ndarray) -> None:
    overlap = set(_as_string_groups(train_groups)) & set(_as_string_groups(test_groups))
    if overlap:
        raise ValueError(f"Group overlap detected: {len(overlap)} groups.")


def indices_for_fold(folds: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    folds = np.asarray(folds).astype(int)
    test = np.flatnonzero(folds == int(fold))
    train = np.flatnonzero(folds != int(fold))
    if len(test) == 0 or len(train) == 0:
        raise ValueError(f"Fold {fold} has an empty train or test partition.")
    return train, test
