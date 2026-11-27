import numpy as np
import pytest

from analysis_addon.src.split_contracts import (
    SPLIT_CONTRACT_VERSION,
    SplitNotEstimable,
    assert_no_group_overlap,
    balanced_group_folds,
    fixed_group_folds,
    fold_hash,
    split_diagnostics,
    validate_group_folds,
)


def synthetic_groups():
    # A non-degenerate, many-singleton fixture reflects the real SoluProt
    # geometry and permits the frozen 5% hard balance gate to be exercised.
    groups = np.asarray([f"g{i:03d}" for i in range(200)])
    y = np.tile([0, 1], 100)
    return y, groups


def partition_signature(groups, folds):
    """Compare partitions without depending on arbitrary fold labels."""
    return sorted(
        tuple(sorted(groups[folds == fold].tolist()))
        for fold in np.unique(folds)
    )


def test_group_folds_are_group_exclusive_and_row_order_invariant():
    y, groups = synthetic_groups()
    folds = balanced_group_folds(y, groups, 5, 17)
    validate_group_folds(y, groups, folds, 5)
    reference = {group: int(fold) for group, fold in zip(groups, folds)}
    for repeat in range(100):
        permutation = np.random.default_rng(1000 + repeat).permutation(len(y))
        shuffled = balanced_group_folds(y[permutation], groups[permutation], 5, 17)
        assert all(
            reference[group] == int(fold)
            for group, fold in zip(groups[permutation], shuffled)
        )
    different_seed = balanced_group_folds(y, groups, 5, 18)
    assert partition_signature(groups, folds) != partition_signature(groups, different_seed)


def test_mixed_small_groups_and_adversarial_row_order_pass_hard_gates():
    labels: list[int] = []
    group_ids: list[str] = []
    for index in range(800):
        labels.append(index % 2)
        group_ids.append(f"singleton_{index:04d}")
    patterns = ([0, 0], [1, 1], [0, 1], [0, 0, 1], [1, 1, 0])
    for index in range(100):
        pattern = patterns[index % len(patterns)]
        labels.extend(pattern)
        group_ids.extend([f"multi_{index:03d}"] * len(pattern))
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(group_ids)
    adversarial_order = np.lexsort((groups, y))
    y = y[adversarial_order]
    groups = groups[adversarial_order]
    folds = balanced_group_folds(
        y, groups, 5, 23, min_class_groups_per_fold=20
    )
    diagnostics = split_diagnostics(
        y, groups, folds, 5, min_class_groups_per_fold=20
    )
    assert diagnostics["all_hard_gates_passed"]
    validate_group_folds(y, groups, folds, 5)


def test_structurally_impossible_grouped_splits_fail_closed():
    giant_y = np.asarray([1] * 60 + [0] * 70 + [1] * 70)
    giant_groups = np.asarray(
        ["GIANT"] * 60
        + [f"negative_{index:03d}" for index in range(70)]
        + [f"positive_{index:03d}" for index in range(70)]
    )
    with pytest.raises(SplitNotEstimable, match="indivisible group"):
        balanced_group_folds(
            giant_y, giant_groups, 5, 17, min_class_groups_per_fold=1
        )

    support_y = np.asarray([1] * 10 + [0] * 200)
    support_groups = np.asarray(
        [f"positive_{index:03d}" for index in range(10)]
        + [f"negative_{index:03d}" for index in range(200)]
    )
    with pytest.raises(SplitNotEstimable, match="class-bearing groups"):
        balanced_group_folds(
            support_y, support_groups, 5, 17, min_class_groups_per_fold=20
        )


def test_split_contract_version_and_golden_hash_are_frozen():
    y, groups = synthetic_groups()
    folds = balanced_group_folds(y, groups, 5, 17)
    assert SPLIT_CONTRACT_VERSION == "v4-sgkf-1.2.0-seeded-token-shuffle-off-gates-v2"
    assert fold_hash(groups, folds) == "a0acae449abc682855fd4d73a5f6ba8ced5127247a9af42358a77bf011411ebc"


def test_fixed_fold_normalizes_one_based_assignments():
    y, groups = synthetic_groups()
    supplied = np.tile(np.arange(1, 6), 40)
    folds = fixed_group_folds(y, groups, supplied, 5)
    assert set(np.unique(folds)) == set(range(5))
    assert fold_hash(groups, folds) == fold_hash(groups, folds)
    assert_no_group_overlap(groups[folds != 0], groups[folds == 0])
    with pytest.raises(ValueError):
        assert_no_group_overlap(np.asarray(["a"]), np.asarray(["a"]))
