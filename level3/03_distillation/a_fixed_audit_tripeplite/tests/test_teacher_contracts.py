from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PLAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLAN_ROOT / "src"))

from teacher_crossfit import (  # noqa: E402
    DevelopmentData,
    fixed_teacher_crossfit,
    fixed_teacher_evaluation,
    load_config,
    make_crossfit_plan,
    stable_hash,
)


def synthetic_fold_table():
    folds = np.repeat(np.arange(7), 12)
    groups = np.asarray([f"group_{index:03d}" for index in range(len(folds))])
    return folds, groups


def test_design_crossfit_excludes_gate_and_confirmation_folds() -> None:
    folds, groups = synthetic_fold_table()
    plan = make_crossfit_plan(folds, groups, [2, 3, 4, 5, 6])
    assert [item.heldout_fold for item in plan] == [2, 3, 4, 5, 6]
    for item in plan:
        assert set(folds[item.train_indices]) == set(item.train_folds)
        assert set(folds[item.target_indices]) == {item.heldout_fold}
        assert 0 not in set(folds[item.train_indices])
        assert 1 not in set(folds[item.train_indices])
        assert not (set(groups[item.train_indices]) & set(groups[item.target_indices]))


def test_confirmation_crossfit_never_uses_confirmation_fold() -> None:
    folds, groups = synthetic_fold_table()
    plan = make_crossfit_plan(folds, groups, [0, 2, 3, 4, 5, 6])
    for item in plan:
        assert 1 not in set(folds[item.train_indices])
        assert 1 not in set(folds[item.target_indices])


def test_crossfit_requires_distinct_folds_and_group_exclusivity() -> None:
    folds, groups = synthetic_fold_table()
    with pytest.raises(ValueError, match="at least two"):
        make_crossfit_plan(folds, groups, [2])
    groups[0] = groups[12]
    with pytest.raises(ValueError, match="overlap"):
        make_crossfit_plan(folds, groups, [0, 1])


def test_contract_hash_is_order_stable_for_json_keys() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_fixed_teacher_synthetic_crossfit_is_resumable_and_leakage_bounded(
    tmp_path,
) -> None:
    rng = np.random.default_rng(20260813)
    folds = np.repeat(np.arange(7), 12)
    labels = np.tile(np.asarray([0, 1] * 6), 7)
    groups = np.asarray([f"group_{index:03d}" for index in range(len(folds))])
    data = DevelopmentData(
        record_ids=np.asarray([f"record_{index:03d}" for index in range(len(folds))]),
        sequences=["ACDEFGHIKLMNPQRSTVWY" for _ in folds],
        labels=labels,
        groups=groups,
        folds=folds,
        b0=rng.normal(size=(len(folds), 25)).astype(np.float32),
        b1=rng.normal(size=(len(folds), 40)).astype(np.float32),
        b0_names=[f"b0_{index}" for index in range(25)],
        b1_names=[f"b1_{index}" for index in range(40)],
    )
    embedding = rng.normal(size=(len(folds), 24)).astype(np.float32)
    config = load_config(
        PLAN_ROOT / "config" / "DISTILLATION_CONFIG_20260813.json"
    )
    first = fixed_teacher_crossfit(
        data,
        embedding,
        [2, 3, 4, 5, 6],
        config=config,
        run_root=tmp_path,
        stage="synthetic_test",
        teacher_name="synthetic",
        embedding_source_hash="synthetic_embedding_hash",
    )
    second = fixed_teacher_crossfit(
        data,
        embedding,
        [2, 3, 4, 5, 6],
        config=config,
        run_root=tmp_path,
        stage="synthetic_test",
        teacher_name="synthetic",
        embedding_source_hash="synthetic_embedding_hash",
    )
    train_mask = np.isin(folds, [2, 3, 4, 5, 6])
    assert np.isfinite(first[train_mask]).all()
    assert np.isnan(first[~train_mask]).all()
    assert np.array_equal(first[train_mask], second[train_mask])

    indices, probabilities = fixed_teacher_evaluation(
        data,
        embedding,
        [2, 3, 4, 5, 6],
        0,
        config=config,
        run_root=tmp_path,
        stage="synthetic_test",
        teacher_name="synthetic",
        embedding_source_hash="synthetic_embedding_hash",
    )
    assert set(folds[indices]) == {0}
    assert probabilities.shape == (12,)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
