from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest


PLAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLAN_ROOT / "src"))

from run_distillation_stage import (  # noqa: E402
    hybrid_gate,
    paired_student_bootstrap,
    student_contract_hash,
    write_private_predictions,
)
from teacher_crossfit import DevelopmentData, load_config  # noqa: E402


def frozen_config() -> dict[str, object]:
    return load_config(PLAN_ROOT / "config" / "DISTILLATION_CONFIG_20260813.json")


def test_student_contract_binds_exact_teacher_targets() -> None:
    config = frozen_config()
    train = np.asarray([0, 2, 4, 6], dtype=np.int64)
    first = np.asarray([0.1, np.nan, 0.2, np.nan, 0.3, np.nan, 0.4])
    second = first.copy()
    second[4] = 0.31
    common = {
        "stage": "synthetic",
        "arm": "PROTT5_DISTILLED_TRIPEPLITE",
        "seed": 20260813,
        "train": train,
    }
    first_hash = student_contract_hash(config, teacher_targets=first, **common)
    second_hash = student_contract_hash(config, teacher_targets=second, **common)
    assert first_hash != second_hash
    hard_hash = student_contract_hash(config, teacher_targets=None, **common)
    assert hard_hash not in {first_hash, second_hash}


def test_hybrid_gate_requires_all_four_predeclared_checks() -> None:
    config = frozen_config()
    metrics = {
        "PROTT5_TEACHER": {"log_loss": 0.600, "roc_auc": 0.800},
        "PROTT5_ESM2_TEACHER": {"log_loss": 0.590, "roc_auc": 0.799},
        "PROTT5_DISTILLED_TRIPEPLITE": {"log_loss": 0.620, "roc_auc": 0.760},
        "PROTT5_ESM2_DISTILLED_TRIPEPLITE": {
            "log_loss": 0.610,
            "roc_auc": 0.758,
        },
    }
    selected = hybrid_gate(metrics, config)
    assert selected["selected_teacher"] == "hybrid"
    assert all(selected["checks"].values())

    failed = copy.deepcopy(metrics)
    failed["PROTT5_ESM2_DISTILLED_TRIPEPLITE"]["log_loss"] = 0.619
    fallback = hybrid_gate(failed, config)
    assert fallback["selected_teacher"] == "prott5"
    assert fallback["checks"]["student_log_loss_improvement_pass"] is False


def test_primary_bootstrap_is_distilled_minus_hard_at_cluster_level() -> None:
    config = copy.deepcopy(frozen_config())
    config["confirmation"]["bootstrap_replicates"] = 200
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    hard = np.asarray([0.40, 0.65, 0.45, 0.70, 0.60, 0.55, 0.65, 0.50])
    distilled = np.asarray([0.10, 0.15, 0.20, 0.25, 0.90, 0.85, 0.80, 0.75])
    groups = np.asarray([f"g{index}" for index in range(len(labels))])
    result = paired_student_bootstrap(labels, hard, distilled, groups, config)
    assert set(result) == {"roc_auc", "auprc", "brier", "log_loss"}
    for metric_result in result.values():
        assert metric_result["bootstrap_unit"] == "cluster"
        assert metric_result["n_replicates_requested"] == 200
        assert metric_result["point_gain"] > 0


def test_private_prediction_parquet_is_resumable_and_conflict_safe(tmp_path) -> None:
    rows = 4
    data = DevelopmentData(
        record_ids=np.asarray([f"r{index}" for index in range(rows)]),
        sequences=["ACDEFG" for _ in range(rows)],
        labels=np.asarray([0, 1, 0, 1]),
        groups=np.asarray([f"g{index}" for index in range(rows)]),
        folds=np.asarray([0, 0, 0, 0]),
        b0=np.zeros((rows, 25), dtype=np.float32),
        b1=np.zeros((rows, 425), dtype=np.float32),
        b0_names=[f"b0_{index}" for index in range(25)],
        b1_names=[f"b1_{index}" for index in range(425)],
    )
    evaluation = np.arange(rows, dtype=np.int64)
    path = tmp_path / "private_predictions.parquet"
    scores = {"ARM": np.asarray([0.1, 0.8, 0.2, 0.9])}
    write_private_predictions(data, evaluation, scores, path)
    write_private_predictions(data, evaluation, scores, path)
    assert path.is_file()
    with pytest.raises(FileExistsError, match="different existing prediction"):
        write_private_predictions(
            data,
            evaluation,
            {"ARM": np.asarray([0.1, 0.8, 0.2, 0.7])},
            path,
        )
