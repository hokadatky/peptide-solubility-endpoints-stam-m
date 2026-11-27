import numpy as np
import pandas as pd
import pytest

from analysis_addon.src.recover_v2 import _validate_oof


def _fixture():
    train = pd.DataFrame(
        {
            "record_id": ["r0", "r1", "r2", "r3"],
            "cluster_id": ["g0", "g1", "g2", "g3"],
            "label": [0, 1, 0, 1],
            "development_fold": [0, 1, 0, 1],
        }
    )
    oof = train[["record_id", "cluster_id", "label"]].copy()
    oof["fold"] = train["development_fold"]
    oof["score"] = [0.1, 0.8, 0.2, 0.7]
    return train, oof


def test_recovery_oof_requires_exact_row_label_group_fold_alignment(tmp_path):
    train, oof = _fixture()
    path = tmp_path / "oof.parquet"
    oof.to_parquet(path, index=False)
    loaded, summary = _validate_oof(
        path,
        train,
        expected_folds=train["development_fold"].to_numpy(),
        expected_n_folds=2,
        require_record_id=True,
    )
    assert len(loaded) == 4
    assert summary["label_group_fold_alignment_checked"] is True


def test_recovery_oof_rejects_shifted_record_order(tmp_path):
    train, oof = _fixture()
    path = tmp_path / "shifted.parquet"
    oof.iloc[[1, 0, 2, 3]].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="record order mismatch"):
        _validate_oof(
            path,
            train,
            expected_folds=np.asarray([0, 1, 0, 1]),
            expected_n_folds=2,
            require_record_id=True,
        )
