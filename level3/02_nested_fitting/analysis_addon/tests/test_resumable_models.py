import json

import numpy as np
import pytest

from analysis_addon.src.resumable_models import checkpointed_nested_oof
def test_checkpointed_nested_oof_reuses_identical_outer_fold_artifacts(tmp_path):
    rng = np.random.default_rng(91)
    n = 48
    y = np.tile([0, 1], n // 2)
    groups = np.repeat([f"g{i:02d}" for i in range(n // 2)], 2)
    base = rng.normal(size=(n, 5))
    folds = np.repeat(np.tile(np.arange(4), n // 8), 2)
    first, first_info = checkpointed_nested_oof(
        base, None, y, groups, folds, "none", [0.1, 1.0], [1], 3, 17,
        tmp_path, "reference", "test-contract",
    )
    second, second_info = checkpointed_nested_oof(
        base, None, y, groups, folds, "none", [0.1, 1.0], [1], 3, 17,
        tmp_path, "reference", "test-contract",
    )
    assert np.array_equal(first, second)
    assert first_info == second_info
    assert len(list(tmp_path.glob("*.npz"))) == 4
    with np.load(tmp_path / "reference_outer00.npz", allow_pickle=False) as payload:
        contract = json.loads(str(payload["contract_json"][0]))
    assert contract["schema"] == "nested_oof_outer_fold_checkpoint_v2"
    assert contract["inner_assignment_hash"] == first_info[0]["inner_fold_hash"]
    assert set(contract["global_hashes"]) == {
        "base", "embedding", "groups", "labels", "outer_assignment"
    }


def test_checkpoint_contract_change_fails_closed(tmp_path):
    rng = np.random.default_rng(92)
    n = 48
    y = np.tile([0, 1], n // 2)
    groups = np.repeat([f"g{i:02d}" for i in range(n // 2)], 2)
    base = rng.normal(size=(n, 3))
    folds = np.repeat(np.tile(np.arange(4), n // 8), 2)
    checkpointed_nested_oof(
        base, None, y, groups, folds, "none", [1.0], [1], 2, 18,
        tmp_path, "reference", "contract-a",
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        checkpointed_nested_oof(
            base, None, y, groups, folds, "none", [1.0], [1], 2, 18,
            tmp_path, "reference", "contract-b",
        )


def test_checkpoint_same_shape_feature_change_and_old_schema_are_rejected(tmp_path):
    rng = np.random.default_rng(93)
    n = 48
    y = np.tile([0, 1], n // 2)
    groups = np.repeat([f"g{i:02d}" for i in range(n // 2)], 2)
    base = rng.normal(size=(n, 3))
    folds = np.repeat(np.tile(np.arange(4), n // 8), 2)
    checkpointed_nested_oof(
        base, None, y, groups, folds, "none", [1.0], [1], 2, 19,
        tmp_path, "reference", "contract-v2",
    )
    changed = base.copy()
    changed[0, 0] += 1e-6
    with pytest.raises(ValueError, match="contract mismatch"):
        checkpointed_nested_oof(
            changed, None, y, groups, folds, "none", [1.0], [1], 2, 19,
            tmp_path, "reference", "contract-v2",
        )

    old_dir = tmp_path / "old"
    old_dir.mkdir()
    np.savez_compressed(
        old_dir / "reference_outer00.npz",
        contract_json=np.asarray(['{"schema":"nested_oof_outer_fold_checkpoint_v1"}']),
        test_indices=np.asarray([], dtype=np.int64),
        scores=np.asarray([], dtype=float),
        fold_info_json=np.asarray(["{}"]),
    )
    with pytest.raises(ValueError, match="schema/integrity fields missing"):
        checkpointed_nested_oof(
            base, None, y, groups, folds, "none", [1.0], [1], 2, 19,
            old_dir, "reference", "contract-v2",
        )


def test_checkpoint_artifact_digest_detects_tampering(tmp_path):
    rng = np.random.default_rng(94)
    n = 48
    y = np.tile([0, 1], n // 2)
    groups = np.repeat([f"g{i:02d}" for i in range(n // 2)], 2)
    base = rng.normal(size=(n, 3))
    folds = np.repeat(np.tile(np.arange(4), n // 8), 2)
    checkpointed_nested_oof(
        base, None, y, groups, folds, "none", [1.0], [1], 2, 20,
        tmp_path, "reference", "tamper-test",
    )
    path = tmp_path / "reference_outer00.npz"
    with np.load(path, allow_pickle=False) as payload:
        values = {name: payload[name].copy() for name in payload.files}
    values["scores"][0] = min(1.0, float(values["scores"][0]) + 1e-4)
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        checkpointed_nested_oof(
            base, None, y, groups, folds, "none", [1.0], [1], 2, 20,
            tmp_path, "reference", "tamper-test",
        )
