import inspect

import numpy as np
import pytest

from analysis_addon.src.nested_models import (
    fit_external_development_seal,
    predict_head,
)
from analysis_addon.src.run_soluprot_reanalysis import (
    _load_external_checkpoint,
    _save_external_checkpoint,
    run_external,
)


@pytest.mark.parametrize("mode", ["none", "pls"])
def test_external_development_seal_uses_three_disjoint_partitions(mode):
    rng = np.random.default_rng(121)
    n = 240
    y = np.tile([0, 1], n // 2)
    groups = np.asarray([f"g{index:03d}" for index in range(n)])
    base = rng.normal(size=(n, 5))
    base[:, 0] += 0.8 * y
    embedding = rng.normal(size=(n, 12))
    embedding[:, 0] += 1.2 * y
    fit = np.arange(0, 120)
    model_selection = np.arange(120, 180)
    calibration = np.arange(180, 240)
    head, calibrator, selected, grid, diagnostics = fit_external_development_seal(
        base,
        embedding if mode == "pls" else None,
        y,
        groups,
        fit,
        model_selection,
        calibration,
        mode,
        [0.1, 1.0],
        [1, 2],
    )
    raw = predict_head(
        head,
        base[calibration],
        embedding[calibration] if mode == "pls" else None,
    )
    calibrated = calibrator.predict(raw)
    assert selected[1] in {0.1, 1.0}
    assert all(row["eligible"] for row in grid)
    assert diagnostics["n_fit"] == 120
    assert diagnostics["n_model_selection"] == 60
    assert diagnostics["n_refit"] == 180
    assert diagnostics["n_calibration"] == 60
    assert diagnostics["refit"]["logistic"]["converged"] is True
    assert diagnostics["calibration"]["converged"] is True
    assert np.isfinite(calibrated).all()
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_external_seal_rejects_partition_group_overlap():
    rng = np.random.default_rng(122)
    y = np.tile([0, 1], 60)
    groups = np.asarray([f"g{index:03d}" for index in range(120)])
    groups[80] = groups[0]
    base = rng.normal(size=(120, 4))
    with pytest.raises(ValueError, match="Group overlap"):
        fit_external_development_seal(
            base, None, y, groups, np.arange(60), np.arange(60, 90),
            np.arange(90, 120), "none", [1.0], [1]
        )


def test_external_outcomes_are_not_an_input_to_development_seal():
    parameters = inspect.signature(fit_external_development_seal).parameters
    assert "external_y" not in parameters
    assert "external_labels" not in parameters
    source = inspect.getsource(run_external)
    assert source.index("atomic_write_json(seal_payload") < source.index(
        'y_ext = external["label"]'
    )
    assert "fit_mask = ~calibration_mask" not in source


def test_external_checkpoint_is_integrity_bound(tmp_path):
    path = tmp_path / "external.npz"
    raw = np.linspace(0.1, 0.9, 12)
    calibrated = np.linspace(0.2, 0.8, 12)
    diagnostics = {"refit": {"logistic": {"converged": True}}}
    _save_external_checkpoint(
        path, "contract-v2", raw, calibrated, (None, 1.0), [], 6, diagnostics
    )
    observed = _load_external_checkpoint(path, "contract-v2", 12)
    assert np.array_equal(observed["raw"], raw)
    with np.load(path, allow_pickle=False) as payload:
        values = {name: payload[name].copy() for name in payload.files}
    values["calibrated"][0] += 0.01
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        _load_external_checkpoint(path, "contract-v2", 12)
