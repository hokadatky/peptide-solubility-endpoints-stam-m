import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from analysis_addon.src import nested_models as nested_module
from analysis_addon.src.nested_models import (
    ModelConvergenceError,
    _fit_logistic,
    fit_head,
    nested_oof,
    predict_head,
    select_hyperparameter,
)
from analysis_addon.src.split_contracts import balanced_group_folds


def test_nested_pls_does_not_change_test_size_or_alignment():
    rng = np.random.default_rng(4)
    n = 300
    groups = np.asarray([f"g{i:03d}" for i in range(n)])
    y = np.tile([0, 1], n // 2)
    base = rng.normal(size=(n, 4))
    emb = rng.normal(size=(n, 12))
    outer = balanced_group_folds(y, groups, 5, 11)
    pred, info = nested_oof(
        base, emb, y, groups, outer, "pls", [0.1, 1.0], [1, 2], 3, 21
    )
    assert pred.shape == (n,)
    assert np.isfinite(pred).all()
    assert len(info) == 5


def test_pls_fit_transform_contract_and_forbidden_row_shift_detection_by_shape():
    rng = np.random.default_rng(5)
    y = np.asarray([0, 1] * 20)
    base = rng.normal(size=(40, 2))
    emb = rng.normal(size=(40, 8))
    head = fit_head(base, emb, y, "pls", 1.0, 2)
    prediction = predict_head(head, base, emb)
    assert prediction.shape == (40,)
    try:
        predict_head(head, base[:-1], emb)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched row alignment must fail closed.")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_max_k_pls_projection_contains_each_declared_prefix(dtype):
    rng = np.random.default_rng(73)
    embedding = rng.normal(size=(80, 18)).astype(dtype)
    embedding[:, 0] = 1.0
    embedding[:, 1] = (1.0 + rng.normal(scale=1e-7, size=80)).astype(dtype)
    y = rng.integers(0, 2, size=80)
    scaled = StandardScaler().fit_transform(embedding)
    full = PLSRegression(n_components=8, max_iter=500).fit(scaled, y).transform(scaled)
    for k in (1, 2, 3, 5, 8):
        separate = PLSRegression(n_components=k, max_iter=500).fit(scaled, y).transform(scaled)
        assert np.allclose(separate, full[:, :k], atol=1e-10, rtol=1e-10)


def test_logistic_nonconvergence_is_diagnosed_and_can_fail_closed():
    rng = np.random.default_rng(109)
    matrix = rng.normal(size=(240, 80))
    y = np.tile([0, 1], 120)
    _, diagnostics = _fit_logistic(
        matrix, y, 10_000.0, require_convergence=False, max_iter=1
    )
    assert diagnostics["converged"] is False
    assert diagnostics["convergence_warning"] is True
    assert diagnostics["n_iter"] == [1]
    assert any(row["category"] == "ConvergenceWarning" for row in diagnostics["warnings"])
    with pytest.raises(ModelConvergenceError, match="did not converge"):
        _fit_logistic(matrix, y, 10_000.0, require_convergence=True, max_iter=1)


def test_inner_nonconverged_candidate_is_ineligible(monkeypatch):
    rng = np.random.default_rng(110)
    n = 200
    y = np.tile([0, 1], n // 2)
    groups = np.asarray([f"g{index:03d}" for index in range(n)])
    base = rng.normal(size=(n, 6))
    original = nested_module._fit_logistic

    def controlled_fit(matrix, labels, c_value, *, require_convergence, max_iter=None):
        classifier, diagnostics = original(
            matrix,
            labels,
            c_value,
            require_convergence=require_convergence,
            max_iter=max_iter,
        )
        if float(c_value) == 10.0:
            diagnostics = {**diagnostics, "converged": False}
        return classifier, diagnostics

    monkeypatch.setattr(nested_module, "_fit_logistic", controlled_fit)
    selected, grid, _ = select_hyperparameter(
        base, None, y, groups, "none", [0.1, 10.0], [1], 5, 17
    )
    assert selected == (None, 0.1)
    rejected = next(row for row in grid if row["c"] == 10.0)
    assert rejected["eligible"] is False


def test_outer_refit_nonconvergence_stops(monkeypatch):
    rng = np.random.default_rng(111)
    base = rng.normal(size=(240, 80))
    y = np.tile([0, 1], 120)
    monkeypatch.setattr(nested_module, "LOGISTIC_MAX_ITER", 1)
    with pytest.raises(ModelConvergenceError, match="did not converge"):
        fit_head(base, None, y, "none", 10_000.0)


def test_pls_and_logistic_diagnostics_are_retained():
    rng = np.random.default_rng(112)
    y = np.tile([0, 1], 60)
    base = rng.normal(size=(120, 4))
    embedding = rng.normal(size=(120, 16))
    head = fit_head(base, embedding, y, "pls", 1.0, 3)
    assert head.diagnostics["pls"]["converged"] is True
    assert head.diagnostics["pls"]["projection_rank"] == 3
    assert head.diagnostics["logistic"]["converged"] is True
