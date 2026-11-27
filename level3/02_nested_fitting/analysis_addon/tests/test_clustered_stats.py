import numpy as np

from analysis_addon.src.clustered_stats import (
    decision_flags,
    gain_value,
    paired_cluster_bootstrap,
    cluster_swap_permutation,
)


def test_gain_sign_and_reproducible_cluster_bootstrap():
    y = np.asarray([0, 0, 1, 1, 0, 1])
    ref = np.asarray([0.2, 0.7, 0.3, 0.8, 0.4, 0.6])
    cand = np.asarray([0.1, 0.3, 0.7, 0.9, 0.2, 0.8])
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    assert np.isclose(gain_value(y, ref, cand, "roc_auc"), -gain_value(y, cand, ref, "roc_auc"))
    first = paired_cluster_bootstrap(y, ref, cand, groups, replicates=200, seed=19)
    second = paired_cluster_bootstrap(y, ref, cand, groups, replicates=200, seed=19)
    assert first == second
    assert first["ci90"][0] >= first["ci95"][0]
    assert first["ci90"][1] <= first["ci95"][1]


def test_identical_scores_have_zero_gain_and_swap_pvalue_one():
    y = np.asarray([0, 0, 1, 1])
    score = np.asarray([0.1, 0.2, 0.8, 0.9])
    groups = np.asarray(["a", "b", "c", "d"])
    result = paired_cluster_bootstrap(y, score, score, groups, replicates=100, seed=1)
    assert result["point_gain"] == 0.0
    assert result["ci90"] == [0.0, 0.0]
    swap = cluster_swap_permutation(y, score, score, groups, replicates=31, seed=2)
    assert swap["p_value_two_sided"] == 1.0


def test_vectorized_cluster_swap_matches_scalar_reference():
    y = np.asarray([0, 1, 0, 1, 1, 0, 1])
    ref = np.asarray([0.15, 0.55, 0.35, 0.65, 0.75, 0.45, 0.85])
    cand = np.asarray([0.10, 0.70, 0.30, 0.80, 0.60, 0.20, 0.90])
    groups = np.asarray(["g1", "g1", "g2", "g3", "g3", "g4", "g5"])
    replicates = 41
    seed = 773
    observed = gain_value(y, ref, cand, "roc_auc")
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(replicates):
        ref_scalar = ref.copy()
        cand_scalar = cand.copy()
        for group in sorted(np.unique(groups).tolist()):
            if bool(rng.integers(0, 2)):
                mask = groups == group
                ref_scalar[mask], cand_scalar[mask] = cand_scalar[mask].copy(), ref_scalar[mask].copy()
        null.append(gain_value(y, ref_scalar, cand_scalar, "roc_auc"))
    expected_p = (1 + np.sum(np.abs(null) >= abs(observed))) / (replicates + 1)
    actual = cluster_swap_permutation(y, ref, cand, groups, replicates=replicates, seed=seed)
    assert np.isclose(actual["p_value_two_sided"], expected_p)
    assert np.allclose(actual["null_ci95"], np.percentile(np.asarray(null), [2.5, 97.5]))


def test_margin_decision_keeps_statistical_and_practical_flags_separate():
    flags = decision_flags([0.005, 0.015], [0.001, 0.019], 0.02)
    assert flags["directional_advantage_95"] is True
    assert flags["meaningful_superiority_95"] is False
    assert flags["decision_status"] == "STATISTICAL_BUT_SUBMARGIN"
