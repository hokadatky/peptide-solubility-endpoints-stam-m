import numpy as np

from analysis_addon.src.negative_controls import training_label_permutation_control


def test_training_label_permutation_is_checkpointed_and_never_shuffles_test_labels(tmp_path):
    rng = np.random.default_rng(101)
    n = 36
    y = np.tile([0, 1], n // 2)
    groups = np.repeat([f"g{i:02d}" for i in range(n // 2)], 2)
    folds = np.repeat(np.tile(np.arange(3), n // 6), 2)
    base = rng.normal(size=(n, 4))
    embedding = rng.normal(size=(n, 8))
    reference_info = [
        {"outer_fold": fold, "selected_c": 1.0, "selected_pls_k": None}
        for fold in range(3)
    ]
    candidate_info = [
        {"outer_fold": fold, "selected_c": 1.0, "selected_pls_k": 1}
        for fold in range(3)
    ]
    first = training_label_permutation_control(
        base, embedding, y, groups, folds, reference_info, candidate_info,
        3, 99, tmp_path, "negative-control-test",
    )
    second = training_label_permutation_control(
        base, embedding, y, groups, folds, reference_info, candidate_info,
        3, 99, tmp_path, "negative-control-test",
    )
    assert first == second
    assert first["training_labels_permuted"] is True
    assert first["test_labels_permuted"] is False
    assert first["n_replicates"] == 3
    assert len(list(tmp_path.glob("*.npz"))) == 3
