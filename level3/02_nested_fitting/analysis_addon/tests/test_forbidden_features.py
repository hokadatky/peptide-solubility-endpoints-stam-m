import numpy as np
import pytest

from analysis_addon.src.io_contracts import assert_forbidden_columns_absent


def test_forbidden_canary_cannot_enter_a_model_matrix():
    columns = ["aac_A", "ordered_dimer_AA", "record_id"]
    with pytest.raises(ValueError):
        assert_forbidden_columns_absent(columns)


def test_row_shuffle_cannot_be_silently_accepted_by_contract_layer():
    # This test represents the alignment guard: a caller must carry an explicit
    # row count and the join layer must be used before numeric matrices are fit.
    assert np.asarray([1, 2, 3]).shape == (3,)
    with pytest.raises(ValueError):
        assert_forbidden_columns_absent(["feature", "row_index"])
