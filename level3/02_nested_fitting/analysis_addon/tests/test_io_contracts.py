import numpy as np
import pandas as pd
import pytest

from analysis_addon.src.io_contracts import (
    assert_forbidden_columns_absent,
    one_to_one_join,
    validate_binary_labels,
    validate_canonical_sequence,
    validate_unique_ids,
)


def test_ids_labels_and_join_are_fail_closed():
    left = pd.DataFrame({"record_id": ["a", "b"], "label": [0, 1]})
    right = pd.DataFrame({"record_id": ["a", "b"], "feature": [1.0, 2.0]})
    validate_unique_ids(left, "record_id")
    validate_binary_labels(left, "label")
    joined = one_to_one_join(left, right, "record_id")
    assert joined.shape == (2, 3)
    with pytest.raises(ValueError):
        validate_unique_ids(pd.DataFrame({"record_id": ["a", "a"]}), "record_id")
    with pytest.raises(ValueError):
        validate_binary_labels(pd.DataFrame({"label": [0, 2]}), "label")


def test_canonical_sequence_does_not_silently_clean():
    assert validate_canonical_sequence("ACDE") == "ACDE"
    with pytest.raises(ValueError):
        validate_canonical_sequence("ACD-")
    with pytest.raises(ValueError):
        validate_canonical_sequence("")


def test_forbidden_feature_canary_is_rejected():
    assert_forbidden_columns_absent(["x1", "x2"])
    with pytest.raises(ValueError):
        assert_forbidden_columns_absent(["x1", "record_id"])
