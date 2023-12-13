import pytest
from htrflow_core.main import aggregate_mean

@pytest.mark.parametrize("column, expected", [("feature_1", {0: 3, 1: 4})])
def test_aggregate_mean_feature_1(data, column, expected):
    assert expected == aggregate_mean(data, column)
