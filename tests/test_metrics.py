import pytest
from ml_eval.metrics import calculate_metrics


def test_calculate_metrics_basic():
    label = [0, 0, 1]
    predi = [0, 0, 1]

    result = calculate_metrics(label, predi)

    # Assertions
    assert result["global"]["accuracy"] == 1.0
    assert result["global"]["n_missclassifications"] == 0


def test_calculate_metrics_longer_input():

    labels = [0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2]
    predic = [0,0,1,1,1,1,1,1,1,1,1,1,1,2,2,2]

    result = calculate_metrics(labels, predic)

    assert result["global"]["accuracy"] == 0.9375