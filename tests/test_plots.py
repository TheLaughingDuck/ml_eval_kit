import pytest
from ml_eval.plots import get_conf_matrix


def test_get_conf_matrix_basic():
    # Test with basic input
    labels = [0, 0, 1]
    predictions = [0, 0, 1]
    n_classes = 2

    result = get_conf_matrix(labels, predictions, n_classes)

    expected_result = [[2, 0], [0, 1]]
    assert result == expected_result

def test_get_conf_matrix_2():

    labels = [0,1,2,2,1,0,0,1,2,2]
    predictions = [0,1,2,2,1,0,0,1,2,2]

    out = get_conf_matrix(labels, predictions, n_classes=3)

    assert out == [[3, 0, 0], [0, 3, 0], [0, 0, 4]], "Confusion matrix does not match expected output."