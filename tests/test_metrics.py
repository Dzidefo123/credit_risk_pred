"""Hand-checkable discrimination and probability metric cases."""

import numpy as np
import pytest

from credit_risk.validation.metrics import binary_metrics


def test_perfect_ranking_gini_ks_and_probability_loss():
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    assert result["roc_auc"] == 1
    assert result["gini"] == 1
    assert result["ks"] == 1
    assert result["average_precision"] == 1
    assert result["brier"] == pytest.approx(0.025)
    assert result["precision"] == 1 and result["recall"] == 1


def test_tied_and_reversed_ranking():
    tied = binary_metrics([0, 1], [0.5, 0.5])
    assert tied["roc_auc"] == 0.5 and tied["gini"] == 0 and tied["ks"] == 0
    reversed_scores = binary_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert reversed_scores["gini"] == -1
    assert reversed_scores["ks"] == 1  # Two-sided KS is directionless.


def test_strong_ranking_can_have_poor_probability_loss():
    sharp = binary_metrics([0, 0, 1, 1], [0.01, 0.02, 0.98, 0.99])
    compressed = binary_metrics([0, 0, 1, 1], [0.45, 0.46, 0.54, 0.55])
    assert sharp["roc_auc"] == compressed["roc_auc"]
    assert sharp["brier"] < compressed["brier"]
    assert sharp["log_loss"] < compressed["log_loss"]


def test_single_class_metrics_are_explicitly_undefined():
    result = binary_metrics([0, 0], [0.1, 0.2])
    assert result["roc_auc"] is None and result["ks"] is None and result["pr_auc"] is None
    assert result["brier"] == pytest.approx(0.025)


@pytest.mark.parametrize(
    "labels, scores",
    [
        ([0, 1], [0.1]),
        ([], []),
        ([0, 2], [0.1, 0.2]),
        ([0, 1], [np.nan, 0.2]),
        ([0, 1], [-0.1, 0.2]),
        ([0, 1], [0.1, np.inf]),
    ],
)
def test_invalid_metric_inputs(labels, scores):
    with pytest.raises(ValueError):
        binary_metrics(labels, scores)
