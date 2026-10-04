"""Probability calibration behavior, boundaries, and meaningful miscalibration fixture."""

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.exceptions import NotFittedError

from credit_risk.models.calibration import ProbabilityCalibrator


@pytest.mark.parametrize("method", ["raw", "sigmoid", "isotonic"])
def test_monotone_clipped_finite_and_fitted_parameters(method):
    p = np.linspace(0.01, 0.99, 100)
    y = np.tile([0, 0, 1, 0, 1], 20)
    calibrator = ProbabilityCalibrator(method).fit(p, y)
    values = calibrator.transform(np.r_[0, p, 1])
    assert np.isfinite(values).all() and (np.diff(values) >= 0).all()
    assert values.min() >= 1e-6 and values.max() <= 1 - 1e-6
    assert calibrator.parameters()["fit_rows"] == 100
    if method == "raw":
        assert np.allclose(values[1:-1], p)
    if method == "sigmoid":
        assert calibrator.slope_ > 0
    if method == "isotonic":
        assert len(np.unique(values)) < len(values)


def test_sigmoid_corrects_known_log_odds_distortion_on_independent_sample():
    rng = np.random.default_rng(12)
    true = expit(rng.normal(-1.3, 1, 12000))
    distorted = expit(1.8 * logit(true) + 0.7)
    y = rng.binomial(1, true)
    calibrator = ProbabilityCalibrator("sigmoid").fit(distorted[:8000], y[:8000])
    calibrated = calibrator.transform(distorted[8000:])
    assert np.mean((calibrated - y[8000:]) ** 2) < np.mean((distorted[8000:] - y[8000:]) ** 2)
    assert calibrator.slope_ == pytest.approx(1 / 1.8, abs=0.08)


@pytest.mark.parametrize(
    "probabilities,labels",
    [
        ([], []),
        ([np.nan, 0.5], [0, 1]),
        ([-0.1, 0.3], [0, 1]),
        ([0.1, 1.1], [0, 1]),
        ([0.1, 0.2], [0, 0]),
        ([0.1, 0.2], [0, 2]),
        ([0.1, 0.2], [1]),
    ],
)
def test_invalid_calibration_rejected(probabilities, labels):
    with pytest.raises(ValueError):
        ProbabilityCalibrator().fit(probabilities, labels)


def test_unfitted_and_unknown_method_rejected():
    with pytest.raises(NotFittedError):
        ProbabilityCalibrator().transform([0.1])
    with pytest.raises(ValueError, match="Unsupported"):
        ProbabilityCalibrator("unknown").fit([0.1, 0.2], [0, 1])
