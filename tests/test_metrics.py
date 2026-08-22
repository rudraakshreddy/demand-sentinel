"""Tests for evaluation metrics — computed from first principles."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (
    mape, smape, mae, rmse, wrmsse,
    prediction_interval_coverage, mean_interval_width,
    bias, compute_all_metrics,
)


class TestMAPE:
    def test_perfect_forecast(self):
        y = np.array([10.0, 20.0, 30.0])
        assert mape(y, y) == pytest.approx(0.0, abs=1e-9)

    def test_fifty_percent_error(self):
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([150.0, 50.0])
        assert mape(y_true, y_pred) == pytest.approx(0.5, rel=1e-6)

    def test_zeros_excluded(self):
        """Rows where y_true=0 must be excluded from MAPE."""
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([5.0, 15.0])
        # Only second row counts: |10-15|/10 = 0.5
        assert mape(y_true, y_pred) == pytest.approx(0.5, rel=1e-6)

    def test_all_zeros_returns_nan(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        assert np.isnan(mape(y_true, y_pred))


class TestSMAPE:
    def test_perfect_forecast(self):
        y = np.array([5.0, 10.0, 20.0])
        assert smape(y, y) == pytest.approx(0.0, abs=1e-9)

    def test_zero_actual(self):
        """sMAPE is defined even when y_true=0 (unlike MAPE)."""
        y_true = np.array([0.0])
        y_pred = np.array([4.0])
        result = smape(y_true, y_pred)
        assert not np.isnan(result)
        assert result == pytest.approx(2.0, rel=1e-6)  # 2 × |0-4|/(|0|+|4|) = 2.0

    def test_symmetry(self):
        """sMAPE should be symmetric: swapping y_true and y_pred gives same result."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([15.0, 25.0])
        assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true), rel=1e-6)


class TestMAE:
    def test_known_value(self):
        y_true = np.array([3.0, 5.0, 7.0])
        y_pred = np.array([1.0, 5.0, 9.0])
        # |3-1| + |5-5| + |7-9| = 2+0+2 = 4, mean = 4/3
        assert mae(y_true, y_pred) == pytest.approx(4.0 / 3.0, rel=1e-6)

    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0, abs=1e-9)


class TestRMSE:
    def test_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # sqrt((9+16)/2) = sqrt(12.5)
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5), rel=1e-6)

    def test_rmse_geq_mae(self):
        """RMSE ≥ MAE for any input (consequence of Jensen's inequality)."""
        rng = np.random.default_rng(7)
        y_true = rng.uniform(0, 100, 200)
        y_pred = rng.uniform(0, 100, 200)
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred) - 1e-9


class TestWRMSSE:
    def test_perfect_forecast_is_zero(self):
        train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y     = np.array([6.0, 7.0])
        assert wrmsse(y, y, train) == pytest.approx(0.0, abs=1e-9)

    def test_naive_scale(self):
        """If forecast matches the naive (random walk) model, WRMSSE ~ 1."""
        train  = np.arange(1.0, 101.0)  # 1,2,...,100
        y_true = np.array([101.0, 102.0])
        y_pred = np.array([100.0, 101.0])  # naive: repeat last value
        result = wrmsse(y_true, y_pred, train)
        assert result == pytest.approx(1.0, rel=0.01)


class TestIntervalMetrics:
    def test_coverage_perfect(self):
        y     = np.array([5.0, 10.0, 15.0])
        lower = np.array([0.0,  5.0, 10.0])
        upper = np.array([10.0, 15.0, 20.0])
        assert prediction_interval_coverage(y, lower, upper) == pytest.approx(1.0)

    def test_coverage_none(self):
        y     = np.array([100.0, 200.0])
        lower = np.array([0.0,   0.0])
        upper = np.array([1.0,   1.0])
        assert prediction_interval_coverage(y, lower, upper) == pytest.approx(0.0)

    def test_coverage_partial(self):
        y     = np.array([5.0, 100.0])
        lower = np.array([0.0,  0.0])
        upper = np.array([10.0,  5.0])
        assert prediction_interval_coverage(y, lower, upper) == pytest.approx(0.5)

    def test_interval_width(self):
        lower = np.array([0.0, 10.0])
        upper = np.array([5.0, 20.0])
        assert mean_interval_width(lower, upper) == pytest.approx(7.5)


class TestBias:
    def test_overforecast_is_negative(self):
        y_true = np.array([10.0])
        y_pred = np.array([15.0])
        assert bias(y_true, y_pred) == pytest.approx(-5.0)

    def test_unbiased(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([15.0,  5.0])
        # (10-15 + 20-5) / 2 = (-5+15)/2 = 5
        assert bias(y_true, y_pred) == pytest.approx(5.0)


class TestComputeAllMetrics:
    def test_keys_present(self):
        y = np.array([10.0, 20.0, 30.0])
        p = np.array([11.0, 19.0, 31.0])
        result = compute_all_metrics(y, p)
        for key in ["mape", "smape", "mae", "rmse", "bias"]:
            assert key in result, f"Missing key: {key}"

    def test_with_intervals(self):
        y = np.array([10.0, 20.0])
        p = np.array([10.0, 20.0])
        lo = np.array([5.0, 15.0])
        hi = np.array([15.0, 25.0])
        result = compute_all_metrics(y, p, lo, hi)
        assert "coverage_95" in result
        assert result["coverage_95"] == pytest.approx(1.0)
