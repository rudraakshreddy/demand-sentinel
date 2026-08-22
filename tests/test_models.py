"""Tests for model interfaces — fit/predict contracts and DB persistence shapes."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.arima_model import stationarity_tests, select_representative_series
from src.models.isolation_forest import build_anomaly_features
from src.risk.volatility import compute_volatility
from src.risk.shortfall import compute_sre


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_series() -> pd.Series:
    rng = np.random.default_rng(0)
    trend = np.linspace(10, 20, 200)
    noise = rng.normal(0, 2, 200)
    return pd.Series(trend + noise, index=pd.date_range("2019-01-01", periods=200))


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Multi-series panel for testing risk and anomaly functions."""
    rng = np.random.default_rng(1)
    n = 100
    records = []
    for item in ["FOOD_1", "FOOD_2"]:
        for store in ["CA_1", "TX_1"]:
            sales = rng.integers(0, 30, size=n).astype(float)
            sales[80] = 500.0   # artificial spike
            records.append(pd.DataFrame({
                "item_id":    item,
                "store_id":   store,
                "date":       pd.date_range("2021-01-01", periods=n),
                "sales":      sales,
                "sell_price": rng.uniform(1.0, 5.0, n),
                "dept_id":    "FOOD",
                "cat_id":     "FOOD",
                "state_id":   "CA",
            }))
    return pd.concat(records, ignore_index=True)


# ── ARIMA: stationarity tests ─────────────────────────────────────────────────

class TestStationarityTests:
    def test_returns_dict(self, sample_series):
        result = stationarity_tests(sample_series, "test_series")
        assert isinstance(result, dict)

    def test_adf_keys(self, sample_series):
        result = stationarity_tests(sample_series, "test_series")
        assert "adf_stat" in result
        assert "adf_p" in result
        assert "adf_reject" in result

    def test_stationary_series_detected(self):
        """White noise should be detected as stationary by ADF."""
        rng = np.random.default_rng(42)
        series = pd.Series(rng.normal(0, 1, 500))
        result = stationarity_tests(series, "white_noise")
        assert result.get("adf_reject") is True   # H0 of unit root rejected


class TestRepresentativeSeries:
    def test_returns_correct_n(self, sample_df):
        selected = select_representative_series(sample_df, n=4)
        assert len(selected) <= 4

    def test_returns_list_of_pairs(self, sample_df):
        selected = select_representative_series(sample_df, n=2)
        for pair in selected:
            assert len(pair) == 2   # (item_id, store_id)

    def test_reproducible(self, sample_df):
        s1 = select_representative_series(sample_df, n=4, seed=42)
        s2 = select_representative_series(sample_df, n=4, seed=42)
        assert s1 == s2


# ── Isolation Forest features ─────────────────────────────────────────────────

class TestAnomalyFeatures:
    def test_columns_created(self, sample_df):
        result = build_anomaly_features(sample_df.copy())
        for col in ["rolling_mean_28", "rolling_std_28", "z_score", "cv"]:
            assert col in result.columns, f"Missing: {col}"

    def test_z_score_spike_detected(self, sample_df):
        """The artificial spike at row 80 per series should have high z-score."""
        result = build_anomaly_features(sample_df.copy())
        spikes = result[result["sales"] > 400]
        assert len(spikes) > 0
        # z-scores for the spike rows should be well above 2
        assert (spikes["z_score"] > 2).all()

    def test_cv_nonnegative(self, sample_df):
        result = build_anomaly_features(sample_df.copy())
        assert (result["cv"].dropna() >= 0).all()


# ── Risk layer: volatility ────────────────────────────────────────────────────

class TestVolatility:
    def test_output_shape_preserved(self, sample_df):
        result = compute_volatility(sample_df.copy())
        assert len(result) == len(sample_df)

    def test_cv_columns_present(self, sample_df):
        result = compute_volatility(sample_df.copy())
        for w in [7, 14, 28]:
            assert f"cv_{w}" in result.columns

    def test_volatility_regime_values(self, sample_df):
        result = compute_volatility(sample_df.copy())
        assert set(result["volatility_regime"].dropna().unique()).issubset(
            {"Low", "Medium", "High"}
        )

    def test_spike_classified_high(self, sample_df):
        """Series with an extreme spike should have at least some High-regime rows."""
        result = compute_volatility(sample_df.copy())
        assert "High" in result["volatility_regime"].values


# ── Risk layer: shortfall ─────────────────────────────────────────────────────

class TestShortfall:
    def test_sre_column_created(self, sample_df):
        """SRE column appears when forecasts are merged."""
        # Create synthetic forecasts matching sample_df dates/series
        forecasts = sample_df[["item_id", "store_id", "date", "sales"]].copy()
        forecasts["forecast"] = forecasts["sales"] * 0.9
        merged = sample_df.merge(
            forecasts[["item_id", "store_id", "date", "forecast"]],
            on=["item_id", "store_id", "date"],
        )
        merged["error"] = merged["sales"] - merged["forecast"]
        result = compute_sre(merged)
        assert "sre" in result.columns
        assert "safety_stock" in result.columns

    def test_safety_stock_nonnegative(self, sample_df):
        forecasts = sample_df[["item_id", "store_id", "date", "sales"]].copy()
        forecasts["forecast"] = forecasts["sales"] * 0.9
        merged = sample_df.merge(
            forecasts[["item_id", "store_id", "date", "forecast"]],
            on=["item_id", "store_id", "date"],
        )
        merged["error"] = merged["sales"] - merged["forecast"]
        result = compute_sre(merged)
        assert (result["safety_stock"].dropna() >= 0).all()
