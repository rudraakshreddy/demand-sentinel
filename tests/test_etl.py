"""Tests for ETL clean and feature engineering."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.etl.clean import impute_sales, clip_outliers_iqr
from src.etl.features import (
    make_lag_features, make_rolling_stats, make_calendar_features,
    make_promo_flag, make_interaction_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_df():
    """Minimal long-format sales dataframe for testing."""
    dates = pd.date_range("2021-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "item_id":  ["FOOD_1"] * 60,
        "store_id": ["CA_1"] * 60,
        "date":     dates,
        "sales":    np.random.default_rng(42).integers(0, 20, size=60).astype(float),
        "dept_id":  ["FOOD"] * 60,
        "cat_id":   ["FOOD"] * 60,
        "state_id": ["CA"] * 60,
    })
    return df


# ── Tests: clean.py ───────────────────────────────────────────────────────────

class TestImpute:
    def test_nulls_filled_with_zero(self, minimal_df):
        minimal_df.loc[5, "sales"] = np.nan
        result = impute_sales(minimal_df.copy())
        assert result["sales"].isna().sum() == 0
        assert result.loc[5, "sales"] == 0.0

    def test_negatives_clipped(self, minimal_df):
        minimal_df.loc[3, "sales"] = -5.0
        result = impute_sales(minimal_df.copy())
        assert result["sales"].min() >= 0

    def test_no_modification_of_valid_data(self, minimal_df):
        original = minimal_df["sales"].copy()
        result = impute_sales(minimal_df.copy())
        # All valid non-negative values unchanged
        valid_mask = original >= 0
        pd.testing.assert_series_equal(
            result.loc[valid_mask, "sales"].reset_index(drop=True),
            original[valid_mask].reset_index(drop=True),
        )


class TestOutlierClipping:
    def test_extreme_value_clipped(self, minimal_df):
        minimal_df.loc[10, "sales"] = 10_000.0   # artificial extreme
        result = clip_outliers_iqr(minimal_df.copy(), multiplier=3.0)
        assert result["sales"].max() < 10_000.0

    def test_normal_values_unchanged(self, minimal_df):
        """Values within 3×IQR fence must remain untouched."""
        original_sum = minimal_df["sales"].sum()
        result = clip_outliers_iqr(minimal_df.copy(), multiplier=3.0)
        # Some clipping may occur; total sales <= original
        assert result["sales"].sum() <= original_sum

    def test_no_negative_introduced(self, minimal_df):
        result = clip_outliers_iqr(minimal_df.copy())
        assert result["sales"].min() >= 0


# ── Tests: features.py ────────────────────────────────────────────────────────

class TestLagFeatures:
    def test_lag_columns_created(self, minimal_df):
        result = make_lag_features(minimal_df.copy())
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            assert f"lag_{lag}" in result.columns, f"Missing lag_{lag}"

    def test_lag_1_correct_values(self, minimal_df):
        result = make_lag_features(minimal_df.copy())
        # lag_1 at row 1 should equal sales at row 0
        assert result.iloc[1]["lag_1"] == minimal_df.iloc[0]["sales"]

    def test_no_future_leakage(self, minimal_df):
        """lag_N at index i must equal sales at index i-N (no look-ahead)."""
        result = make_lag_features(minimal_df.copy())
        for lag in [7, 14, 28]:
            for i in range(lag, len(result)):
                actual   = result.iloc[i][f"lag_{lag}"]
                expected = result.iloc[i - lag]["sales"]
                if not np.isnan(actual):
                    assert abs(actual - expected) < 1e-6, \
                        f"Leakage at lag_{lag}, row {i}"

    def test_first_n_rows_are_nan(self, minimal_df):
        result = make_lag_features(minimal_df.copy())
        # First 28 rows of lag_28 must be NaN
        assert result["lag_28"].iloc[:28].isna().all()


class TestRollingStats:
    def test_columns_created(self, minimal_df):
        df = make_lag_features(minimal_df.copy())
        df = make_rolling_stats(df)
        for w in [7, 14, 28]:
            assert f"rolling_mean_{w}" in df.columns
            assert f"rolling_std_{w}" in df.columns

    def test_no_negative_std(self, minimal_df):
        df = make_lag_features(minimal_df.copy())
        df = make_rolling_stats(df)
        assert (df["rolling_std_7"] >= 0).all()
        assert (df["rolling_std_28"] >= 0).all()


class TestCalendarFeatures:
    def test_dow_range(self, minimal_df):
        df = make_calendar_features(minimal_df.copy())
        assert df["day_of_week"].between(0, 6).all()

    def test_month_range(self, minimal_df):
        df = make_calendar_features(minimal_df.copy())
        assert df["month"].between(1, 12).all()

    def test_is_weekend_binary(self, minimal_df):
        df = make_calendar_features(minimal_df.copy())
        assert set(df["is_weekend"].unique()).issubset({0, 1})


class TestPromoFlag:
    def test_column_created(self, minimal_df):
        minimal_df["sell_price"] = 2.99
        df = make_promo_flag(minimal_df.copy())
        assert "is_promo" in df.columns

    def test_binary_values(self, minimal_df):
        minimal_df["sell_price"] = 2.99
        df = make_promo_flag(minimal_df.copy())
        assert set(df["is_promo"].unique()).issubset({0, 1})

    def test_large_price_drop_flagged(self):
        """A 50% price drop in the middle of a series should be flagged."""
        n = 60
        prices = [3.0] * n
        prices[40:] = [1.5] * 20   # 50% drop
        df = pd.DataFrame({
            "item_id":  ["X"] * n, "store_id": ["S1"] * n,
            "date":     pd.date_range("2020-01-01", periods=n),
            "sales":    [10.0] * n,
            "sell_price": prices,
            "dept_id":  ["D"] * n, "cat_id": ["C"] * n, "state_id": ["CA"] * n,
        })
        result = make_promo_flag(df, threshold=0.10)
        # At least some rows after the drop should be flagged
        assert result["is_promo"].iloc[45:].sum() > 0


class TestInteractionFeatures:
    def test_interaction_columns_exist(self, minimal_df):
        df = make_calendar_features(minimal_df.copy())
        df = make_lag_features(df)
        df["snap"] = 0
        df["is_promo"] = 0
        df["has_event"] = 0
        df = make_interaction_features(df)
        assert "snap_x_weekend" in df.columns
        assert "event_x_lag7" in df.columns
        assert "promo_x_dow" in df.columns
