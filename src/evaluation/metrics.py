"""
Evaluation Layer — metrics.py

Complete metric suite for demand forecasting evaluation.

All metrics implemented from first principles with citations:
  - MAPE : Hyndman & Koehler (2006) — "Another look at measures of forecast accuracy"
  - sMAPE: Makridakis (1993)
  - WRMSSE: M5 Competition official metric (Makridakis et al., 2020)
  - Coverage: Chatfield (1993) — "Calculating interval forecasts"
"""

import numpy as np
import pandas as pd


def mape(y_true: np.ndarray, y_pred: np.ndarray,
         epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    Undefined when y_true = 0; rows with y_true < epsilon are excluded.
    Range: [0, ∞). Lower is better.
    """
    mask = y_true > epsilon
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (Makridakis, 1993).
    Defined for y_true = 0 (unlike MAPE). Range: [0, 2]. Lower is better.
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, np.abs(y_true - y_pred) / denom, 0.0)
    return float(np.mean(ratio))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error. Units same as the target. Robust to outliers."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error. Penalises large errors quadratically."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wrmsse(y_true: np.ndarray, y_pred: np.ndarray,
           train_series: np.ndarray, scale_weights: np.ndarray | None = None) -> float:
    """
    Weighted Root Mean Squared Scaled Error — M5 Competition official metric.

    RMSSE_i = RMSE_i / (1/(n-1) * Σ(y_{t} - y_{t-1})^2)^0.5  [scaling]
    WRMSSE  = Σ w_i * RMSSE_i  [aggregation with revenue weights]

    Reference: Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020).
    """
    # Naive scale: mean absolute diff of training series
    if len(train_series) < 2:
        return np.nan
    naive_mae = np.mean(np.abs(np.diff(train_series)))
    if naive_mae == 0:
        return np.nan
    rmsse_val = rmse(y_true, y_pred) / naive_mae
    if scale_weights is not None:
        return float(np.average([rmsse_val], weights=scale_weights))
    return float(rmsse_val)


def prediction_interval_coverage(y_true: np.ndarray,
                                  lower: np.ndarray,
                                  upper: np.ndarray) -> float:
    """
    Fraction of actuals that fall within [lower, upper].
    For a correctly calibrated 95% CI, this should be ≈ 0.95.
    """
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average width of the prediction interval (narrower = more precise)."""
    return float(np.mean(upper - lower))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Error (signed). Positive = systematic underforecast."""
    return float(np.mean(y_true - y_pred))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         lower_ci: np.ndarray | None = None,
                         upper_ci: np.ndarray | None = None,
                         train_series: np.ndarray | None = None) -> dict:
    """Compute the full metric suite and return as a dict."""
    result = {
        "mape":  mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mae":   mae(y_true, y_pred),
        "rmse":  rmse(y_true, y_pred),
        "bias":  bias(y_true, y_pred),
    }
    if lower_ci is not None and upper_ci is not None:
        result["coverage_95"]    = prediction_interval_coverage(y_true, lower_ci, upper_ci)
        result["interval_width"] = mean_interval_width(lower_ci, upper_ci)
    if train_series is not None:
        result["wrmsse"] = wrmsse(y_true, y_pred, train_series)
    return result
