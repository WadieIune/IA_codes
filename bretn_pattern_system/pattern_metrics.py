from __future__ import annotations

from typing import Dict

import numpy as np

from .patterns import find_turning_points, smooth_series



def compute_pattern_metrics(values: np.ndarray) -> Dict[str, float]:
    values = smooth_series(np.asarray(values, dtype=float), window=3)
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    fitted = slope * x + intercept
    residual = values - fitted

    range_ = float(np.max(values) - np.min(values) + 1e-8)
    maxima, minima = find_turning_points(values)
    returns = np.diff(np.r_[values[0], values]) / np.maximum(values, 1e-8)

    first_half = values[: max(2, len(values) // 2)]
    second_half = values[max(1, len(values) // 2) :]
    first_ret = returns[: max(2, len(returns) // 2)]
    second_ret = returns[max(1, len(returns) // 2) :]

    symmetry = 1.0
    if maxima.size >= 2:
        symmetry = 1.0 - min(abs(values[maxima[0]] - values[maxima[-1]]) / range_, 1.0)
    elif minima.size >= 2:
        symmetry = 1.0 - min(abs(values[minima[0]] - values[minima[-1]]) / range_, 1.0)

    metrics = {
        "trend_slope": float(slope),
        "residual_std": float(np.std(residual)),
        "pattern_range": range_,
        "n_maxima": float(len(maxima)),
        "n_minima": float(len(minima)),
        "symmetry_score": float(np.clip(symmetry, 0.0, 1.0)),
        "drawdown": float(np.min(values / np.maximum.accumulate(values) - 1.0)),
        "volatility_ratio": float(np.std(second_ret) / (np.std(first_ret) + 1e-8)),
        "compression_ratio": float((np.max(second_half) - np.min(second_half)) / (np.max(first_half) - np.min(first_half) + 1e-8)),
        "breakout_strength": float((values[-1] - np.median(values)) / range_),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
    }
    return metrics


METRIC_COLUMNS = [
    "trend_slope",
    "residual_std",
    "pattern_range",
    "n_maxima",
    "n_minima",
    "symmetry_score",
    "drawdown",
    "volatility_ratio",
    "compression_ratio",
    "breakout_strength",
    "return_mean",
    "return_std",
]
