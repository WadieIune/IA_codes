from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .config import PATTERN_CLASSES



def smooth_series(values: np.ndarray, window: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")



def find_turning_points(values: np.ndarray, min_separation: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.size < 5:
        return np.array([], dtype=int), np.array([], dtype=int)

    diff = np.diff(values)
    sign = np.sign(diff)
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    change = np.diff(sign)
    maxima = np.where(change < 0)[0] + 1
    minima = np.where(change > 0)[0] + 1

    def _sparsify(points: np.ndarray) -> np.ndarray:
        if points.size <= 1:
            return points
        kept = [int(points[0])]
        for point in points[1:]:
            if int(point) - kept[-1] >= min_separation:
                kept.append(int(point))
        return np.asarray(kept, dtype=int)

    return _sparsify(maxima), _sparsify(minima)



def _linear_fit_metrics(values: np.ndarray) -> Tuple[float, float, float]:
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    fitted = slope * x + intercept
    residual = values - fitted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    resid_std = float(np.std(residual))
    return float(slope), float(r2), resid_std



def _series_range(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values) + 1e-8)



def score_double_bottom(values: np.ndarray) -> float:
    maxima, minima = find_turning_points(values)
    if minima.size < 2:
        return 0.0
    range_ = _series_range(values)
    best = 0.0
    for i in range(len(minima) - 1):
        m1 = minima[i]
        m2 = minima[i + 1]
        if m2 - m1 < max(4, len(values) // 10):
            continue
        bridge_maxima = maxima[(maxima > m1) & (maxima < m2)]
        if bridge_maxima.size == 0:
            continue
        peak = bridge_maxima[np.argmax(values[bridge_maxima])]
        similarity = 1.0 - min(abs(values[m1] - values[m2]) / range_, 1.0)
        rebound = min((values[peak] - max(values[m1], values[m2])) / range_, 1.0)
        confirmation = min((values[-1] - values[m2]) / range_, 1.0)
        score = max(0.0, 0.45 * similarity + 0.35 * rebound + 0.20 * max(confirmation, 0.0))
        best = max(best, score)
    return float(np.clip(best, 0.0, 1.0))



def score_double_top(values: np.ndarray) -> float:
    maxima, minima = find_turning_points(values)
    if maxima.size < 2:
        return 0.0
    range_ = _series_range(values)
    best = 0.0
    for i in range(len(maxima) - 1):
        p1 = maxima[i]
        p2 = maxima[i + 1]
        if p2 - p1 < max(4, len(values) // 10):
            continue
        bridge_minima = minima[(minima > p1) & (minima < p2)]
        if bridge_minima.size == 0:
            continue
        valley = bridge_minima[np.argmin(values[bridge_minima])]
        similarity = 1.0 - min(abs(values[p1] - values[p2]) / range_, 1.0)
        retracement = min((min(values[p1], values[p2]) - values[valley]) / range_, 1.0)
        confirmation = min((values[p2] - values[-1]) / range_, 1.0)
        score = max(0.0, 0.45 * similarity + 0.35 * retracement + 0.20 * max(confirmation, 0.0))
        best = max(best, score)
    return float(np.clip(best, 0.0, 1.0))



def score_head_shoulders(values: np.ndarray) -> float:
    maxima, minima = find_turning_points(values)
    if maxima.size < 3:
        return 0.0
    range_ = _series_range(values)
    best = 0.0
    for i in range(len(maxima) - 2):
        left, head, right = maxima[i : i + 3]
        if not (left < head < right):
            continue
        side_similarity = 1.0 - min(abs(values[left] - values[right]) / range_, 1.0)
        head_margin = min((values[head] - max(values[left], values[right])) / range_, 1.0)
        neckline_points = minima[(minima > left) & (minima < right)]
        neckline = 0.0
        if neckline_points.size >= 1:
            neckline_level = float(np.mean(values[neckline_points]))
            neckline = min((min(values[left], values[right]) - neckline_level) / range_, 1.0)
        breakdown = min((values[head] - values[-1]) / range_, 1.0)
        score = max(0.0, 0.35 * side_similarity + 0.35 * head_margin + 0.15 * neckline + 0.15 * max(breakdown, 0.0))
        best = max(best, score)
    return float(np.clip(best, 0.0, 1.0))