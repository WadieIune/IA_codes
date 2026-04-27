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



def score_inverse_head_shoulders(values: np.ndarray) -> float:
    inv = -np.asarray(values, dtype=float)
    return score_head_shoulders(inv)



def score_ascending_channel(values: np.ndarray) -> float:
    slope, r2, resid_std = _linear_fit_metrics(values)
    range_ = _series_range(values)
    maxima, minima = find_turning_points(values)
    oscillation_bonus = min((len(maxima) + len(minima)) / 8.0, 1.0)
    residual_score = 1.0 - min(resid_std / range_, 1.0)
    slope_score = 1.0 if slope > 0 else 0.0
    score = 0.40 * slope_score + 0.30 * max(r2, 0.0) + 0.20 * residual_score + 0.10 * oscillation_bonus
    return float(np.clip(score, 0.0, 1.0))



def score_descending_channel(values: np.ndarray) -> float:
    inv = -np.asarray(values, dtype=float)
    return score_ascending_channel(inv)



def score_high_tight_flag(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 12:
        return 0.0
    first = values[: n // 2]
    second = values[n // 2 :]
    range_ = _series_range(values)
    impulse = min((np.max(first) - np.min(first)) / range_, 1.0)
    initial_trend = max((first[-1] - first[0]) / range_, 0.0)
    consolidation_range = (np.max(second) - np.min(second)) / range_
    consolidation_score = 1.0 - min(consolidation_range / 0.5, 1.0)
    drift_score = max((values[-1] - np.median(second)) / range_, 0.0)
    score = 0.35 * impulse + 0.30 * initial_trend + 0.20 * consolidation_score + 0.15 * drift_score
    return float(np.clip(score, 0.0, 1.0))



def score_range(values: np.ndarray) -> float:
    slope, r2, resid_std = _linear_fit_metrics(values)
    range_ = _series_range(values)
    flatness = 1.0 - min(abs(slope) * len(values) / range_, 1.0)
    residual_score = min(resid_std / range_, 1.0)
    turning = find_turning_points(values)
    oscillation_score = min((len(turning[0]) + len(turning[1])) / 6.0, 1.0)
    score = 0.50 * flatness + 0.25 * (1.0 - max(r2, 0.0)) + 0.15 * residual_score + 0.10 * oscillation_score
    return float(np.clip(score, 0.0, 1.0))



def score_price_window(values: np.ndarray) -> Dict[str, float]:
    values = smooth_series(np.asarray(values, dtype=float), window=3)
    return {
        "double_bottom": score_double_bottom(values),
        "double_top": score_double_top(values),
        "ascending_channel": score_ascending_channel(values),
        "descending_channel": score_descending_channel(values),
        "high_tight_flag": score_high_tight_flag(values),
        "head_shoulders": score_head_shoulders(values),
        "inverse_head_shoulders": score_inverse_head_shoulders(values),
        "range": score_range(values),
    }



def label_price_window(values: np.ndarray, threshold: float = 0.35) -> Tuple[str, float, Dict[str, float]]:
    scores = score_price_window(values)
    best_label = max(scores, key=scores.get)
    confidence = float(scores[best_label])
    if confidence < threshold:
        return "unclassified", confidence, scores
    return best_label, confidence, scores



def class_to_index(label: str) -> int:
    if label not in PATTERN_CLASSES:
        raise ValueError(f"Clase desconocida: {label}")
    return PATTERN_CLASSES.index(label)

