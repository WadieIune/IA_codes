from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PATTERN_CLASSES
from .pattern_metrics import METRIC_COLUMNS



def solve_decay_beta(window: int = 10, target_cumulative_weight: float = 0.99) -> float:
    """Resuelve beta para 1 - beta**window ~= target_cumulative_weight."""
    if window <= 0:
        raise ValueError("window debe ser positivo")
    target_cumulative_weight = float(np.clip(target_cumulative_weight, 1e-6, 1 - 1e-6))
    return float((1.0 - target_cumulative_weight) ** (1.0 / window))



def cumulative_decay_weights(window: int = 10, target_cumulative_weight: float = 0.99) -> tuple[np.ndarray, float]:
    beta = solve_decay_beta(window=window, target_cumulative_weight=target_cumulative_weight)
    weights = np.asarray([1.0 - beta ** (i + 1) for i in range(window)], dtype=float)
    return weights, beta



def normalized_incremental_weights(window: int = 10, target_cumulative_weight: float = 0.99) -> tuple[np.ndarray, float]:
    cumulative, beta = cumulative_decay_weights(window, target_cumulative_weight)
    incremental = np.diff(np.r_[0.0, cumulative])
    incremental = incremental / np.sum(incremental)
    return incremental, beta



def hadamard_temporal_weighting(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1, 1)
    if matrix.shape[0] != weights.shape[0]:
        raise ValueError("El número de filas de la matriz debe coincidir con la longitud del vector de pesos.")
    return matrix * weights



def categorical_weighted_vote(class_indices: Sequence[int], weights: Sequence[float], n_classes: int) -> np.ndarray:
    class_indices = np.asarray(class_indices, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if len(class_indices) != len(weights):
        raise ValueError("class_indices y weights deben tener la misma longitud.")
    vote = np.zeros(n_classes, dtype=float)
    for idx, weight in zip(class_indices, weights):
        if 0 <= idx < n_classes:
            vote[idx] += weight
    return vote



def fit_metric_templates(
    labeled_table: pd.DataFrame,
    metric_cols: Sequence[str] = METRIC_COLUMNS,
    label_col: str = "pattern_label",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    templates: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, group in labeled_table.groupby(label_col):
        templates[str(label)] = {
            "mean": {col: float(group[col].mean()) for col in metric_cols if col in group.columns},
            "std": {
                col: float(group[col].std(ddof=0)) if float(group[col].std(ddof=0)) > 1e-8 else 1.0
                for col in metric_cols
                if col in group.columns
            },
        }
    return templates



def metric_distance_from_template(
    row: pd.Series,
    predicted_label: str,
    templates: Dict[str, Dict[str, Dict[str, float]]],
    metric_cols: Sequence[str] = METRIC_COLUMNS,
) -> float:
    template = templates.get(predicted_label)
    if not template:
        return 0.0
    terms: List[float] = []
    for col in metric_cols:
        if col in row and col in template["mean"] and col in template["std"]:
            z = (float(row[col]) - float(template["mean"][col])) / (float(template["std"][col]) + 1e-8)
            terms.append(z * z)
    if not terms:
        return 0.0
    return float(math.sqrt(float(np.mean(terms))))



def add_entropy_and_distance(
    pred_df: pd.DataFrame,
    templates: Dict[str, Dict[str, Dict[str, float]]],
    metric_cols: Sequence[str] = METRIC_COLUMNS,
) -> pd.DataFrame:
    out = pred_df.copy()
    prob_cols = [f"prob_{label}" for label in PATTERN_CLASSES if f"prob_{label}" in out.columns]
    probs = out[prob_cols].to_numpy(dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    out["pattern_entropy"] = -np.sum(probs * np.log(probs), axis=1) / np.log(len(prob_cols))
    out["metric_distance"] = out.apply(
        lambda row: metric_distance_from_template(row, str(row["predicted_pattern"]), templates, metric_cols=metric_cols),
        axis=1,
    )
    return out



def volatility_spike(series: pd.Series, window: int = 20) -> pd.Series:
    base = series.rolling(window, min_periods=max(3, window // 4)).std(ddof=0)
    trend = base.rolling(window, min_periods=max(3, window // 4)).mean()
    spike = base / (trend + 1e-8)
    return spike.fillna(1.0)



def apply_outlier_control(
    pred_df: pd.DataFrame,
    config: Dict[str, object],
) -> pd.DataFrame:
    out = pred_df.copy().reset_index(drop=True)
    outlier_cfg = config["outlier"]
    prob_cols = [f"prob_{label}" for label in PATTERN_CLASSES if f"prob_{label}" in out.columns]
    if not prob_cols:
        raise ValueError("No se encontraron columnas de probabilidades para aplicar el control de outliers.")

    weights, beta = normalized_incremental_weights(
        window=int(outlier_cfg.get("rolling_window", 10)),
        target_cumulative_weight=float(outlier_cfg.get("target_cumulative_weight", 0.99)),
    )
    out["decay_beta"] = beta

    if "future_return" in out.columns:
        spike_source = pd.Series(out["future_return"], dtype=float)
    else:
        spike_source = pd.Series(out.get("pred_return", 0.0), dtype=float)
    out["volatility_spike"] = volatility_spike(spike_source, window=int(outlier_cfg.get("volatility_spike_window", 20)))

    dominant_pattern: List[str] = []
    dominant_confidence: List[float] = []
    weighted_score: List[float] = []
    outlier_flag: List[int] = []

    for i in range(len(out)):
        start = max(0, i - len(weights) + 1)
        hist = out.iloc[start : i + 1].copy()
        local_weights = weights[-len(hist) :]
        local_weights = local_weights / np.sum(local_weights)

        prob_matrix = hist[prob_cols].to_numpy(dtype=float)
        weighted_probs = np.sum(prob_matrix * local_weights[:, None], axis=0)
        best_idx = int(np.argmax(weighted_probs))
        best_pattern = prob_cols[best_idx].replace("prob_", "")
        best_conf = float(weighted_probs[best_idx])

        current_entropy = hist["pattern_entropy"].to_numpy(dtype=float)
        current_distance = 1.0 - np.exp(-hist["metric_distance"].to_numpy(dtype=float))
        current_spike = np.tanh(np.maximum(hist["volatility_spike"].to_numpy(dtype=float) - 1.0, 0.0))
        pattern_flip = (hist["predicted_pattern"].to_numpy(dtype=object) != best_pattern).astype(float)

        score = (
            0.45 * np.sum(local_weights * current_distance)
            + 0.25 * np.sum(local_weights * current_entropy)
            + 0.20 * np.sum(local_weights * current_spike)
            + 0.10 * np.sum(local_weights * pattern_flip)
        )
        score = float(np.clip(score, 0.0, 1.0))
        is_outlier = int(
            score >= float(outlier_cfg.get("score_threshold", 0.68))
            or best_conf < float(outlier_cfg.get("min_dominant_confidence", 0.55))
            or float(hist["metric_distance"].iloc[-1]) >= float(outlier_cfg.get("metric_distance_threshold", 2.5))
        )

        dominant_pattern.append(best_pattern)
        dominant_confidence.append(best_conf)
        weighted_score.append(score)
        outlier_flag.append(is_outlier)

    out["dominant_pattern_window"] = dominant_pattern
    out["dominant_confidence_window"] = dominant_confidence
    out["weighted_outlier_score"] = weighted_score
    out["outlier_flag"] = outlier_flag
    return out