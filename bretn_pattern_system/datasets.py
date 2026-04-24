from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import PATTERN_CLASSES
from .feature_engineering import available_feature_columns, resolve_feature_columns, train_valid_test_boundaries
from .image_encoding import build_multichannel_image
from .pattern_metrics import METRIC_COLUMNS, compute_pattern_metrics
from .patterns import class_to_index, label_price_window
from .series_bundle import SeriesBundle
from .synthetic_patterns import generate_synthetic_example, synthetic_label_from_index


@dataclass(frozen=True)
class WindowRecord:
    start: int
    end: int
    end_date: str
    pattern_label: str
    label_confidence: float
    future_return: float
    future_low: float
    future_high: float



def build_window_table(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    stride: int = 1,
    target_col: str = "BRENT",
    min_label_confidence: float = 0.35,
    keep_unclassified: bool = False,
    future_return_col: str | None = None,
    future_return_is_log: bool = False,
    support_resistance_horizon: int | None = None,
    required_cols: Sequence[str] | None = None,
    skip_nan_windows: bool = True,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    df = df.reset_index(drop=True).copy()
    sr_horizon = int(support_resistance_horizon or horizon)
    max_future = max(int(horizon), int(sr_horizon))
    required_cols = list(required_cols or [])

    for end in range(lookback, len(df) - max_future + 1, stride):
        start = end - lookback
        window = df.iloc[start:end].copy()
        future = df.iloc[end : end + max_future].copy()

        if skip_nan_windows and required_cols:
            check_cols = [c for c in required_cols if c in window.columns]
            if check_cols and window[check_cols].isna().any().any():
                continue

        prices = window[target_col].to_numpy(dtype=float)
        if np.isnan(prices).any():
            continue

        label, confidence, scores = label_price_window(prices, threshold=min_label_confidence)
        if label == "unclassified" and not keep_unclassified:
            continue

        current_price = float(prices[-1])
        future_prices = future[target_col].to_numpy(dtype=float)
        if future_prices.size < max_future or np.isnan(future_prices[:max_future]).any():
            continue

        if future_return_col and future_return_col in window.columns:
            future_return = float(window[future_return_col].iloc[-1])
            if np.isnan(future_return):
                continue
            if future_return_is_log:
                future_return = float(np.expm1(future_return))
        else:
            target_future = future_prices[int(horizon) - 1]
            future_return = float(target_future / current_price - 1.0)

        sr_prices = future_prices[:sr_horizon]
        metrics = compute_pattern_metrics(prices)

        row: Dict[str, float] = {
            "start": start,
            "end": end,
            "end_idx": end - 1,
            "end_date": str(pd.Timestamp(window["date"].iloc[-1]).date()),
            "pattern_label": label,
            "label_confidence": confidence,
            "current_price": current_price,
            "future_return": future_return,
            "future_low": float(np.min(sr_prices) / current_price - 1.0),
            "future_high": float(np.max(sr_prices) / current_price - 1.0),
            "future_low_price": float(np.min(sr_prices)),
            "future_high_price": float(np.max(sr_prices)),
        }
        row.update(metrics)
        for pattern_name, pattern_score in scores.items():
            row[f"score_{pattern_name}"] = float(pattern_score)
        records.append(row)

    window_table = pd.DataFrame(records)
    if window_table.empty:
        raise ValueError("No se generaron ventanas de entrenamiento. Revisa lookback, horizon o las columnas de entrada.")

    window_table["pattern_idx"] = window_table["pattern_label"].apply(lambda x: class_to_index(x) if x in PATTERN_CLASSES else -1)
    return window_table



def assign_splits(
    window_table: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    precomputed_train_count: int | None = None,
    valid_ratio_within_precomputed_train: float = 0.10,
) -> pd.DataFrame:
    table = window_table.copy().reset_index(drop=True)
    split = np.full(len(table), "test", dtype=object)

    if precomputed_train_count is not None and 0 < precomputed_train_count < len(table):
        valid_count = int(round(precomputed_train_count * valid_ratio_within_precomputed_train))
        valid_count = min(max(valid_count, 1), max(precomputed_train_count - 1, 1))
        train_end = max(0, precomputed_train_count - valid_count)
        valid_end = precomputed_train_count
    else:
        train_end, valid_end = train_valid_test_boundaries(len(table), train_ratio, valid_ratio)

    split[:train_end] = "train"
    split[train_end:valid_end] = "valid"
    split[valid_end:] = "test"
    table["split"] = split
    return table



def build_image_for_record(
    df: pd.DataFrame,
    record: pd.Series,
    feature_cols: Sequence[str],
    image_size: int,
    target_col: str = "BRENT",
) -> np.ndarray:
    window = df.iloc[int(record["start"]) : int(record["end"])]
    return build_multichannel_image(window, target_col=target_col, feature_cols=feature_cols, image_size=image_size)



def build_metadata_payload(feature_cols: Sequence[str], config: Dict[str, object], extra: Dict[str, object] | None = None) -> Dict[str, object]:
    payload = {
        "feature_cols": list(feature_cols),
        "pattern_classes": list(PATTERN_CLASSES),
        "lookback": int(config["dataset"]["lookback"]),
        "horizon": int(config["dataset"]["horizon"]),
        "support_resistance_horizon": int(config["dataset"].get("support_resistance_horizon", config["dataset"]["horizon"])),
        "image_size": int(config["dataset"]["image_size"]),
        "target_col": str(config["data"].get("target_col", "BRENT")),
    }
    if extra:
        payload.update(extra)
    return payload



def prepare_dataset_from_config(config: Dict[str, object], market_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    feature_cols = available_feature_columns(market_df, config["dataset"]["feature_cols"])
    window_table = build_window_table(
        market_df,
        lookback=int(config["dataset"]["lookback"]),
        horizon=int(config["dataset"]["horizon"]),
        stride=int(config["dataset"].get("stride", 1)),
        target_col=str(config["data"].get("target_col", "brent_close")),
        min_label_confidence=float(config["dataset"].get("min_label_confidence", 0.35)),
        keep_unclassified=bool(config["dataset"].get("keep_unclassified", False)),
        support_resistance_horizon=int(config["dataset"].get("support_resistance_horizon", config["dataset"]["horizon"])),
        required_cols=[str(config["data"].get("target_col", "brent_close"))] + feature_cols,
    )
    window_table = assign_splits(
        window_table,
        train_ratio=float(config["dataset"].get("train_split", 0.70)),
        valid_ratio=float(config["dataset"].get("valid_split", 0.15)),
    )
    return market_df.reset_index(drop=True), window_table, feature_cols



def _bundle_feature_names(bundle: SeriesBundle, config: Dict[str, object]) -> List[str]:
    selected = resolve_feature_columns(
        bundle.feature_cols,
        requested=config["dataset"].get("feature_cols", []),
        regex_keep=config["dataset"].get("feature_regex_keep", []),
        regex_drop=config["dataset"].get("feature_regex_drop", []),
    )
    if not selected:
        raise ValueError("No quedan features tras aplicar feature_cols / feature_regex_keep / feature_regex_drop.")
    return selected



def prepare_dataset_from_bundle(config: Dict[str, object], bundle: SeriesBundle) -> tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, object]]:
    selected_features = _bundle_feature_names(bundle, config)
    prefixed_feature_cols = [f"z__{col}" for col in selected_features]

    base_frame = pd.DataFrame({
        "date": bundle.raw_wide.index,
        str(config["data"].get("target_col", bundle.price_col)): bundle.raw_wide[bundle.price_col].to_numpy(dtype=float),
        bundle.target_col: bundle.target.to_numpy(dtype=float),
    })
    z_frame = bundle.zscore_wide[selected_features].copy()
    z_frame.columns = prefixed_feature_cols
    frame = pd.concat([base_frame.reset_index(drop=True), z_frame.reset_index(drop=True)], axis=1)

    window_table = build_window_table(
        frame,
        lookback=int(config["dataset"]["lookback"]),
        horizon=int(config["dataset"]["horizon"]),
        stride=int(config["dataset"].get("stride", 1)),
        target_col=str(config["data"].get("target_col", bundle.price_col)),
        min_label_confidence=float(config["dataset"].get("min_label_confidence", 0.35)),
        keep_unclassified=bool(config["dataset"].get("keep_unclassified", False)),
        future_return_col=bundle.target_col,
        future_return_is_log=bool(config["data"]["series_bundle"].get("bundle_target_is_log_return", True)),
        support_resistance_horizon=int(config["dataset"].get("support_resistance_horizon", config["dataset"]["horizon"])),
        required_cols=[str(config["data"].get("target_col", bundle.price_col))] + prefixed_feature_cols,
        skip_nan_windows=True,
    )

    use_precomputed = bool(config["data"]["series_bundle"].get("use_precomputed_split", True))
    compatible_precomputed = (
        use_precomputed
        and bundle.precomputed_train_count is not None
        and bundle.precomputed_lookback == int(config["dataset"]["lookback"])
        and int(config["dataset"]["horizon"]) == 1
        and int(config["dataset"].get("stride", 1)) == 1
    )

    window_table = assign_splits(
        window_table,
        train_ratio=float(config["dataset"].get("train_split", 0.72)),
        valid_ratio=float(config["dataset"].get("valid_split", 0.08)),
        precomputed_train_count=int(bundle.precomputed_train_count) if compatible_precomputed else None,
        valid_ratio_within_precomputed_train=float(config["dataset"].get("valid_ratio_within_precomputed_train", 0.10)),
    )

    extra = {
        "mode": "series_bundle",
        "bundle_feature_count": len(bundle.feature_cols),
        "selected_feature_count": len(selected_features),
        "precomputed_train_count": bundle.precomputed_train_count,
        "precomputed_test_count": bundle.precomputed_test_count,
        "precomputed_lookback": bundle.precomputed_lookback,
    }
    return frame.reset_index(drop=True), window_table, prefixed_feature_cols, extra



def build_inference_table(
    config: Dict[str, object],
    market_df: pd.DataFrame,
    required_cols: Sequence[str] | None = None,
    skip_nan_windows: bool = True,
) -> tuple[pd.DataFrame, List[str]]:
    feature_cols = available_feature_columns(market_df, config["dataset"]["feature_cols"])
    lookback = int(config["dataset"]["lookback"])
    stride = int(config["dataset"].get("stride", 1))
    rows: List[Dict[str, object]] = []
    required_cols = list(required_cols or [])
    for end in range(lookback, len(market_df) + 1, stride):
        start = end - lookback
        window = market_df.iloc[start:end]
        if skip_nan_windows and required_cols:
            cols = [c for c in required_cols if c in window.columns]
            if cols and window[cols].isna().any().any():
                continue
        rows.append({"start": start, "end": end, "end_date": str(pd.Timestamp(market_df["date"].iloc[end - 1]).date())})
    if not rows:
        raise ValueError("No hay suficientes observaciones para construir ventanas de inferencia.")
    return pd.DataFrame(rows), feature_cols



def build_inference_table_for_bundle(
    config: Dict[str, object],
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    bundle: SeriesBundle,
) -> pd.DataFrame:
    lookback = int(config["dataset"]["lookback"])
    stride = int(config["dataset"].get("stride", 1))
    target_col = str(config["data"].get("target_col", bundle.price_col))
    required_cols = [target_col] + list(feature_cols)

    rows: List[Dict[str, object]] = []
    for end in range(lookback, len(frame) + 1, stride):
        start = end - lookback
        window = frame.iloc[start:end]
        if window[required_cols].isna().any().any():
            continue
        rows.append({"start": start, "end": end, "end_date": str(pd.Timestamp(frame["date"].iloc[end - 1]).date())})
    if not rows:
        raise ValueError("No hay suficientes observaciones válidas para construir ventanas de inferencia en el bundle.")
    return pd.DataFrame(rows)



def synthetic_example_from_index(index: int, config: Dict[str, object]) -> tuple[np.ndarray, int, np.ndarray]:
    label = synthetic_label_from_index(index)
    horizon = int(config["dataset"]["horizon"])
    sr_horizon = int(config["dataset"].get("support_resistance_horizon", horizon))
    sample = generate_synthetic_example(
        pattern_label=label,
        lookback=int(config["dataset"]["lookback"]),
        horizon=max(horizon, sr_horizon),
        seed=int(config["dataset"].get("synthetic_seed", 123)) + int(index),
    )
    image = build_multichannel_image(
        sample.frame,
        target_col="brent_close",
        feature_cols=["brent_close", "eurusd", "inflation"],
        image_size=int(config["dataset"]["image_size"]),
    )
    current = float(sample.frame["brent_close"].iloc[-1])
    future_path = sample.future_path
    target = np.asarray([
        float(future_path[horizon - 1] / current - 1.0),
        float(np.min(future_path[:sr_horizon]) / current - 1.0),
        float(np.max(future_path[:sr_horizon]) / current - 1.0),
    ], dtype=np.float32)
    return image, class_to_index(label), target
