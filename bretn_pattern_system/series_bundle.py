from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeriesBundle:
    raw_wide: pd.DataFrame
    zscore_wide: pd.DataFrame
    target: pd.Series
    feature_means: pd.Series
    feature_stds: pd.Series
    feature_cols: List[str]
    target_col: str
    price_col: str
    x_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    x_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    precomputed_lookback: Optional[int] = None
    precomputed_train_count: Optional[int] = None
    precomputed_test_count: Optional[int] = None


class BundleValidationError(RuntimeError):
    pass



def _resolve_dataset_path(dataset_dir: str | None, explicit_path: str | None, filename: str) -> str:
    if explicit_path:
        return explicit_path
    if not dataset_dir:
        return filename
    return os.path.join(dataset_dir, filename)



def _read_indexed_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el fichero requerido: {path}")
    df = pd.read_csv(path, index_col=0)
    try:
        df.index = pd.to_datetime(df.index, errors="raise")
    except Exception:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date")
        else:
            raise
    df = df.sort_index()
    return df



def _read_named_series(path: str) -> pd.Series:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el fichero requerido: {path}")
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"El fichero {path} no tiene formato de serie nombre-valor.")
    key_col = df.columns[0]
    value_col = df.columns[1]
    s = pd.Series(df[value_col].to_numpy(), index=df[key_col].astype(str).to_numpy(), dtype=float)
    s.index.name = None
    return s



def _maybe_load_npy(path: str | None) -> np.ndarray | None:
    if path and os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None



def load_series_bundle(config: Dict[str, object]) -> SeriesBundle:
    data_cfg = config["data"]["series_bundle"]
    dataset_dir = data_cfg.get("dataset_dir")

    raw_wide_path = _resolve_dataset_path(dataset_dir, data_cfg.get("raw_wide_path"), "dataset_wide_with_target.csv")
    zscore_path = _resolve_dataset_path(dataset_dir, data_cfg.get("zscore_path"), "dataset_wide_features_zscore.csv")
    means_path = _resolve_dataset_path(dataset_dir, data_cfg.get("means_path"), "feature_means.csv")
    stds_path = _resolve_dataset_path(dataset_dir, data_cfg.get("stds_path"), "feature_stds.csv")
    x_train_path = _resolve_dataset_path(dataset_dir, data_cfg.get("x_train_path"), "X_train.npy")
    y_train_path = _resolve_dataset_path(dataset_dir, data_cfg.get("y_train_path"), "y_train.npy")
    x_test_path = _resolve_dataset_path(dataset_dir, data_cfg.get("x_test_path"), "X_test.npy")
    y_test_path = _resolve_dataset_path(dataset_dir, data_cfg.get("y_test_path"), "y_test.npy")

    raw_wide = _read_indexed_csv(raw_wide_path)
    zscore_wide = _read_indexed_csv(zscore_path)
    means = _read_named_series(means_path)
    stds = _read_named_series(stds_path)

    target_col = str(data_cfg.get("bundle_target_col", "BRENT_fwd_logret_1"))
    price_col = str(data_cfg.get("bundle_price_col", "BRENT"))

    if target_col not in raw_wide.columns:
        raise BundleValidationError(f"La columna objetivo {target_col} no existe en {raw_wide_path}.")
    if price_col not in raw_wide.columns:
        raise BundleValidationError(f"La columna de precio {price_col} no existe en {raw_wide_path}.")

    feature_cols = [c for c in zscore_wide.columns]
    missing_features = [c for c in feature_cols if c not in raw_wide.columns]
    if missing_features:
        raise BundleValidationError(
            "Las columnas del z-score no están completas en el dataset raw. Primeras ausentes: "
            + ", ".join(missing_features[:10])
        )

    raw_subset = raw_wide[feature_cols].copy()
    target = raw_wide[target_col].copy()

    x_train = _maybe_load_npy(x_train_path)
    y_train = _maybe_load_npy(y_train_path)
    x_test = _maybe_load_npy(x_test_path)
    y_test = _maybe_load_npy(y_test_path)

    precomputed_lookback = None
    precomputed_train_count = None
    precomputed_test_count = None
    if x_train is not None:
        precomputed_lookback = int(x_train.shape[1])
        precomputed_train_count = int(x_train.shape[0])
    if x_test is not None:
        precomputed_test_count = int(x_test.shape[0])
        if precomputed_lookback is None:
            precomputed_lookback = int(x_test.shape[1])

    means = means.reindex(feature_cols)
    stds = stds.reindex(feature_cols)
    if means.isna().any():
        fill_means = raw_subset.mean(skipna=True)
        means = means.fillna(fill_means)
    if stds.isna().any():
        fill_stds = raw_subset.std(skipna=True).replace(0.0, 1.0)
        stds = stds.fillna(fill_stds)
    stds = stds.replace(0.0, 1.0)

    return SeriesBundle(
        raw_wide=raw_subset,
        zscore_wide=zscore_wide[feature_cols].copy(),
        target=target,
        feature_means=means.astype(float),
        feature_stds=stds.astype(float),
        feature_cols=feature_cols,
        target_col=target_col,
        price_col=price_col,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        precomputed_lookback=precomputed_lookback,
        precomputed_train_count=precomputed_train_count,
        precomputed_test_count=precomputed_test_count,
    )



def reconstruct_valid_windows(bundle: SeriesBundle, lookback: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    values = bundle.zscore_wide.to_numpy(dtype=float)
    target_values = bundle.target.to_numpy(dtype=float)
    index = bundle.zscore_wide.index

    X: List[np.ndarray] = []
    y: List[float] = []
    dates: List[pd.Timestamp] = []

    for end_idx in range(lookback - 1, len(bundle.zscore_wide), stride):
        start_idx = end_idx - lookback + 1
        x_win = values[start_idx : end_idx + 1]
        y_val = target_values[end_idx]
        if np.isnan(x_win).any() or np.isnan(y_val):
            continue
        X.append(x_win.astype(np.float32))
        y.append(np.float32(y_val))
        dates.append(index[end_idx])

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    dates_s = pd.Series(pd.to_datetime(dates), dtype="datetime64[ns]")
    return X_arr, y_arr, dates_s



def validate_series_bundle(bundle: SeriesBundle, lookback: int | None = None, n_checks: int = 3) -> Dict[str, object]:
    report: Dict[str, object] = {
        "raw_shape": [int(bundle.raw_wide.shape[0]), int(bundle.raw_wide.shape[1])],
        "zscore_shape": [int(bundle.zscore_wide.shape[0]), int(bundle.zscore_wide.shape[1])],
        "target_non_null": int(bundle.target.notna().sum()),
        "feature_count": int(len(bundle.feature_cols)),
        "price_col": bundle.price_col,
        "target_col": bundle.target_col,
    }

    if not bundle.raw_wide.index.equals(bundle.zscore_wide.index):
        raise BundleValidationError("Los índices temporales del raw wide y del z-score no coinciden.")

    if list(bundle.raw_wide.columns) != list(bundle.zscore_wide.columns):
        raise BundleValidationError("El orden de columnas entre raw wide y z-score no coincide.")

    if lookback is None:
        lookback = int(bundle.precomputed_lookback or 32)

    X_all, y_all, dates = reconstruct_valid_windows(bundle, lookback=lookback)
    report["reconstructed_window_count"] = int(len(X_all))
    report["reconstructed_first_date"] = str(dates.iloc[0].date()) if len(dates) else None
    report["reconstructed_last_date"] = str(dates.iloc[-1].date()) if len(dates) else None

    if bundle.x_train is not None and bundle.y_train is not None:
        report["precomputed_train_shape"] = [int(x) for x in bundle.x_train.shape]
        report["precomputed_y_train_shape"] = [int(x) for x in bundle.y_train.shape]
        report["train_matches_reconstruction"] = bool(
            bundle.x_train.shape[0] <= X_all.shape[0]
            and np.allclose(bundle.x_train[: min(n_checks, len(bundle.x_train))], X_all[: min(n_checks, len(bundle.x_train))], atol=1e-6, equal_nan=True)
            and np.allclose(bundle.y_train[: min(n_checks, len(bundle.y_train))], y_all[: min(n_checks, len(bundle.y_train))], atol=1e-6, equal_nan=True)
        )
    else:
        report["train_matches_reconstruction"] = None

    if bundle.x_test is not None and bundle.y_test is not None and bundle.x_train is not None and bundle.y_train is not None:
        start = bundle.x_train.shape[0]
        stop = start + bundle.x_test.shape[0]
        report["precomputed_test_shape"] = [int(x) for x in bundle.x_test.shape]
        report["precomputed_y_test_shape"] = [int(x) for x in bundle.y_test.shape]
        report["test_matches_reconstruction"] = bool(
            stop <= X_all.shape[0]
            and np.allclose(bundle.x_test[: min(n_checks, len(bundle.x_test))], X_all[start : start + min(n_checks, len(bundle.x_test))], atol=1e-6, equal_nan=True)
            and np.allclose(bundle.y_test[: min(n_checks, len(bundle.y_test))], y_all[start : start + min(n_checks, len(bundle.y_test))], atol=1e-6, equal_nan=True)
        )
    else:
        report["test_matches_reconstruction"] = None

    return report
