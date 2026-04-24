from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .feature_engineering import robust_scale_matrix



def minmax_scale_minus_one_one(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if np.isclose(vmin, vmax):
        return np.zeros_like(values, dtype=float)
    scaled = 2.0 * (values - vmin) / (vmax - vmin) - 1.0
    return np.clip(scaled, -1.0, 1.0)



def gramian_angular_field(values: np.ndarray, method: str = "summation") -> np.ndarray:
    scaled = minmax_scale_minus_one_one(values)
    phi = np.arccos(np.clip(scaled, -1.0, 1.0))
    if method == "summation":
        return np.cos(phi[:, None] + phi[None, :])
    if method == "difference":
        return np.sin(phi[:, None] - phi[None, :])
    raise ValueError("method debe ser 'summation' o 'difference'.")



def recurrence_plot(values: np.ndarray, quantile: float = 0.20) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    dist = np.abs(values[:, None] - values[None, :])
    eps = np.quantile(dist, quantile)
    eps = eps if eps > 0 else 1.0
    return np.exp(-dist / eps)



def _interp_resize_1d(values: np.ndarray, new_length: int) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, num=len(values))
    x_new = np.linspace(0.0, 1.0, num=new_length)
    return np.interp(x_new, x_old, values)



def resize_2d(array: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if array.shape == new_shape:
        return array.copy()

    resized_rows = np.vstack([_interp_resize_1d(row, new_shape[1]) for row in array])
    tmp = np.asarray(resized_rows, dtype=float)
    cols = np.vstack([_interp_resize_1d(tmp[:, j], new_shape[0]) for j in range(tmp.shape[1])]).T
    return cols



def normalize_to_unit_interval(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    amin = np.nanmin(array)
    amax = np.nanmax(array)
    if np.isclose(amin, amax):
        return np.zeros_like(array, dtype=float)
    return (array - amin) / (amax - amin)



def build_feature_map(window_df: pd.DataFrame, feature_cols: Sequence[str], square_size: int) -> np.ndarray:
    feature_cols = [col for col in feature_cols if col in window_df.columns]
    if not feature_cols:
        raise ValueError("No hay columnas de features disponibles para construir la imagen.")
    matrix = window_df[feature_cols].T.to_numpy(dtype=float)
    matrix = robust_scale_matrix(matrix)
    matrix = np.clip((matrix + 3.0) / 6.0, 0.0, 1.0)
    feature_map = resize_2d(matrix, (square_size, square_size))
    return feature_map



def build_multichannel_image(
    window_df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    image_size: int = 240,
    output_dtype=np.float32,
) -> np.ndarray:
    """Convierte una ventana temporal multiactivo a una imagen RGB interpretable.

    Canal 1: GASF del nivel BRENT.
    Canal 2: GADF de retornos BRENT.
    Canal 3: mapa de features multivariantes (BRENT + exógenas).
    """
    brent = window_df[target_col].to_numpy(dtype=float)
    returns = np.diff(np.r_[brent[0], brent]) / np.maximum(brent, 1e-8)

    base_size = len(window_df)
    ch0 = normalize_to_unit_interval(gramian_angular_field(brent, method="summation"))
    ch1 = normalize_to_unit_interval(gramian_angular_field(returns, method="difference"))
    ch2 = build_feature_map(window_df, feature_cols=feature_cols, square_size=base_size)

    if base_size != image_size:
        ch0 = resize_2d(ch0, (image_size, image_size))
        ch1 = resize_2d(ch1, (image_size, image_size))
        ch2 = resize_2d(ch2, (image_size, image_size))

    img = np.stack([ch0, ch1, ch2], axis=-1)
    img = np.clip(img * 255.0, 0.0, 255.0).astype(output_dtype)
    return img