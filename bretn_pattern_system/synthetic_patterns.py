from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import PATTERN_CLASSES


@dataclass(frozen=True)
class SyntheticExample:
    frame: pd.DataFrame
    label: str
    regression_target: np.ndarray
    future_path: np.ndarray


BASE_TEMPLATES: Dict[str, np.ndarray] = {
    "double_bottom": np.array([1.00, 0.86, 0.70, 0.86, 1.00, 0.88, 0.72, 0.90, 1.08]),
    "double_top": np.array([1.00, 1.12, 1.28, 1.14, 1.00, 1.12, 1.27, 1.08, 0.92]),
    "ascending_channel": np.array([0.82, 0.92, 0.88, 1.02, 0.98, 1.12, 1.08, 1.22, 1.18]),
    "descending_channel": np.array([1.20, 1.10, 1.14, 1.00, 1.04, 0.92, 0.96, 0.84, 0.88]),
    "high_tight_flag": np.array([0.78, 0.88, 1.05, 1.28, 1.55, 1.48, 1.46, 1.50, 1.62]),
    "head_shoulders": np.array([1.00, 1.18, 1.10, 1.34, 1.12, 1.20, 1.02]),
    "inverse_head_shoulders": np.array([1.00, 0.82, 0.90, 0.68, 0.90, 0.80, 1.02]),
    "range": np.array([1.00, 1.08, 0.97, 1.07, 0.96, 1.06, 0.98, 1.05, 0.99]),
}

FUTURE_BIAS: Dict[str, Tuple[float, float, float]] = {
    "double_bottom": (0.04, -0.01, 0.07),
    "double_top": (-0.04, -0.07, 0.01),
    "ascending_channel": (0.03, -0.01, 0.05),
    "descending_channel": (-0.03, -0.05, 0.01),
    "high_tight_flag": (0.06, -0.01, 0.09),
    "head_shoulders": (-0.05, -0.08, 0.01),
    "inverse_head_shoulders": (0.05, -0.01, 0.08),
    "range": (0.00, -0.03, 0.03),
}



def _interp_template(template: np.ndarray, length: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, num=len(template))
    xi = np.linspace(0.0, 1.0, num=length)
    return np.interp(xi, x, template)



def _ar1_noise(length: int, rng: np.random.Generator, phi: float = 0.65, sigma: float = 0.015) -> np.ndarray:
    noise = np.zeros(length, dtype=float)
    for i in range(1, length):
        noise[i] = phi * noise[i - 1] + rng.normal(0.0, sigma)
    return noise



def generate_synthetic_example(
    pattern_label: str,
    lookback: int = 64,
    horizon: int = 10,
    seed: int | None = None,
) -> SyntheticExample:
    if pattern_label not in PATTERN_CLASSES:
        raise ValueError(f"Patrón no soportado: {pattern_label}")

    rng = np.random.default_rng(seed)
    template = _interp_template(BASE_TEMPLATES[pattern_label], lookback)
    price_noise = _ar1_noise(lookback, rng, phi=0.72, sigma=0.018)
    local_noise = rng.normal(0.0, 0.010, size=lookback)
    warp = 1.0 + 0.03 * np.sin(np.linspace(0.0, 3.0 * np.pi, lookback) + rng.uniform(0.0, np.pi))

    price_path = template * warp + price_noise + local_noise
    price_path = np.maximum(price_path, 0.20)
    level = rng.uniform(45.0, 110.0)
    brent = level * (price_path / np.maximum(price_path[0], 1e-6))

    mu, low_bias, high_bias = FUTURE_BIAS[pattern_label]
    future_steps = rng.normal(mu / max(horizon, 1), 0.012, size=horizon)
    future_path = [brent[-1]]
    for step in future_steps:
        future_path.append(max(future_path[-1] * (1.0 + step), 1e-3))
    future_path = np.asarray(future_path[1:], dtype=float)

    latent = np.diff(np.r_[brent[0], brent]) / np.maximum(brent, 1e-6)
    rho = rng.uniform(-0.45, 0.45)
    eurusd = 1.05 + np.cumsum(rho * latent + rng.normal(0.0, 0.0015, size=lookback))
    eurusd = np.maximum(eurusd, 0.50)

    inflation_base = 100.0 + np.cumsum(rng.normal(0.01, 0.02, size=lookback))
    inflation_base += np.linspace(0.0, rng.uniform(0.1, 0.6), lookback)
    inflation = np.maximum(inflation_base, 1.0)

    dates = pd.bdate_range(start="2000-01-03", periods=lookback)
    frame = pd.DataFrame({"date": dates, "brent_close": brent, "eurusd": eurusd, "inflation": inflation})

    current = float(brent[-1])
    future_return = float(future_path[-1] / current - 1.0)
    future_low = float(np.min(future_path) / current - 1.0)
    future_high = float(np.max(future_path) / current - 1.0)

    regression_target = np.asarray([future_return, future_low, future_high], dtype=np.float32)
    return SyntheticExample(frame=frame, label=pattern_label, regression_target=regression_target, future_path=future_path)



def synthetic_label_from_index(index: int) -> str:
    return PATTERN_CLASSES[index % len(PATTERN_CLASSES)]
