from __future__ import annotations
import re
from typing import Iterable, List, Sequence
import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(3, window // 4)).mean()
    std = series.rolling(window, min_periods=max(3, window // 4)).std(ddof=0)
    return (series - mean) / (std.replace(0.0, np.nan))


def _ensure_business_frequency(df: pd.DataFrame, resample_rule: str = "B") -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    out = out.set_index("date")
    out = out.resample(resample_rule).last().ffill()
    out = out.reset_index()
    return out

def compute_market_features(
    df: pd.DataFrame,
    target_col: str = "brent_close",
    resample_rule: str = "B",
) -> pd.DataFrame:
    out = _ensure_business_frequency(df, resample_rule=resample_rule)

    if target_col != "brent_close":
        out = out.rename(columns={target_col: "brent_close"})

    numeric_cols = [c for c in out.columns if c != "date"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[numeric_cols] = out[numeric_cols].ffill().bfill()

    brent = out["brent_close"].astype(float)
    out["brent_return"] = np.log(brent).diff()
    out["brent_pct"] = brent.pct_change()
    out["brent_vol_10"] = out["brent_return"].rolling(10, min_periods=5).std(ddof=0)
    out["brent_vol_20"] = out["brent_return"].rolling(20, min_periods=10).std(ddof=0)
    out["brent_z_20"] = rolling_zscore(brent, 20)
    out["brent_ma_10"] = brent.rolling(10, min_periods=5).mean()
    out["brent_ma_30"] = brent.rolling(30, min_periods=10).mean()
    out["brent_ma_gap_10"] = brent / out["brent_ma_10"] - 1.0
    out["brent_ma_gap_30"] = brent / out["brent_ma_30"] - 1.0
    out["brent_roll_max_60"] = brent.rolling(60, min_periods=10).max()
    out["brent_drawdown_60"] = brent / out["brent_roll_max_60"] - 1.0

    if "eurusd" in out.columns:
        eurusd = out["eurusd"].astype(float)
        out["eurusd_return"] = np.log(eurusd).diff()
        out["brent_eurusd_corr_20"] = out["brent_return"].rolling(20, min_periods=10).corr(out["eurusd_return"])
        spread = out["brent_return"] - out["eurusd_return"]
        out["brent_eurusd_spread_z"] = rolling_zscore(spread, 20)
    else:
        out["eurusd_return"] = 0.0
        out["brent_eurusd_corr_20"] = 0.0
        out["brent_eurusd_spread_z"] = 0.0

    if "inflation" in out.columns:
        inflation = out["inflation"].astype(float)
        out["inflation_change_21"] = inflation.pct_change(21)
        out["inflation_change_126"] = inflation.pct_change(126)
        out["real_brent_proxy"] = out["brent_return"] - out["inflation_change_21"].fillna(0.0) / 21.0
    else:
        out["inflation_change_21"] = 0.0
        out["inflation_change_126"] = 0.0
        out["real_brent_proxy"] = out["brent_return"]

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.ffill().bfill().dropna().reset_index(drop=True)
    return out