from __future__ import annotations

import io
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests


class DataSourceError(RuntimeError):
    """Error controlado al descargar o parsear series externas."""



def _standardize_series_frame(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    df = df.copy()
    date_col = None
    value_col = None

    for candidate in ["date", "DATE", "observation_date", "TIME_PERIOD", "time_period"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        date_col = df.columns[0]

    for candidate in [value_name, "value", "VALUE", "OBS_VALUE", "obs_value"]:
        if candidate in df.columns:
            value_col = candidate
            break

    if value_col is None:
        numeric_candidates = [c for c in df.columns if c != date_col]
        if not numeric_candidates:
            raise DataSourceError("No se encontró una columna numérica en la serie descargada.")
        value_col = numeric_candidates[0]

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", value_name]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out



def fetch_fred_series(
    series_id: str,
    value_name: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    value_name = value_name or series_id.lower()

    if api_key:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if "observations" not in payload:
            raise DataSourceError(f"Respuesta inesperada de FRED para {series_id}: {payload}")
        df = pd.DataFrame(payload["observations"])[["date", "value"]].copy()
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))

    out = _standardize_series_frame(df, value_name)
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out.reset_index(drop=True)



def fetch_ecb_series(
    flow_ref: str,
    series_key: str,
    value_name: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    url = f"https://data-api.ecb.europa.eu/service/data/{flow_ref}/{series_key}?format=csvdata"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    out = _standardize_series_frame(df, value_name)
    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out.reset_index(drop=True)



def read_market_csv(path: str, date_col: str = "date", rename_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    if rename_map:
        df = df.rename(columns=rename_map)
    if "date" not in df.columns:
        raise ValueError("El CSV de entrada debe contener una columna de fecha.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = [c for c in df.columns if c != "date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)



def merge_series(series_frames: Iterable[pd.DataFrame], how: str = "outer") -> pd.DataFrame:
    frames: List[pd.DataFrame] = [df.copy() for df in series_frames]
    if not frames:
        raise ValueError("Debe proporcionarse al menos una serie para el merge.")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how=how)

    return merged.sort_values("date").drop_duplicates("date").reset_index(drop=True)



def build_default_brent_bundle(download_cfg: Dict[str, object]) -> pd.DataFrame:
    start = download_cfg.get("start")
    end = download_cfg.get("end")
    fred_api_key = download_cfg.get("fred_api_key")

    brent_cfg = download_cfg["brent"]
    eurusd_cfg = download_cfg["eurusd"]
    inflation_cfg = download_cfg["inflation"]

    brent = fetch_fred_series(
        series_id=brent_cfg["series_id"],
        value_name=brent_cfg.get("column_name", "brent_close"),
        start=start,
        end=end,
        api_key=fred_api_key,
    )

    if eurusd_cfg.get("provider") == "ecb":
        eurusd = fetch_ecb_series(
            flow_ref=eurusd_cfg.get("flow_ref", "EXR"),
            series_key=eurusd_cfg["series_key"],
            value_name=eurusd_cfg.get("column_name", "eurusd"),
            start=start,
            end=end,
        )
    else:
        eurusd = fetch_fred_series(
            series_id=eurusd_cfg["series_id"],
            value_name=eurusd_cfg.get("column_name", "eurusd"),
            start=start,
            end=end,
            api_key=fred_api_key,
        )

    inflation = fetch_fred_series(
        series_id=inflation_cfg["series_id"],
        value_name=inflation_cfg.get("column_name", "inflation"),
        start=start,
        end=end,
        api_key=fred_api_key,
    )

    return merge_series([brent, eurusd, inflation], how="outer")



def is_series_bundle_mode(config: Dict[str, object]) -> bool:
    return str(config.get("data", {}).get("mode", "csv")) == "series_bundle"



def load_data_from_config(config: Dict[str, object]) -> pd.DataFrame:
    data_cfg = config["data"]
    if is_series_bundle_mode(config):
        raise ValueError("load_data_from_config no debe usarse en modo 'series_bundle'. Usa load_series_bundle().")
    if data_cfg.get("mode", "csv") == "download" or data_cfg.get("download", {}).get("enabled", False):
        df = build_default_brent_bundle(data_cfg["download"])
    else:
        df = read_market_csv(
            path=data_cfg["csv_path"],
            date_col=data_cfg.get("date_col", "date"),
            rename_map=data_cfg.get("rename_map", {}),
        )
    return df