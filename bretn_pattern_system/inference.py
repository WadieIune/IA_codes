from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import load_config
from .data_sources import is_series_bundle_mode, load_data_from_config
from .datasets import (
    build_inference_table,
    build_inference_table_for_bundle,
    prepare_dataset_from_bundle,
    prepare_dataset_from_config,
)
from .feature_engineering import compute_market_features
from .outlier_control import add_entropy_and_distance, apply_outlier_control, fit_metric_templates
from .pattern_metrics import METRIC_COLUMNS, compute_pattern_metrics
from .series_bundle import load_series_bundle
from .tf_model import predict_tf_model
from .torch_model import predict_torch_model



def _attach_window_metrics(frame: pd.DataFrame, infer_table: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for _, row in infer_table.iterrows():
        window = frame.iloc[int(row["start"]) : int(row["end"])]
        metrics = compute_pattern_metrics(window[target_col].to_numpy(dtype=float))
        payload = {"start": int(row["start"]), "end": int(row["end"]), **metrics}
        rows.append(payload)
    return pd.DataFrame(rows)



def _load_templates_or_fit(config: Dict[str, object], frame: pd.DataFrame, window_table: pd.DataFrame | None = None) -> Dict[str, object]:
    template_path = os.path.join(config["output"]["metadata_dir"], config["output"]["metric_template_name"])
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    if window_table is None:
        raise ValueError("No existe plantilla de métricas y tampoco se proporcionó window_table para ajustarla.")
    return fit_metric_templates(window_table[window_table["split"] == "train"], metric_cols=METRIC_COLUMNS)



def run(config_path: str | None = None, backend: str = "tensorflow") -> pd.DataFrame:
    config = load_config(config_path)

    if is_series_bundle_mode(config):
        bundle = load_series_bundle(config)
        frame, window_table, feature_cols, _ = prepare_dataset_from_bundle(config, bundle)
        infer_table = build_inference_table_for_bundle(config, frame, feature_cols, bundle)
        target_col = str(config["data"].get("target_col", bundle.price_col))
    else:
        raw_df = load_data_from_config(config)
        market_df = compute_market_features(
            raw_df,
            target_col=str(config["data"].get("target_col", "brent_close")),
            resample_rule=str(config["data"].get("resample_rule", "B")),
        )
        frame, window_table, feature_cols = prepare_dataset_from_config(config, market_df)
        infer_table, feature_cols = build_inference_table(
            config,
            frame,
            required_cols=[str(config["data"].get("target_col", "brent_close"))] + feature_cols,
            skip_nan_windows=True,
        )
        target_col = str(config["data"].get("target_col", "brent_close"))

    if backend == "tensorflow":
        model_path = os.path.join(config["output"]["model_dir"], config["output"]["tf_model_name"])
        pred_df = predict_tf_model(model_path, frame, infer_table, feature_cols, config)
    elif backend == "torch":
        model_path = os.path.join(config["output"]["model_dir"], config["output"]["torch_model_name"])
        pred_df = predict_torch_model(model_path, frame, infer_table, feature_cols, config)
    else:
        raise ValueError("backend debe ser 'tensorflow' o 'torch'")

    metric_df = _attach_window_metrics(frame, infer_table, target_col=target_col)
    pred_df = pred_df.merge(metric_df, on=["start", "end"], how="left")

    templates = _load_templates_or_fit(config, frame, window_table=window_table)
    pred_df = add_entropy_and_distance(pred_df, templates, metric_cols=METRIC_COLUMNS)
    pred_df = apply_outlier_control(pred_df, config)

    pred_df["current_price"] = pred_df["end"].apply(lambda e: float(frame.iloc[int(e) - 1][target_col]))
    pred_df["pred_return_pct"] = 100.0 * pred_df["pred_return"]
    pred_df["pred_low_pct"] = 100.0 * pred_df["pred_low"]
    pred_df["pred_high_pct"] = 100.0 * pred_df["pred_high"]

    out_path = os.path.join(config["output"]["report_dir"], f"{backend}_{config['output']['prediction_report_name']}")
    pred_df.to_csv(out_path, index=False)

    latest = pred_df.iloc[-1].to_dict()
    latest_path = os.path.join(config["output"]["report_dir"], f"{backend}_latest_signal.json")
    with open(latest_path, "w", encoding="utf-8") as fh:
        json.dump({k: (float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v) for k, v in latest.items()}, fh, indent=2, ensure_ascii=False)

    outlier_path = os.path.join(config["output"]["report_dir"], f"{backend}_{config['output']['outlier_report_name']}")
    pred_df[pred_df["outlier_flag"] == 1].to_csv(outlier_path, index=False)
    return pred_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia y control de outliers para BRENT")
    parser.add_argument("--config", type=str, default=None, help="Ruta al JSON de configuración")
    parser.add_argument("--backend", type=str, default="tensorflow", choices=["tensorflow", "torch"], help="Backend de inferencia")
    args = parser.parse_args()
    run(args.config, backend=args.backend)
