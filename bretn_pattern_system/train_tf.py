from __future__ import annotations

import argparse
import os

from .config import load_config, save_json
from .data_sources import is_series_bundle_mode, load_data_from_config
from .datasets import build_metadata_payload, prepare_dataset_from_bundle, prepare_dataset_from_config
from .feature_engineering import compute_market_features
from .outlier_control import fit_metric_templates
from .pattern_metrics import METRIC_COLUMNS
from .series_bundle import load_series_bundle, validate_series_bundle
from .tf_model import train_tf_pipeline



def run(config_path: str | None = None) -> dict:
    config = load_config(config_path)

    if is_series_bundle_mode(config):
        bundle = load_series_bundle(config)
        validation = validate_series_bundle(bundle, lookback=int(config["dataset"]["lookback"]))
        frame, window_table, feature_cols, extra = prepare_dataset_from_bundle(config, bundle)
    else:
        raw_df = load_data_from_config(config)
        market_df = compute_market_features(
            raw_df,
            target_col=str(config["data"].get("target_col", "brent_close")),
            resample_rule=str(config["data"].get("resample_rule", "B")),
        )
        frame, window_table, feature_cols = prepare_dataset_from_config(config, market_df)
        validation = {}
        extra = {"mode": "csv_or_download", "selected_feature_count": len(feature_cols)}

    output_cfg = config["output"]
    model_dir = output_cfg["model_dir"]
    metadata_dir = output_cfg["metadata_dir"]
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    window_table.to_csv(os.path.join(metadata_dir, output_cfg["window_table_name"]), index=False)
    save_json(os.path.join(metadata_dir, "dataset_metadata.json"), build_metadata_payload(feature_cols, config, extra=extra))
    if validation:
        save_json(os.path.join(metadata_dir, output_cfg.get("validation_name", "bundle_validation.json")), validation)

    templates = fit_metric_templates(window_table[window_table["split"] == "train"], metric_cols=METRIC_COLUMNS)
    save_json(os.path.join(metadata_dir, output_cfg["metric_template_name"]), templates)

    report = train_tf_pipeline(
        frame=frame,
        table=window_table,
        feature_cols=feature_cols,
        config=config,
        output_dir=model_dir,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento TensorFlow para BRENT chartism + outliers")
    parser.add_argument("--config", type=str, default=None, help="Ruta al JSON de configuración")
    args = parser.parse_args()
    run(args.config)