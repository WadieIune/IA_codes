from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

PATTERN_CLASSES = [
    "double_bottom",
    "double_top",
    "ascending_channel",
    "descending_channel",
    "high_tight_flag",
    "head_shoulders",
    "inverse_head_shoulders",
    "range",
]

DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "mode": "series_bundle",
        "csv_path": "data/brent_market.csv",
        "date_col": "date",
        "target_col": "BRENT",
        "rename_map": {},
        "resample_rule": "B",
        "download": {
            "enabled": False,
            "fred_api_key": None,
            "brent": {"provider": "fred", "series_id": "DCOILBRENTEU", "column_name": "brent_close"},
            "eurusd": {"provider": "ecb", "flow_ref": "EXR", "series_key": "D.USD.EUR.SP00.A", "column_name": "eurusd"},
            "inflation": {"provider": "fred", "series_id": "CP0000EZ19M086NEST", "column_name": "inflation"},
            "start": "2000-01-01",
            "end": None,
        },
        "series_bundle": {
            "dataset_dir": "market_dataset",
            "raw_wide_path": None,
            "zscore_path": None,
            "means_path": None,
            "stds_path": None,
            "x_train_path": None,
            "y_train_path": None,
            "x_test_path": None,
            "y_test_path": None,
            "bundle_target_col": "BRENT_fwd_logret_1",
            "bundle_price_col": "BRENT",
            "bundle_target_is_log_return": True,
            "use_precomputed_split": True,
        },
    },
    "dataset": {
        "lookback": 32,
        "horizon": 1,
        "support_resistance_horizon": 5,
        "stride": 1,
        "image_size": 240,
        "train_split": 0.72,
        "valid_split": 0.08,
        "valid_ratio_within_precomputed_train": 0.10,
        "min_label_confidence": 0.35,
        "keep_unclassified": False,
        "use_synthetic_pretrain": True,
        "synthetic_samples": 8000,
        "synthetic_seed": 123,
        "feature_cols": [],
        "feature_regex_keep": [],
        "feature_regex_drop": [],
    },
    "training": {
        "epochs": 12,
        "fine_tune_epochs": 8,
        "batch_size": 16,
        "learning_rate": 3e-4,
        "fine_tune_learning_rate": 1e-4,
        "dropout": 0.30,
        "regression_loss_weight": 0.50,
        "unfreeze_last_layers": 40,
        "patience": 5,
        "num_workers": 0,
        "device": "cuda",
        "pretrained_backbone": True,
    },
    "outlier": {
        "rolling_window": 10,
        "target_cumulative_weight": 0.99,
        "score_threshold": 0.68,
        "min_dominant_confidence": 0.55,
        "metric_distance_threshold": 2.50,
        "volatility_spike_window": 20,
    },
    "output": {
        "root_dir": "artifacts",
        "model_dir": "artifacts/models",
        "report_dir": "artifacts/reports",
        "dataset_dir": "artifacts/datasets",
        "metadata_dir": "artifacts/metadata",
        "tf_model_name": "brent_efficientnet_b1_tf.keras",
        "torch_model_name": "brent_efficientnet_b1_torch.pt",
        "window_table_name": "window_table.csv",
        "prediction_report_name": "prediction_report.csv",
        "outlier_report_name": "outlier_report.csv",
        "metric_template_name": "metric_templates.json",
        "profile_name": "bundle_profile.json",
        "validation_name": "bundle_validation.json",
    },
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result
