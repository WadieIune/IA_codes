from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd

from .config import load_config, save_json
from .datasets import prepare_dataset_from_bundle
from .series_bundle import load_series_bundle, validate_series_bundle



def _latest_starting_features(zscore_wide: pd.DataFrame, top_n: int = 15) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for col in zscore_wide.columns:
        first_valid = zscore_wide[col].first_valid_index()
        items.append({"feature": col, "first_valid_date": str(first_valid.date()) if first_valid is not None else None})
    items = sorted(items, key=lambda x: x["first_valid_date"] or "9999-12-31", reverse=True)
    return items[:top_n]



def run(config_path: str | None = None) -> Dict[str, object]:
    config = load_config(config_path)
    bundle = load_series_bundle(config)
    validation = validate_series_bundle(bundle, lookback=int(config["dataset"]["lookback"]))
    frame, window_table, feature_cols, extra = prepare_dataset_from_bundle(config, bundle)

    groups = {
        "raw_close_cols": [c for c in bundle.feature_cols if not any(c.endswith(s) for s in ["_logret1", "_vol20", "_rangepct", "_ma520"]) and not c.startswith("SPREAD_") and not c.startswith("RATIO_") and not c.endswith("_RATIO")],
        "logret_cols": [c for c in bundle.feature_cols if c.endswith("_logret1")],
        "vol20_cols": [c for c in bundle.feature_cols if c.endswith("_vol20")],
        "rangepct_cols": [c for c in bundle.feature_cols if c.endswith("_rangepct")],
        "ma520_cols": [c for c in bundle.feature_cols if c.endswith("_ma520")],
        "spread_ratio_cols": [c for c in bundle.feature_cols if c.startswith("SPREAD_") or c.startswith("RATIO_") or c.endswith("_RATIO")],
    }

    profile = {
        "validation": validation,
        "raw_wide_rows": int(bundle.raw_wide.shape[0]),
        "raw_wide_cols": int(bundle.raw_wide.shape[1]),
        "zscore_rows": int(bundle.zscore_wide.shape[0]),
        "zscore_cols": int(bundle.zscore_wide.shape[1]),
        "selected_feature_count": int(len(feature_cols)),
        "selected_prefixed_features": list(feature_cols[:20]),
        "window_count": int(len(window_table)),
        "window_first_date": str(window_table["end_date"].iloc[0]),
        "window_last_date": str(window_table["end_date"].iloc[-1]),
        "split_counts": {k: int(v) for k, v in window_table["split"].value_counts().to_dict().items()},
        "pattern_counts": {k: int(v) for k, v in window_table["pattern_label"].value_counts().to_dict().items()},
        "feature_group_counts": {k: len(v) for k, v in groups.items()},
        "latest_starting_features": _latest_starting_features(bundle.zscore_wide, top_n=20),
        "extra": extra,
    }

    out_path = os.path.join(config["output"]["metadata_dir"], config["output"].get("profile_name", "bundle_profile.json"))
    save_json(out_path, profile)
    window_table[["end_date", "pattern_label", "label_confidence", "future_return", "future_low", "future_high", "split"]].to_csv(
        os.path.join(config["output"]["metadata_dir"], "weak_label_window_summary.csv"), index=False
    )
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perfila el bundle procedente de SeriesDownloader_2_features.py")
    parser.add_argument("--config", type=str, default=None, help="Ruta al JSON de configuración")
    args = parser.parse_args()
    run(args.config)
