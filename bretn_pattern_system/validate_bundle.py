from __future__ import annotations

import argparse
import os

from .config import load_config, save_json
from .series_bundle import load_series_bundle, validate_series_bundle



def run(config_path: str | None = None) -> dict:
    config = load_config(config_path)
    bundle = load_series_bundle(config)
    report = validate_series_bundle(bundle, lookback=int(config["dataset"]["lookback"]))
    out_path = os.path.join(config["output"]["metadata_dir"], config["output"].get("validation_name", "bundle_validation.json"))
    save_json(out_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valida consistencia entre raw wide, z-score y tensores .npy")
    parser.add_argument("--config", type=str, default=None, help="Ruta al JSON de configuración")
    args = parser.parse_args()
    run(args.config)
