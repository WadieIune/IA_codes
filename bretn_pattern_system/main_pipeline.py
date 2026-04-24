from __future__ import annotations

import argparse

from .inference import run as run_inference
from .train_tf import run as run_tf
from .train_torch import run as run_torch



def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline completo BRENT CNN chartism + outliers")
    parser.add_argument("--config", type=str, default=None, help="Ruta al JSON de configuración")
    parser.add_argument("--backend", type=str, default="tensorflow", choices=["tensorflow", "torch", "both"])
    parser.add_argument("--train", action="store_true", help="Ejecuta entrenamiento")
    parser.add_argument("--infer", action="store_true", help="Ejecuta inferencia")
    args = parser.parse_args()

    if args.backend in {"tensorflow", "both"} and args.train:
        run_tf(args.config)
    if args.backend in {"torch", "both"} and args.train:
        run_torch(args.config)
    if args.backend in {"tensorflow", "both"} and args.infer:
        run_inference(args.config, backend="tensorflow")
    if args.backend in {"torch", "both"} and args.infer:
        run_inference(args.config, backend="torch")


if __name__ == "__main__":
    main()
