from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import PATTERN_CLASSES
from .datasets import build_image_for_record, synthetic_example_from_index

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class FallbackConvBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.out_features = 192

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



def _load_backbone(force_backend: str | None = None) -> tuple[nn.Module, int, str]:
    backend_order = [force_backend] if force_backend else ["torchvision", "timm", "fallback"]
    for backend in backend_order:
        if backend == "torchvision":
            try:
                from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

                model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
                feature_dim = int(model.classifier[1].in_features)
                model.classifier = nn.Identity()
                return model, feature_dim, "torchvision"
            except Exception:
                try:
                    from torchvision.models import efficientnet_b1

                    model = efficientnet_b1(weights=None)
                    feature_dim = int(model.classifier[1].in_features)
                    model.classifier = nn.Identity()
                    return model, feature_dim, "torchvision"
                except Exception:
                    continue
        if backend == "timm":
            try:
                import timm

                try:
                    model = timm.create_model("efficientnet_b1", pretrained=True, num_classes=0, global_pool="avg")
                except Exception:
                    model = timm.create_model("efficientnet_b1", pretrained=False, num_classes=0, global_pool="avg")
                feature_dim = int(model.num_features)
                return model, feature_dim, "timm"
            except Exception:
                continue
        if backend == "fallback":
            model = FallbackConvBackbone()
            return model, model.out_features, "fallback"
    model = FallbackConvBackbone()
    return model, model.out_features, "fallback"



def _normalize_image(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class TorchBrentDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        table: pd.DataFrame,
        feature_cols: Sequence[str],
        config: Dict[str, object],
        split: str,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.table = table[table["split"] == split].reset_index(drop=True)
        self.feature_cols = list(feature_cols)
        self.config = config
        self.image_size = int(config["dataset"]["image_size"])
        self.target_col = str(config["data"].get("target_col", "BRENT"))

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        image = build_image_for_record(
            self.frame,
            row,
            feature_cols=self.feature_cols,
            image_size=self.image_size,
            target_col=self.target_col,
        )
        x = _normalize_image(image)
        y_class = torch.tensor(int(row["pattern_idx"]), dtype=torch.long)
        y_reg = torch.tensor(row[["future_return", "future_low", "future_high"]].to_numpy(dtype=np.float32), dtype=torch.float32)
        return x, y_class, y_reg


class SyntheticTorchDataset(Dataset):
    def __init__(self, config: Dict[str, object], n_samples: int) -> None:
        self.config = config
        self.n_samples = int(n_samples)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        image, label, target = synthetic_example_from_index(idx, self.config)
        x = _normalize_image(image)
        y_class = torch.tensor(int(label), dtype=torch.long)
        y_reg = torch.tensor(target, dtype=torch.float32)
        return x, y_class, y_reg


class BrentTorchModel(nn.Module):
    def __init__(self, dropout: float = 0.30, force_backend: str | None = None) -> None:
        super().__init__()
        backbone, feature_dim, backend = _load_backbone(force_backend=force_backend)
        self.backbone = backbone
        self.backend = backend
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(256, len(PATTERN_CLASSES))
        self.regressor = nn.Linear(256, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        shared = self.head(feats)
        return self.classifier(shared), self.regressor(shared)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_layers(self, n_last_layers: int = 40) -> None:
        children = list(self.backbone.children())
        if not children:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return
        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in children[-n_last_layers:]:
            for param in module.parameters():
                param.requires_grad = True


@dataclass
class EpochStats:
    loss: float
    cls_loss: float
    reg_loss: float
    accuracy: float
    mae: float



def _run_epoch(
    model: BrentTorchModel,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    regression_weight: float,
    train: bool,
) -> EpochStats:
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss()

    model.train(mode=train)
    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    total_correct = 0
    total_obs = 0
    total_mae = 0.0

    for x, y_class, y_reg in loader:
        x = x.to(device)
        y_class = y_class.to(device)
        y_reg = y_reg.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits, reg_out = model(x)
        cls_loss = cls_criterion(logits, y_class)
        reg_loss = reg_criterion(reg_out, y_reg)
        loss = cls_loss + regression_weight * reg_loss

        if train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == y_class).sum().item())
            total_obs += int(x.size(0))
            total_mae += float(torch.mean(torch.abs(reg_out - y_reg)).item()) * int(x.size(0))
            total_loss += float(loss.item()) * int(x.size(0))
            total_cls += float(cls_loss.item()) * int(x.size(0))
            total_reg += float(reg_loss.item()) * int(x.size(0))

    denom = max(1, total_obs)
    return EpochStats(
        loss=total_loss / denom,
        cls_loss=total_cls / denom,
        reg_loss=total_reg / denom,
        accuracy=total_correct / denom,
        mae=total_mae / denom,
    )



def _make_loader(dataset: Dataset, config: Dict[str, object], shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 16)),
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )



def _select_device(config: Dict[str, object]) -> torch.device:
    requested = str(config["training"].get("device", "cuda"))
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def train_torch_pipeline(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    feature_cols: Sequence[str],
    config: Dict[str, object],
    output_dir: str,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    device = _select_device(config)
    model = BrentTorchModel(dropout=float(config["training"].get("dropout", 0.30))).to(device)
    regression_weight = float(config["training"].get("regression_loss_weight", 0.50))

    train_ds = TorchBrentDataset(frame, table, feature_cols, config, split="train")
    valid_ds = TorchBrentDataset(frame, table, feature_cols, config, split="valid")
    train_loader = _make_loader(train_ds, config, shuffle=True)
    valid_loader = _make_loader(valid_ds, config, shuffle=False)

    model.freeze_backbone()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config["training"].get("learning_rate", 3e-4)))

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_accuracy": [], "train_mae": [],
        "valid_loss": [], "valid_accuracy": [], "valid_mae": [],
    }

    if bool(config["dataset"].get("use_synthetic_pretrain", True)):
        synth_ds = SyntheticTorchDataset(config, n_samples=int(config["dataset"].get("synthetic_samples", 8000)))
        synth_loader = _make_loader(synth_ds, config, shuffle=True)
        synth_epochs = max(1, min(5, int(config["training"].get("epochs", 12)) // 2))
        for _ in range(synth_epochs):
            _run_epoch(model, synth_loader, optimizer, device, regression_weight, train=True)

    best_state = None
    best_metric = -np.inf
    patience = int(config["training"].get("patience", 5))
    patience_left = patience

    for _ in range(int(config["training"].get("epochs", 12))):
        train_stats = _run_epoch(model, train_loader, optimizer, device, regression_weight, train=True)
        with torch.no_grad():
            valid_stats = _run_epoch(model, valid_loader, optimizer, device, regression_weight, train=False)

        history["train_loss"].append(train_stats.loss)
        history["train_accuracy"].append(train_stats.accuracy)
        history["train_mae"].append(train_stats.mae)
        history["valid_loss"].append(valid_stats.loss)
        history["valid_accuracy"].append(valid_stats.accuracy)
        history["valid_mae"].append(valid_stats.mae)

        if valid_stats.accuracy > best_metric:
            best_metric = valid_stats.accuracy
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.unfreeze_last_layers(n_last_layers=int(config["training"].get("unfreeze_last_layers", 40)))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config["training"].get("fine_tune_learning_rate", 1e-4)))

    best_state = None
    best_metric = -np.inf
    patience_left = patience
    fine_history: Dict[str, List[float]] = {
        "train_loss": [], "train_accuracy": [], "train_mae": [],
        "valid_loss": [], "valid_accuracy": [], "valid_mae": [],
    }

    for _ in range(int(config["training"].get("fine_tune_epochs", 8))):
        train_stats = _run_epoch(model, train_loader, optimizer, device, regression_weight, train=True)
        with torch.no_grad():
            valid_stats = _run_epoch(model, valid_loader, optimizer, device, regression_weight, train=False)

        fine_history["train_loss"].append(train_stats.loss)
        fine_history["train_accuracy"].append(train_stats.accuracy)
        fine_history["train_mae"].append(train_stats.mae)
        fine_history["valid_loss"].append(valid_stats.loss)
        fine_history["valid_accuracy"].append(valid_stats.accuracy)
        fine_history["valid_mae"].append(valid_stats.mae)

        if valid_stats.accuracy > best_metric:
            best_metric = valid_stats.accuracy
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = os.path.join(output_dir, str(config["output"].get("torch_model_name", "brent_efficientnet_b1_torch.pt")))
    torch.save({
        "state_dict": model.state_dict(),
        "pattern_classes": PATTERN_CLASSES,
        "backend": model.backend,
    }, model_path)

    report = {
        "model_path": model_path,
        "device": str(device),
        "backbone_backend": model.backend,
        "train_windows": int(len(train_ds)),
        "valid_windows": int(len(valid_ds)),
        "warm_history": history,
        "fine_tune_history": fine_history,
    }
    with open(os.path.join(output_dir, "torch_training_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report



def load_torch_model(model_path: str, config: Dict[str, object]) -> tuple[BrentTorchModel, torch.device]:
    device = _select_device(config)
    checkpoint = torch.load(model_path, map_location=device)
    model = BrentTorchModel(
        dropout=float(config["training"].get("dropout", 0.30)),
        force_backend=str(checkpoint.get("backend", "fallback")),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, device



def predict_torch_model(
    model_path: str,
    frame: pd.DataFrame,
    infer_table: pd.DataFrame,
    feature_cols: Sequence[str],
    config: Dict[str, object],
) -> pd.DataFrame:
    model, device = load_torch_model(model_path, config)
    target_col = str(config["data"].get("target_col", "BRENT"))
    image_size = int(config["dataset"]["image_size"])
    batch_size = int(config["training"].get("batch_size", 16))

    results: List[Dict[str, float]] = []
    batch_images: List[torch.Tensor] = []
    batch_rows: List[pd.Series] = []

    def _flush() -> None:
        nonlocal batch_images, batch_rows, results
        if not batch_images:
            return
        x = torch.stack(batch_images, dim=0).to(device)
        with torch.no_grad():
            logits, reg = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            reg_np = reg.cpu().numpy()

        for row, prob_row, level_row in zip(batch_rows, probs, reg_np):
            current_price = float(frame.iloc[int(row["end"]) - 1][target_col])
            pred_pattern = PATTERN_CLASSES[int(np.argmax(prob_row))]
            payload = {
                "start": int(row["start"]),
                "end": int(row["end"]),
                "end_date": row["end_date"],
                "predicted_pattern": pred_pattern,
                "pred_return": float(level_row[0]),
                "pred_low": float(level_row[1]),
                "pred_high": float(level_row[2]),
                "target_level": float(current_price * (1.0 + float(level_row[0]))),
                "support_level": float(current_price * (1.0 + float(level_row[1]))),
                "resistance_level": float(current_price * (1.0 + float(level_row[2]))),
            }
            for i, label in enumerate(PATTERN_CLASSES):
                payload[f"prob_{label}"] = float(prob_row[i])
            results.append(payload)
        batch_images = []
        batch_rows = []

    for _, row in infer_table.iterrows():
        image = build_image_for_record(frame, row, feature_cols=feature_cols, image_size=image_size, target_col=target_col)
        batch_images.append(_normalize_image(image))
        batch_rows.append(row)
        if len(batch_images) >= batch_size:
            _flush()
    _flush()

    return pd.DataFrame(results)
