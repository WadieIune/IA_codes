from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PATTERN_CLASSES
from .datasets import build_image_for_record, synthetic_example_from_index



def _require_tf():
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - depende del entorno del usuario
        raise ImportError(
            "TensorFlow no está instalado. Instala tensorflow>=2.16 en tu entorno local para usar esta rama."
        ) from exc
    return tf



def _real_generator(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    feature_cols: Sequence[str],
    image_size: int,
    target_col: str,
):
    for _, row in table.iterrows():
        image = build_image_for_record(frame, row, feature_cols=feature_cols, image_size=image_size, target_col=target_col)
        label = np.int32(row["pattern_idx"])
        levels = row[["future_return", "future_low", "future_high"]].to_numpy(dtype=np.float32)
        yield image.astype(np.float32), {"pattern": label, "levels": levels}



def _synthetic_generator(n_samples: int, config: Dict[str, object]):
    for idx in range(int(n_samples)):
        image, label, target = synthetic_example_from_index(idx, config)
        yield image.astype(np.float32), {"pattern": np.int32(label), "levels": target.astype(np.float32)}



def build_tf_dataset(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    feature_cols: Sequence[str],
    config: Dict[str, object],
    split: str,
    shuffle: bool = False,
):
    tf = _require_tf()
    image_size = int(config["dataset"]["image_size"])
    batch_size = int(config["training"]["batch_size"])
    target_col = str(config["data"].get("target_col", "brent_close"))

    subset = table[table["split"] == split].reset_index(drop=True)
    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        {
            "pattern": tf.TensorSpec(shape=(), dtype=tf.int32),
            "levels": tf.TensorSpec(shape=(3,), dtype=tf.float32),
        },
    )

    ds = tf.data.Dataset.from_generator(
        lambda: _real_generator(frame, subset, feature_cols, image_size=image_size, target_col=target_col),
        output_signature=output_signature,
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=max(256, len(subset)), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, subset



def build_tf_synthetic_dataset(config: Dict[str, object], n_samples: int):
    tf = _require_tf()
    image_size = int(config["dataset"]["image_size"])
    batch_size = int(config["training"]["batch_size"])

    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        {
            "pattern": tf.TensorSpec(shape=(), dtype=tf.int32),
            "levels": tf.TensorSpec(shape=(3,), dtype=tf.float32),
        },
    )
    ds = tf.data.Dataset.from_generator(
        lambda: _synthetic_generator(n_samples=n_samples, config=config),
        output_signature=output_signature,
    )
    ds = ds.shuffle(buffer_size=max(256, n_samples), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds



def build_tf_model(config: Dict[str, object]):
    tf = _require_tf()

    image_size = int(config["dataset"]["image_size"])
    dropout = float(config["training"].get("dropout", 0.30))
    lr = float(config["training"].get("learning_rate", 3e-4))
    regression_weight = float(config["training"].get("regression_loss_weight", 0.50))

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(mode="horizontal"),
            tf.keras.layers.RandomTranslation(height_factor=0.02, width_factor=0.02),
            tf.keras.layers.RandomRotation(factor=0.01),
        ],
        name="augmentation",
    )

    x = augmentation(inputs)
    pretrained = bool(config["training"].get("pretrained_backbone", True))
    weights = "imagenet" if pretrained else None
    try:
        base = tf.keras.applications.EfficientNetB1(
            include_top=False,
            weights=weights,
            input_shape=(image_size, image_size, 3),
            pooling="avg",
        )
    except Exception:
        base = tf.keras.applications.EfficientNetB1(
            include_top=False,
            weights=None,
            input_shape=(image_size, image_size, 3),
            pooling="avg",
        )
    base.trainable = False

    x = base(x, training=False)
    x = tf.keras.layers.Dense(256, activation="swish", name="shared_dense")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    class_output = tf.keras.layers.Dense(len(PATTERN_CLASSES), activation="softmax", name="pattern")(x)
    reg_output = tf.keras.layers.Dense(3, activation="linear", name="levels")(x)

    model = tf.keras.Model(inputs=inputs, outputs={"pattern": class_output, "levels": reg_output}, name="brent_tf_efficientnet_b1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            "pattern": tf.keras.losses.SparseCategoricalCrossentropy(),
            "levels": tf.keras.losses.Huber(),
        },
        loss_weights={"pattern": 1.0, "levels": regression_weight},
        metrics={
            "pattern": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            "levels": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )
    return model, base



def unfreeze_tf_backbone(base_model, n_last_layers: int = 40) -> None:
    tf = _require_tf()
    base_model.trainable = True
    for layer in base_model.layers[:-n_last_layers]:
        layer.trainable = False
    for layer in base_model.layers[-n_last_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True



def train_tf_pipeline(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    feature_cols: Sequence[str],
    config: Dict[str, object],
    output_dir: str,
) -> Dict[str, object]:
    tf = _require_tf()
    os.makedirs(output_dir, exist_ok=True)

    model, base = build_tf_model(config)
    train_ds, train_table = build_tf_dataset(frame, table, feature_cols, config, split="train", shuffle=True)
    valid_ds, valid_table = build_tf_dataset(frame, table, feature_cols, config, split="valid", shuffle=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pattern_accuracy",
            patience=int(config["training"].get("patience", 5)),
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_pattern_accuracy",
            factor=0.5,
            patience=max(1, int(config["training"].get("patience", 5)) // 2),
            mode="max",
            min_lr=1e-6,
        ),
    ]

    histories: Dict[str, List[float]] = {}

    if bool(config["dataset"].get("use_synthetic_pretrain", True)):
        synth_samples = int(config["dataset"].get("synthetic_samples", 8000))
        synth_ds = build_tf_synthetic_dataset(config, n_samples=synth_samples)
        pre_hist = model.fit(synth_ds, epochs=max(1, min(5, int(config["training"].get("epochs", 15)) // 2)), verbose=1)
        histories["synthetic_pretrain_loss"] = [float(x) for x in pre_hist.history.get("loss", [])]

    warm_hist = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=int(config["training"].get("epochs", 15)),
        callbacks=callbacks,
        verbose=1,
    )

    unfreeze_tf_backbone(base, n_last_layers=int(config["training"].get("unfreeze_last_layers", 40)))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config["training"].get("fine_tune_learning_rate", 1e-4))),
        loss={
            "pattern": tf.keras.losses.SparseCategoricalCrossentropy(),
            "levels": tf.keras.losses.Huber(),
        },
        loss_weights={"pattern": 1.0, "levels": float(config["training"].get("regression_loss_weight", 0.50))},
        metrics={
            "pattern": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            "levels": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )

    fine_hist = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=int(config["training"].get("fine_tune_epochs", 10)),
        callbacks=callbacks,
        verbose=1,
    )

    model_path = os.path.join(output_dir, str(config["output"].get("tf_model_name", "brent_efficientnet_b1_tf.keras")))
    model.save(model_path)

    report = {
        "model_path": model_path,
        "train_windows": int(len(train_table)),
        "valid_windows": int(len(valid_table)),
        "warm_history": {key: [float(v) for v in values] for key, values in warm_hist.history.items()},
        "fine_tune_history": {key: [float(v) for v in values] for key, values in fine_hist.history.items()},
    }

    with open(os.path.join(output_dir, "tf_training_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report



def load_tf_model(model_path: str):
    tf = _require_tf()
    return tf.keras.models.load_model(model_path)



def predict_tf_model(
    model_path: str,
    frame: pd.DataFrame,
    infer_table: pd.DataFrame,
    feature_cols: Sequence[str],
    config: Dict[str, object],
    batch_size: int | None = None,
) -> pd.DataFrame:
    model = load_tf_model(model_path)
    image_size = int(config["dataset"]["image_size"])
    target_col = str(config["data"].get("target_col", "brent_close"))
    batch_size = int(batch_size or config["training"].get("batch_size", 16))

    results: List[Dict[str, float]] = []
    batch_images: List[np.ndarray] = []
    batch_rows: List[pd.Series] = []

    def _flush() -> None:
        nonlocal batch_images, batch_rows, results
        if not batch_images:
            return
        images = np.stack(batch_images, axis=0).astype(np.float32)
        preds = model.predict(images, verbose=0)
        probs = preds["pattern"]
        levels = preds["levels"]
        for row, prob_row, level_row in zip(batch_rows, probs, levels):
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
        batch_images.append(image.astype(np.float32))
        batch_rows.append(row)
        if len(batch_images) >= batch_size:
            _flush()
    _flush()

    return pd.DataFrame(results)