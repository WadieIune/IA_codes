# BRENT Pattern System adaptado a `SeriesDownloader_2_features.py`

Este paquete queda acoplado al formato real que genera tu descargador:

- `dataset_wide_with_target.csv`
- `dataset_wide_features_zscore.csv`
- `feature_means.csv`
- `feature_stds.csv`
- `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`

## Qué hace

1. **Respeta el bundle generado por `SeriesDownloader_2_features.py`** como fuente canónica.
2. Reconstruye las ventanas válidas y valida que coinciden con los `.npy`.
3. Etiqueta débilmente las ventanas de BRENT con patrones chartistas:
   - `double_bottom`
   - `double_top`
   - `ascending_channel`
   - `descending_channel`
   - `high_tight_flag`
   - `head_shoulders`
   - `inverse_head_shoulders`
   - `range`
4. Convierte cada ventana en imagen RGB para **EfficientNet-B1**:
   - canal 1: GASF de BRENT raw,
   - canal 2: GADF de retornos BRENT,
   - canal 3: mapa multivariante con los **180 features z-score** del bundle.
5. Entrena una cabeza multi-tarea para:
   - clasificación del patrón,
   - regresión de `future_return`, `future_low`, `future_high`.
6. Aplica **control de outliers** con rolling window, decay factor y distancia a plantillas métricas.

## Modos de entrada

### Modo recomendado: `series_bundle`

Usa directamente la salida del `SeriesDownloader_2_features.py`.

Config mínima:

```json
{
  "data": {
    "mode": "series_bundle",
    "target_col": "BRENT",
    "series_bundle": {
      "dataset_dir": "market_dataset",
      "bundle_target_col": "BRENT_fwd_logret_1",
      "bundle_price_col": "BRENT"
    }
  }
}
```

## Validación del bundle

```bash
python -m brent_pattern_system.validate_bundle --config brent_pattern_system/example_config.json
```

Esto comprueba que las ventanas reconstruidas desde el CSV z-score coinciden con `X_train/X_test`.

## Perfilado del dataset

```bash
python -m brent_pattern_system.profile_bundle --config brent_pattern_system/example_config.json
```

Genera:

- `bundle_profile.json`
- `weak_label_window_summary.csv`

## Entrenamiento PyTorch

```bash
python -m brent_pattern_system.train_torch --config brent_pattern_system/example_config.json
```

### Notas de robustez

- Si `torchvision` no está disponible o falla por incompatibilidad binaria, el código cae en:
  1. `timm` si está instalado,
  2. una CNN compacta de respaldo.
- Se guarda qué backend se usó para recargar el modelo sin ambigüedad.

## Entrenamiento TensorFlow

```bash
python -m brent_pattern_system.train_tf --config brent_pattern_system/example_config.json
```

### Notas de robustez

- Si EfficientNet-B1 preentrenada no puede descargarse, cae a `weights=None`.
- TensorFlow se importa de forma perezosa: el resto del paquete funciona aunque no esté instalado.

## Inferencia y outliers

```bash
python -m brent_pattern_system.inference --config brent_pattern_system/example_config.json --backend torch
python -m brent_pattern_system.inference --config brent_pattern_system/example_config.json --backend tensorflow
```

Salidas principales:

- `*_prediction_report.csv`
- `*_outlier_report.csv`
- `*_latest_signal.json`

Cada predicción incluye:

- patrón dominante,
- probabilidades por patrón,
- retorno esperado,
- soporte y resistencia estimados,
- entropía de clasificación,
- distancia métrica a la plantilla de clase,
- `outlier_flag`.

## Pipeline completo

```bash
python -m brent_pattern_system.main_pipeline --config brent_pattern_system/example_config.json --backend both --train --infer
```

## Decisiones metodológicas relevantes

### 1) Se usa BRENT raw para el chartismo

La detección de doble suelo, doble techo, canales o HCH se calcula sobre la serie de precio raw de BRENT, no sobre su z-score.

### 2) Se usa el bundle z-score para el canal multivariante

La tercera banda de la imagen aprovecha exactamente las features normalizadas que ya produce tu pipeline.

### 3) Se respeta el split original del descargador

Cuando el bundle incluye `X_train/X_test` y el `lookback` es 32 con `horizon=1`, se conserva el corte original de train/test y solo se extrae validación desde la cola de `train`.

### 4) Soporte y resistencia

`future_return` sigue el target del bundle (`BRENT_fwd_logret_1`) y `future_low/high` se calculan sobre una ventana futura configurable (`support_resistance_horizon`, por defecto 5).

## Punto importante sobre cobertura temporal

Con el bundle completo de 180 features, la historia efectiva de ventanas válidas empieza bastante más tarde que 2007 porque algunas series entran tarde, especialmente **ESTR** y **SOFR**. Si quieres ampliar historia útil sacrificando algunas variables tardías, puedes excluirlas por regex, por ejemplo:

```json
{
  "dataset": {
    "feature_regex_drop": ["^ESTR", "^SOFR"]
  }
}
```

## Dependencias orientativas

Ver `requirements.txt`.
