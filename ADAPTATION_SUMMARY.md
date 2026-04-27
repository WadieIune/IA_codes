# Resumen de adaptación al bundle de `SeriesDownloader_2_features.py`

## Validación del bundle

- Reconstrucción exacta desde CSV: **1,233 ventanas válidas**.
- Fecha inicial reconstruida: **2019-12-02**.
- Fecha final reconstruida: **2026-03-04**.
- Coincidencia con `X_train/X_test` y `y_train/y_test`: **exacta**.

## Estructura del dataset

- `dataset_wide_with_target.csv`: **7004 x 182** al leerlo con columna fecha, equivalentes a **7004 x 181** columnas útiles más fecha.
- `dataset_wide_features_zscore.csv`: **7004 x 181** al leerlo con fecha, equivalentes a **7004 x 180** features más fecha.
- Universo de features:
  - 37 niveles raw,
  - 37 `logret1`,
  - 37 `vol20`,
  - 28 `rangepct`,
  - 37 `ma520`,
  - 4 spreads/ratios.

## Dataset supervisado usado por el sistema adaptado

Con `lookback=32`, `horizon=1` y `support_resistance_horizon=5`:

- ventanas supervisadas: **1,230**,
- `train`: **887**,
- `valid`: **99**,
- `test`: **244**,
- primera fecha útil: **2019-12-02**,
- última fecha útil: **2026-02-26**.

## Distribución de etiquetas débiles

- `ascending_channel`: **506**
- `descending_channel`: **418**
- `range`: **248**
- `double_top`: **37**
- `double_bottom`: **20**
- `inverse_head_shoulders`: **1**

## Observación relevante

La cobertura efectiva arranca en 2019 principalmente por la entrada tardía de **ESTR** y, en menor medida, por **SOFR**. El paquete permite excluir features tardías mediante `feature_regex_drop` para ampliar profundidad histórica si se desea.
