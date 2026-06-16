# Resumen de entrega

Este repositorio reporta dos resultados principales porque responden a objetivos distintos.

## 1. Modelo reproducible

- Carpeta: `2_modelo_reproducible/`
- Notebook: `b_c_entrenar_y_generar_modelo_final.ipynb`
- Submit: `submission_modelo_reproducible_R2_0.39740.csv`
- R2 publico: `0.39740`
- Descripcion: pipeline entrenable con feature engineering, CatBoost, ExtraTrees, ensamble por ranking y postprocesamiento deterministico por subgenero.

Este es el resultado mas defendible como modelo predictivo, porque se puede generar desde las bases originales.

## 2. Mejor resultado competitivo

- Carpeta: `3_reconstruccion_submit_final/`
- Notebook: `reconstruir_submit_final.ipynb`
- Submit: `submit_competencia_R2_0.46155.csv`
- R2 publico: `0.46155`
- Descripcion: ensamble transductivo calibrado con feedback del leaderboard publico.

Este resultado muestra la mejor performance obtenida en Kaggle. Se informa separado del modelo reproducible porque usa informacion del leaderboard publico.

## Archivos clave

- `1_presentacion/`: presentacion final en PPTX y PDF.
- `2_modelo_reproducible/`: reproduccion del modelo predictivo principal.
- `3_reconstruccion_submit_final/`: reconstruccion del mejor submit competitivo.
- `datos_raw/`: bases de entrada locales.
- `soporte/`: material historico y auxiliar.
- `soporte/docs/REGISTRO_TP.md`: bitacora completa del proceso experimental.

## Nota sobre los datos

Las bases `base_train.csv` y `base_val.csv` deben estar en `datos_raw/` para ejecutar las notebooks. No se suben al repositorio porque GitHub bloqueo el push al detectar un patron sensible dentro de los CSV. La carpeta queda indicada como punto de entrada para que, con los datos originales de la competencia, se pueda reproducir el modelo.
