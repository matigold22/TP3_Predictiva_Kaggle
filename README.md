# TP Competencia Kaggle - Entrega final

Prediccion de popularidad de tracks de Spotify. Metrica: R2 publico.

La entrega separa dos capas, porque cumplen roles distintos:

| Capa | Que es | R2 publico | Reproducible desde |
| --- | --- | --- | --- |
| Modelo reproducible | CatBoost + ExtraTrees + feature engineering + lookup + postprocesamiento deterministico | 0.39740 | datos crudos |
| Mejor resultado competitivo | Ensamble transductivo calibrado con feedback del leaderboard publico | 0.46155 | submits historicos + scores publicos |

El modelo reproducible es la respuesta mas directa a la consigna como pipeline predictivo. El resultado competitivo se reporta aparte porque logra mejor score publico, pero usa informacion del leaderboard.

## Estructura

```text
TP3_Predictiva_Kaggle/
|-- README.md
|-- ENTREGA.md
|-- requirements.txt
|
|-- datos_raw/
|   |-- base_train.csv
|   `-- base_val.csv
|
|-- 1_presentacion/
|   |-- PRESENTACION_MODELOS_FINAL.pptx
|   `-- PRESENTACION_MODELOS_FINAL.pdf
|
|-- 2_pipeline_modelo_predictivo/
|   |-- b_c_entrenar_y_generar_modelo_final.ipynb
|   |-- submission_modelo_reproducible_R2_0.39740.csv
|   `-- README.md
|
|-- 3_mejor_submit_competencia/
|   |-- reconstruir_submit_final.ipynb
|   |-- submit_competencia_R2_0.46155.csv
|   `-- README.md
|
`-- soporte/
    |-- codigo/
    |   `-- notebooks auxiliares del pipeline
    |-- Submits/
    |   `-- submits historicos usados por el solver
    |-- otros csv/
    |   `-- features procesadas y archivos auxiliares
    |-- modelos/
    |   `-- metadatos de entrenamiento
    `-- docs/
        `-- REGISTRO_TP.md
```

La carpeta `soporte/` no es lo primero que hay que leer para corregir la entrega. Se conserva para trazabilidad: contiene notebooks intermedios, submits historicos, features procesadas, metadatos y la bitacora completa.

## Datos de entrada

Las bases originales deben ubicarse localmente en `datos_raw/` con estos nombres:

- `datos_raw/base_train.csv`
- `datos_raw/base_val.csv`

Estos archivos no se versionan en GitHub. Se dejaron excluidos del repositorio porque el push fue bloqueado por GitHub Push Protection al detectar un patron sensible dentro de los CSV. Por eso, para reproducir la entrega, se debe descargar o copiar las bases originales de la competencia en esa carpeta antes de ejecutar las notebooks.

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Como correr

Modelo reproducible:

```powershell
cd 2_pipeline_modelo_predictivo
python -m jupyter execute b_c_entrenar_y_generar_modelo_final.ipynb
```

Mejor resultado competitivo:

```powershell
cd 3_mejor_submit_competencia
python -m jupyter execute reconstruir_submit_final.ipynb
```

## Mapa a la consigna

- Presentacion: `1_presentacion/PRESENTACION_MODELOS_FINAL.pptx` y version PDF en `1_presentacion/PRESENTACION_MODELOS_FINAL.pdf`.
- Modelo baseline: explicado en la presentacion y generado dentro del pipeline reproducible.
- Seleccion de modelos: explicada en la presentacion y documentada en `soporte/docs/REGISTRO_TP.md`.
- Modelo final reproducible: `2_pipeline_modelo_predictivo/`.
- Mejor submit competitivo: `3_mejor_submit_competencia/`.
- Limitaciones y posibles mejoras: ultimas diapositivas de la presentacion.
