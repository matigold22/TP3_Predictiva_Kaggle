# Prediccion de popularidad musical

Repositorio de entrega con dos resultados diferenciados:

1. **Modelo final reproducible**: ensamble CatBoost + ExtraTrees con postprocesamiento deterministico por subgenero.
   - Submit: `Submits/hardcode_rank_neo60_progressive_down_24.csv`
   - R2 publico: `0.39740`
   - Rol en la entrega: modelo/pipeline principal.

2. **Mejor resultado de competencia**: ensamble transductivo calibrado con feedback del leaderboard publico.
   - Submit: `Submits/lb_probe_v9_residual_refined_primary_t3000_a075.csv`
   - R2 publico: `0.46155`
   - Rol en la entrega: experimento adicional de optimizacion de competencia, no modelo principal.

## Estructura

- `codigo/`: scripts usados para generar los dos resultados.
- `Submits/`: submits finales y submits historicos necesarios para reconstruir el solver del leaderboard.
- `otros csv/lb_probe_solver/`: scores publicos y resumen del solver.
- `modelos/`: metadatos necesarios para reproducir configuraciones.
- `bases de datos/`: colocar aca `base_train.csv` y `base_val.csv` antes de ejecutar desde cero.

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproducir el modelo final principal

Con `base_train.csv` y `base_val.csv` dentro de `bases de datos/`:

```powershell
python 01_entrenar_y_generar_modelo_final.py
```

Este comando ejecuta la cadena reproducible:

```text
feature engineering
-> CatBoost base
-> CatBoost top 60
-> postproceso away-l2
-> ExtraTrees
-> ensamble por ranking
-> ajuste neo soul
-> ajuste progressive electro house
-> Submits/hardcode_rank_neo60_progressive_down_24.csv
```

## Reproducir el mejor resultado de competencia

```powershell
python 02_reproducir_mejor_leaderboard.py
```

Este comando reconstruye el submit `0.46155` a partir de los submits historicos y los scores publicos registrados.

## Aclaracion metodologica

El submit `0.39740` es el modelo final principal porque puede explicarse como un pipeline predictivo entrenable y reproducible.

El submit `0.46155` se informa como mejor resultado experimental de competencia, pero usa feedback del leaderboard publico. Por eso tiene mayor riesgo de sobreajuste al conjunto publico y no se presenta como modelo predictivo principal.
