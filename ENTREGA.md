# Entrega

Este repositorio reporta dos resultados porque cumplen roles distintos.

## 1. Modelo principal reproducible

- Submit: `Submits/hardcode_rank_neo60_progressive_down_24.csv`
- R2 publico: `0.39740`
- Rol: modelo final principal.
- Descripcion: pipeline entrenable con feature engineering, CatBoost, ExtraTrees,
  ensamble por ranking y postprocesamiento deterministico por subgenero.

Este es el resultado mas defendible como modelo predictivo, porque se puede
reproducir desde las bases originales con:

```powershell
python 01_entrenar_y_generar_modelo_final.py
```

## 2. Mejor resultado de competencia

- Submit: `Submits/lb_probe_v9_residual_refined_primary_t3000_a075.csv`
- R2 publico: `0.46155`
- Rol: mejor performance obtenida en Kaggle.
- Descripcion: ensamble transductivo calibrado con feedback del leaderboard
  publico.

Este resultado se informa como optimizacion competitiva adicional. Tiene mejor
R2 publico, pero no se presenta como modelo principal porque usa informacion
del leaderboard publico y por eso tiene mayor riesgo de sobreajuste a ese
subconjunto.

Para reconstruirlo:

```powershell
python 02_reproducir_mejor_leaderboard.py
```

## Archivos clave

- `README.md`: instrucciones de instalacion y reproduccion.
- `codigo/`: scripts del pipeline y del solver.
- `Submits/README.md`: explica que submits son finales y cuales son historicos.
- `otros csv/lb_probe_solver/report.md`: diagnostico del solver de leaderboard.
- `docs/REGISTRO_TP.md`: bitacora completa del proceso experimental.
- `presentacion/`: guion, deck y assets visuales para presentar el trabajo.
