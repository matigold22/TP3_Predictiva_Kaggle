# Reconstruccion del mejor submit competitivo

Esta carpeta reconstruye el mejor resultado publico obtenido en Kaggle.

## Resultado

- Submit: `submit_competencia_R2_0.46155.csv`
- R2 publico: `0.46155`
- Metodo: ensamble transductivo calibrado con feedback del leaderboard publico.

## Como correr

Desde esta carpeta:

```powershell
python -m jupyter execute reconstruir_submit_final.ipynb
```

El notebook usa:

- submits historicos en `../Submits/`;
- scores publicos en `../otros csv/lb_probe_solver/leaderboard_results.csv`;
- features auxiliares en `../otros csv/`.

## Aclaracion

Este resultado se reporta como mejor performance competitiva. No reemplaza la capa de modelo reproducible: la complementa.
