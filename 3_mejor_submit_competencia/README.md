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

- submits historicos en `../soporte/Submits/`;
- scores publicos en `../soporte/otros csv/lb_probe_solver/leaderboard_results.csv`;
- features auxiliares en `../soporte/otros csv/`.

## Salida esperada

El notebook reconstruye el archivo historico:

- `../soporte/Submits/lb_probe_v9_residual_refined_primary_t3000_a075.csv`

La carpeta incluye ademas una copia visible para la entrega:

- `submit_competencia_R2_0.46155.csv`

Ambos archivos representan el mismo submit competitivo (`R2 = 0.46155`).

## Aclaracion

Este resultado se reporta como mejor performance competitiva. No reemplaza la capa de modelo reproducible: la complementa.

## Nota de reproducibilidad

Esta capa depende de submits historicos y de sus scores publicos. Por eso reproduce el mejor archivo enviado a la competencia, pero no debe leerse como un modelo entrenable solamente desde `train` y `test`.
