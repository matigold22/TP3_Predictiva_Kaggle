# Modelo reproducible

Esta carpeta contiene el pipeline principal de entrega.

## Resultado

- Submit: `submission_modelo_reproducible_R2_0.39740.csv`
- R2 publico: `0.39740`
- Modelo: CatBoost + ExtraTrees + lookup + postprocesamiento deterministico por subgenero.

## Como correr

Desde esta carpeta:

```powershell
python -m jupyter execute b_c_entrenar_y_generar_modelo_final.ipynb
```

El notebook usa las bases en `../datos_raw/` y los notebooks auxiliares en `../codigo/`.

## Que hace

1. Genera features.
2. Entrena CatBoost base.
3. Entrena CatBoost top 60.
4. Genera postprocesos sobre filas no vistas.
5. Entrena ExtraTrees como fuente de ranking complementaria.
6. Aplica ajustes por subgenero.
7. Genera `Submits/hardcode_rank_neo60_progressive_down_24.csv`.
