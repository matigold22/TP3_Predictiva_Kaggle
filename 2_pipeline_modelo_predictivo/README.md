# Modelo reproducible

Esta carpeta contiene el pipeline principal de entrega.

## Resultado

- Submit: `submission_modelo_reproducible_R2_0.39740.csv`
- R2 publico: `0.39740`
- Modelo: CatBoost + ExtraTrees + lookup + postprocesamiento deterministico por subgenero.

Esta es la capa de modelo predictivo reproducible: se entrena desde las bases originales y no usa feedback del leaderboard para decidir el resultado final.

## Como correr

Desde esta carpeta:

```powershell
python -m jupyter execute b_c_entrenar_y_generar_modelo_final.ipynb
```

El notebook usa las bases en `../datos_raw/` y los notebooks auxiliares en `../soporte/codigo/`.

Antes de correrlo, verificar que existan:

- `../datos_raw/base_train.csv`
- `../datos_raw/base_val.csv`

Las bases no estan versionadas en GitHub porque el push fue bloqueado por GitHub Push Protection al detectar un patron sensible dentro de los CSV. Deben agregarse localmente desde los datos originales de la competencia.

## Que hace

1. Genera features.
2. Entrena CatBoost base.
3. Entrena CatBoost top 60.
4. Genera postprocesos sobre filas no vistas.
5. Entrena ExtraTrees como fuente de ranking complementaria.
6. Aplica ajustes por subgenero.
7. Genera `../soporte/Submits/hardcode_rank_neo60_progressive_down_24.csv`.

## Salida esperada

Al finalizar, el notebook debe dejar generado:

- `../soporte/Submits/hardcode_rank_neo60_progressive_down_24.csv`

La carpeta incluye ademas una copia visible para la entrega:

- `submission_modelo_reproducible_R2_0.39740.csv`

Ambos archivos representan el mismo resultado publico (`R2 = 0.39740`).

## Nota de reproducibilidad

El pipeline ejecuta notebooks auxiliares porque el entrenamiento completo esta compuesto por varios pasos. La carpeta `soporte/` conserva esos pasos para auditoria, pero el punto de entrada de la entrega es este notebook.
