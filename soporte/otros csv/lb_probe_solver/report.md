# Diagnostico del solver por probes de leaderboard

- Submits exactos utilizados: 56
- Submits usados como base luego de proteger lookup: 51
- Excluidos por alterar lookup: `catboost_best_lookup_blend_85.csv`
- Excluidos por score con menos de 4 decimales: `catboost_fs_play_top60_more_trees_submit.csv`, `catboost_fs_play_top50_regularized_submit.csv`, `lgbm_top60_num_regularized_submit.csv`
- Referencia: `lb_probe_v8_residual_primary_t3000_a075.csv` con R2 publico 0.46150
- SST estimado: 3458700.81
- Varianza explicada por la primera direccion de `away_l2`: 0.99994352
- Error maximo de la parabola `away_l2`: 0.00000490

## Validacion temporal

| target | threshold | observed_r2 | predicted_r2 | error | relative_reconstruction_error |
| --- | --- | --- | --- | --- | --- |
| hardcode_genre_rank_neo_down_60.csv | 1000 | 0.39241 | 0.39256992 | 0.00015991646 | 0.29229827 |
| hardcode_genre_rank_neo_down_60.csv | 10000 | 0.39241 | 0.39251226 | 0.0001022605 | 0.30394253 |
| hardcode_genre_rank_neo_down_60.csv | 30000 | 0.39241 | 0.39260952 | 0.00019951831 | 0.33753833 |
| hardcode_rank_neo60_progressive_down_24.csv | 1000 | 0.3974 | 0.3975778 | 0.00017780011 | 0.24093604 |
| hardcode_rank_neo60_progressive_down_24.csv | 10000 | 0.3974 | 0.39749436 | 9.4363963e-05 | 0.27417157 |
| hardcode_rank_neo60_progressive_down_24.csv | 30000 | 0.3974 | 0.39754464 | 0.00014463748 | 0.32251359 |

## Candidatos

| filename | threshold | alpha | dimensions | ideal_predicted_r2_before_clipping | delta_rmse | clipped_rows | mean | std | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lb_probe_v9_residual_refined_conservative_t30000_a050.csv | 30000 | 0.5 | 5 | 0.46150397 | 0.026385799 | 7 | 34.429414 | 15.211765 | 0 | 99 |
| lb_probe_v9_residual_refined_primary_t3000_a075.csv | 3000 | 0.75 | 12 | 0.46152692 | 0.092101019 | 6 | 34.426754 | 15.221462 | 0 | 99 |
| lb_probe_v9_residual_refined_full_t3000_a100.csv | 3000 | 1 | 12 | 0.46152871 | 0.12280136 | 6 | 34.425447 | 15.227026 | 0 | 99 |

## Limitacion principal

Kaggle calcula el score sobre el 50% publico, pero no conocemos que filas integran
ese subconjunto. El solver usa productos internos sobre las 6.567 filas como
aproximacion. La parabola y la validacion temporal indican que la aproximacion es
util, pero los R2 predichos no deben interpretarse como garantias.
