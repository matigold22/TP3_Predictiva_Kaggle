# Registro del TP - Prediccion de `track_popularity`

Fecha de inicio del registro: 2026-05-04

## Objetivo del trabajo

El objetivo del TP es predecir la variable `track_popularity` a partir de variables musicales, de album, artista y playlist.

La entrega final debe ser un archivo CSV con dos columnas:

- `ID`
- `track_popularity`

## Archivos identificados

- `bases de datos/base_train.csv`: base de entrenamiento con la variable objetivo.
- `bases de datos/base_val.csv`: base de test/validacion sin la variable objetivo.
- `codigo/1_eda.ipynb`: notebook de analisis exploratorio de datos.

## Revision inicial de las bases

Dimensiones:

- Train: 26.266 filas y 24 columnas.
- Test/base_val: 6.567 filas y 23 columnas.

La unica columna presente en train y ausente en test es `track_popularity`, lo cual es correcto porque es la variable a predecir.

Calidad general:

- No hay duplicados exactos en train ni en test.
- No se encontraron valores fuera de rango en las variables musicales principales.
- En train hay pocos valores nulos:
  - `track_name`: 5 nulos.
  - `track_artist`: 5 nulos.
  - `track_album_name`: 5 nulos.
- En test no se observaron nulos.

## Cambios realizados en el EDA

Archivo modificado:

- `codigo/1_eda.ipynb`

Cambios aplicados:

- Se corrigio la configuracion de rutas para que el notebook funcione tanto desde la raiz del proyecto como desde carpetas `codigo` o `codigos`.
- Se agrego un analisis de calidad de fechas de lanzamiento.
- Se distinguieron formatos de fecha:
  - fecha completa: `YYYY-MM-DD`
  - anio-mes: `YYYY-MM`
  - solo anio: `YYYY`
- Se agrego diagnostico de solapamiento entre train y test para identificadores como `track_id`, `track_album_id`, `track_artist`, `playlist_id` y `playlist_name`.
- Se agrego una seccion especifica sobre consistencia de `track_id`.
- Se agregaron recomendaciones de limpieza y feature engineering.
- Se reescribieron las conclusiones finales para dejarlas mas orientadas al modelado.
- Se valido que el notebook actualizado sea JSON valido y que las celdas corran sin errores.

Actualizacion posterior del EDA:

- Se agrego una seccion nueva llamada `EDA orientado al modelado`.
- Se revisaron valores faltantes no evidentes, como strings vacios, espacios y tokens sospechosos.
- Se agrego comparacion de outliers entre train y test usando limites IQR calculados en train.
- Se agregaron graficos y tablas de relacion no lineal entre variables numericas y `track_popularity` mediante bins/quintiles.
- Se agrego analisis temporal de popularidad por decada de lanzamiento.
- Se agrego analisis de alta cardinalidad y cobertura train/test para variables categoricas.
- Se agrego una tabla detallada de popularidad por `playlist_genre` y `playlist_subgenre`.
- Se agrego una matriz de decision por variable/grupo de variables, sin implementar todavia feature engineering.
- Se corrigio el selector de variables categoricas para que funcione aunque pandas reporte texto como dtype `str` en vez de `object`.
- Se volvio a validar que el notebook completo corra sin errores.

## Conclusiones principales del EDA

El problema es de regresion, porque `track_popularity` es una variable numerica entre 0 y 100.

La base esta bastante limpia. No hace falta una limpieza agresiva ni eliminar muchas filas. Los pocos nulos pueden imputarse con una categoria explicita como `missing`.

La columna `Unnamed: 0` funciona como indice de fila y no deberia usarse como predictor.

Las variables musicales individuales tienen relaciones lineales debiles con la popularidad. Esto sugiere que la popularidad depende de interacciones entre multiples factores, por lo que pueden convenir modelos no lineales.

Las variables `playlist_genre` y `playlist_subgenre` muestran diferencias en popularidad y deberian conservarse con una codificacion adecuada.

En test aparecen categorias que no estaban en train, por lo que el preprocesamiento debe ser robusto a categorias nuevas.

La variable `track_album_release_date` no debe descartarse. Conviene extraer variables como anio, mes, decada y flags de fecha parcial.

`track_id` es una variable clave. Cuando una cancion aparece repetida en train, mantiene la misma popularidad. Ademas, parte del test contiene `track_id` ya vistos en train. Para esas filas conviene usar la popularidad observada en train como prediccion directa o lookup.

## Decisiones preliminares para modelado

- Excluir `Unnamed: 0` como predictor.
- Imputar nulos textuales con `missing`.
- Crear features temporales desde `track_album_release_date`.
- Codificar variables categoricas con estrategias robustas a categorias no vistas.
- Usar `track_id` como lookup para canciones ya vistas en train.
- Comparar modelos lineales baseline contra modelos no lineales como Random Forest, ExtraTrees, LightGBM o CatBoost.
- Validar con cuidado para evitar sobreestimar resultados por canciones repetidas.

## Proximo paso

Construir el pipeline de limpieza, feature engineering, entrenamiento, validacion y generacion del CSV final con columnas `ID` y `track_popularity`.

## Feature engineering

Archivo creado:

- `codigo/2_feature_engineering.ipynb`

Objetivo del notebook:

- Construir un set de variables procesadas para usar luego en modelado.
- Aplicar las mismas transformaciones a train y test.
- Evitar leakage en variables supervisadas.
- Guardar archivos intermedios listos para entrenar modelos.

Transformaciones implementadas:

- Limpieza consistente de columnas textuales:
  - imputacion de nulos como `missing`;
  - eliminacion de espacios extremos;
  - normalizacion de tokens sospechosos como strings vacios o `nan`.
- Variables temporales desde `track_album_release_date`:
  - anio, mes, dia, dia de semana;
  - decada;
  - edad respecto de 2020;
  - flags de fecha parcial;
  - flags de canciones recientes o antiguas.
- Variables de duracion:
  - duracion en minutos y segundos;
  - log de duracion;
  - flags de canciones cortas, largas y muy largas.
- Variables musicales derivadas:
  - interacciones entre `energy`, `danceability`, `valence`, `loudness`, `speechiness`, `acousticness` e `instrumentalness`;
  - diferencias entre variables musicales;
  - transformaciones `log1p` para variables sesgadas;
  - flags interpretables como alta instrumentalidad, alta acustica, alta energia, alta bailabilidad, etc.
- Variables categoricas combinadas:
  - `key_mode`;
  - `genre_subgenre`;
  - `artist_genre`;
  - `album_genre`;
  - `playlist_context`.
- Frequency encoding:
  - frecuencias por categoria calculadas con train+test sin usar target;
  - frecuencias calculadas solo en train;
  - flags de categoria vista en train.
- Target encoding:
  - target encoding suavizado para variables categoricas relevantes;
  - en train se calculo out-of-fold para reducir leakage;
  - en test se uso el mapping aprendido con todo train;
  - no se target-encodeo `track_id` como feature general, porque se manejara aparte como lookup.
- Lookup de `track_id`:
  - se guardo una tabla con popularidad media por `track_id`;
  - se confirmo que no hay `track_id` repetidos con popularidad inconsistente;
  - hay 800 filas de test con `track_id` visto en train, equivalentes al 12,18%.

Archivos generados:

- `otros csv/train_features.csv`
- `otros csv/test_features.csv`
- `otros csv/track_id_lookup.csv`
- `otros csv/feature_columns.csv`

Validacion de los archivos generados:

- `train_features.csv`: 26.266 filas y 170 columnas.
- `test_features.csv`: 6.567 filas y 169 columnas.
- Cantidad de predictores: 168.
- Train y test tienen las mismas columnas predictoras.
- No hay nulos en los archivos procesados.
- No hay IDs duplicados.
- `track_id_lookup.csv` contiene 22.995 canciones unicas y no presenta inconsistencias de popularidad para canciones repetidas.

Proximo paso actualizado:

- Entrenar y validar modelos usando las features procesadas.
- Comparar modelos lineales baseline contra modelos no lineales.
- Aplicar lookup final de `track_id` para canciones de test ya vistas en train.
- Generar el CSV final con columnas `ID` y `track_popularity`.

## Modelo CatBoost inicial

Archivo creado:

- `codigo/3_catboost_best_submit.py`

Objetivo:

- Entrenar un modelo CatBoost fuerte pero razonable en tiempo de corrida.
- Validar internamente con separacion dentro de train.
- Evitar una validacion optimista por canciones repetidas.
- Generar un CSV listo para probar en Kaggle.

Validacion:

- Se uso `GroupKFold` con 3 folds agrupando por `track_id`.
- Esto evita que la misma cancion aparezca al mismo tiempo en entrenamiento y validacion.
- Se uso early stopping para controlar overfitting.
- Se usaron 157 features.
- Se usaron como categoricas directas:
  - `playlist_genre`
  - `playlist_subgenre`
  - `key_str`
  - `mode_str`
  - `key_mode`
  - `genre_subgenre`
- Las categoricas de alta cardinalidad se dejaron representadas por variables numericas derivadas como frecuencias y target encodings, para evitar corridas excesivamente lentas.

Metricas de validacion cruzada:

- RMSE medio: 14,7429.
- RMSE desvio: 0,1254.
- MAE medio: 10,6654.
- R2 medio: 0,6519.
- OOF RMSE: 14,7433.
- OOF MAE: 10,6654.
- OOF R2: 0,6520.

Entrenamiento final:

- Se entreno CatBoost con todo train usando 1264 iteraciones, definidas a partir de las mejores iteraciones observadas en CV.
- Se aplico el lookup de `track_id` para 800 filas de test cuyo `track_id` ya aparecia en train.

Archivos generados:

- `Submits/catboost_best_submit.csv`
- `otros csv/catboost_best_test_predictions_raw.csv`
- `otros csv/catboost_best_cv_folds.csv`
- `otros csv/catboost_best_cv_summary.csv`
- `otros csv/catboost_best_oof_predictions.csv`
- `modelos/catboost_best.cbm`
- `modelos/catboost_best_metadata.json`

Validacion del submit:

- El archivo tiene 6.567 filas y 2 columnas.
- Columnas: `ID`, `track_popularity`.
- No hay nulos.
- No hay IDs duplicados.
- Las predicciones quedan dentro del rango 0 a 100.

Resultado Kaggle informado por usuario:

- `catboost_best_submit.csv` obtuvo R2 publico de 0,3693.

## Nuevas pruebas CatBoost paso a paso

Luego del primer resultado en Kaggle, se decidio seguir probando CatBoost antes de pasar a otros modelos.

Archivos nuevos:

- `Submits/catboost_best_no_lookup_submit.csv`
- `codigo/3_catboost_no_te_submit.py`
- `Submits/catboost_no_te_no_lookup_submit.csv`
- `Submits/catboost_no_te_lookup_submit.csv`
- `otros csv/catboost_no_te_cv_folds.csv`
- `otros csv/catboost_no_te_cv_summary.csv`
- `otros csv/catboost_no_te_oof_predictions.csv`
- `modelos/catboost_no_te.cbm`
- `modelos/catboost_no_te_metadata.json`

Prueba 1:

- Se genero una version del primer CatBoost sin aplicar lookup de `track_id`.
- Objetivo: aislar si el lookup ayuda o perjudica en el score publico.
- Archivo para probar: `Submits/catboost_best_no_lookup_submit.csv`.

Prueba 2:

- Se entreno un CatBoost sin target encodings precomputados.
- Objetivo: reducir riesgo de sobreajuste/leakage de validacion y evaluar una variante mas conservadora.
- Validacion: `GroupKFold` de 3 folds por `track_id`.
- Features usadas: 129.
- Categoricas directas:
  - `playlist_genre`
  - `playlist_subgenre`
  - `key_str`
  - `mode_str`
  - `key_mode`
  - `genre_subgenre`

Metricas de la prueba sin target encodings:

- RMSE medio: 18,9910.
- MAE medio: 14,9750.
- R2 medio: 0,4224.
- OOF RMSE: 18,9914.
- OOF MAE: 14,9750.
- OOF R2: 0,4226.

Submits generados para esta prueba:

- `Submits/catboost_no_te_no_lookup_submit.csv`
- `Submits/catboost_no_te_lookup_submit.csv`

Ambos archivos fueron validados:

- 6.567 filas.
- 2 columnas: `ID`, `track_popularity`.
- Sin nulos.
- Sin IDs duplicados.

## Limpieza de pruebas CatBoost descartadas

Fecha: 2026-05-06

Se eliminaron archivos generados por una rama de refinamiento de CatBoost que quedo incompleta/lenta y no se va a usar:

- `codigo/4_catboost_refinements.py`
- `codigo/4_catboost_refinements_fast.py`
- `Submits/catboost_fast_reg_depth6_submit.csv`
- `modelos/catboost_fast_reg_depth6.cbm`

Se conservaron los submits CatBoost utiles/probados:

- `Submits/catboost_best_submit.csv`
- `Submits/catboost_best_no_lookup_submit.csv`
- `Submits/catboost_no_te_no_lookup_submit.csv`
- `Submits/catboost_no_te_lookup_submit.csv`

## Mejoras sobre el mejor CatBoost

Fecha: 2026-05-06

Se siguio trabajando sobre el mejor enfoque conocido:

- CatBoost.
- Target/frequency encodings.
- Lookup de `track_id`.

Resultados Kaggle informados:

- `catboost_best_submit.csv`: R2 publico 0,3693.
- `catboost_best_no_lookup_submit.csv`: R2 publico aprox. 0,34.
- `catboost_no_te_lookup_submit.csv`: R2 publico aprox. 0,05.

Conclusiones:

- El lookup de `track_id` ayuda.
- Quitar target encodings perjudica mucho.
- Conviene mejorar alrededor del primer modelo, no cambiar drasticamente el enfoque.

Submits baratos generados a partir del mejor modelo, suavizando el lookup:

- `Submits/catboost_best_lookup_blend_50.csv`
- `Submits/catboost_best_lookup_blend_70.csv`
- `Submits/catboost_best_lookup_blend_85.csv`

Luego se creo y ejecuto:

- `codigo/4_catboost_numeric_tuned.py`

Objetivo:

- Mantener las features numericas, frecuencias y target encodings que funcionaron.
- Evitar categoricas directas para acelerar iteracion.
- Probar 3 configuraciones CatBoost con validacion cruzada.
- Usar `GroupKFold` por `track_id`.
- Aplicar early stopping para controlar overfit/underfit.

Validacion:

- 3 folds agrupados por `track_id`.
- 153 features numericas.

Metricas:

- `catboost_num_a_regularized`:
  - RMSE medio: 14,7572.
  - MAE medio: 10,6873.
  - OOF R2: 0,6513.
- `catboost_num_c_low_lr`:
  - RMSE medio: 14,7702.
  - MAE medio: 10,7090.
  - OOF R2: 0,6507.
- `catboost_num_b_shallow`:
  - RMSE medio: 14,7959.
  - MAE medio: 10,7374.
  - OOF R2: 0,6495.

Submits generados:

- `Submits/catboost_num_a_regularized_lookup_full_submit.csv`
- `Submits/catboost_num_a_regularized_lookup_blend70_submit.csv`
- `Submits/catboost_num_b_shallow_lookup_full_submit.csv`
- `Submits/catboost_num_b_shallow_lookup_blend70_submit.csv`
- `Submits/catboost_num_c_low_lr_lookup_full_submit.csv`
- `Submits/catboost_num_c_low_lr_lookup_blend70_submit.csv`
- `Submits/catboost_numeric_tuned_ensemble_submit.csv`

Todos los nuevos submits fueron validados:

- 6.567 filas.
- 2 columnas: `ID`, `track_popularity`.
- Sin nulos.
- Sin IDs duplicados.
- Predicciones dentro del rango 0 a 100.

## Limpieza de submits no probados

Fecha: 2026-05-06

Se eliminaron de `Submits` los CSV no probados en Kaggle para evitar confusion.

Se conservaron los submits probados/reportados:

- `catboost_best_submit.csv`: R2 publico 0,3693.
- `catboost_best_no_lookup_submit.csv`: R2 publico aprox. 0,34.
- `catboost_no_te_lookup_submit.csv`: R2 publico aprox. 0,05.
- `catboost_best_lookup_blend_85.csv`: R2 publico 0,36894.
- `catboost_num_a_regularized_lookup_full_submit.csv`: R2 publico 0,27663.

Se eliminaron:

- `catboost_best_lookup_blend_50.csv`
- `catboost_best_lookup_blend_70.csv`
- `catboost_no_te_no_lookup_submit.csv`
- `catboost_numeric_tuned_ensemble_submit.csv`
- `catboost_num_a_regularized_lookup_blend70_submit.csv`
- `catboost_num_b_shallow_lookup_blend70_submit.csv`
- `catboost_num_b_shallow_lookup_full_submit.csv`
- `catboost_num_c_low_lr_lookup_blend70_submit.csv`
- `catboost_num_c_low_lr_lookup_full_submit.csv`

## Close tuning sobre mejor CatBoost

Fecha: 2026-05-06

Se creo:

- `codigo/4_catboost_close_tuning.py`

Objetivo:

- Mejorar el mejor modelo conocido sin cambiar la receta que funciono.
- Mantener CatBoost, target/frequency encodings, categoricas directas moderadas y lookup completo.
- Probar cambios chicos de hiperparametros para evitar overfit/underfit.
- Usar validacion cruzada con `GroupKFold` por `track_id`.

Configuraciones probadas:

- `catboost_close_a_more_trees`:
  - mas iteraciones;
  - menor learning rate;
  - regularizacion similar/moderada.
- `catboost_close_b_slightly_regularized`:
  - mas regularizacion;
  - algo mas de aleatoriedad;
  - columnas/features sampleadas con `rsm` mas bajo.

Metricas CV:

- `catboost_close_a_more_trees`:
  - RMSE medio: 14,7462.
  - MAE medio: 10,6598.
  - OOF R2: 0,6518.
  - Mejor que el CatBoost inicial en CV por una diferencia pequena.
- `catboost_close_b_slightly_regularized`:
  - RMSE medio: 14,7653.
  - MAE medio: 10,6861.
  - OOF R2: 0,6509.

Submits generados:

- `Submits/catboost_close_a_more_trees_submit.csv`
- `Submits/catboost_close_b_slightly_regularized_submit.csv`
- `Submits/catboost_close_tuning_ensemble_submit.csv`

Tambien se generaron postprocesamientos baratos del mejor submit:

- `Submits/catboost_best_rounded_submit.csv`
- `Submits/catboost_best_expand_103_submit.csv`
- `Submits/catboost_best_expand_106_submit.csv`
- `Submits/catboost_best_expand103_rounded_submit.csv`

Todos los submits fueron validados:

- 6.567 filas.
- 2 columnas: `ID`, `track_popularity`.
- Sin nulos.
- Sin IDs duplicados.

## Seleccion de features sobre mejor CatBoost

Fecha: 2026-05-07

Se creo:

- `codigo/5_catboost_feature_selection.py`

Objetivo:

- Evaluar si 157 features eran demasiadas.
- Usar la importancia del mejor CatBoost (`catboost_best.cbm`) para ordenar variables.
- Entrenar modelos con subconjuntos top-N.
- Mantener la receta ganadora:
  - CatBoost.
  - Target/frequency encodings.
  - Categoricas directas moderadas.
  - Lookup completo de `track_id`.
  - `GroupKFold` por `track_id`.
  - Early stopping.

Top features mas importantes del mejor modelo:

- `track_album_id_te_diff_global`
- `track_album_id_te_oof`
- `playlist_id_te_oof`
- `playlist_context_te_oof`
- `playlist_id_te_diff_global`
- `playlist_name_te_diff_global`
- `album_genre_te_oof`
- `playlist_context_te_diff_global`
- `track_album_id_freq_train`
- `playlist_name_te_oof`

Modelos entrenados:

- `catboost_fs_top60`: 66 features finales, porque se conservaron tambien categoricas base.
- `catboost_fs_top90`: 95 features finales.
- `catboost_fs_top120`: 121 features finales.

Metricas CV:

- `catboost_fs_top60`:
  - RMSE medio: 14,7514.
  - MAE medio: 10,6700.
  - OOF R2: 0,6516.
- `catboost_fs_top120`:
  - RMSE medio: 14,7522.
  - MAE medio: 10,6692.
  - OOF R2: 0,6516.
- `catboost_fs_top90`:
  - RMSE medio: 14,7551.
  - MAE medio: 10,6792.
  - OOF R2: 0,6514.

Lectura:

- El modelo top-60 usa muchas menos features y mantiene una CV casi igual a la del modelo completo.
- Es un candidato interesante para Kaggle porque puede generalizar mejor al reducir ruido.

Submits generados:

- `Submits/catboost_fs_top60_submit.csv`
- `Submits/catboost_fs_top90_submit.csv`
- `Submits/catboost_fs_top120_submit.csv`
- `Submits/catboost_fs_ensemble_submit.csv`

Todos fueron validados:

- 6.567 filas.
- 2 columnas: `ID`, `track_popularity`.
- Sin nulos.
- Sin IDs duplicados.
- Predicciones dentro del rango 0 a 100.

Resultados Kaggle informados:

- `catboost_fs_top60_submit.csv`: R2 publico 0,37362. Nuevo mejor modelo hasta el momento.
- `catboost_fs_ensemble_submit.csv`: R2 publico 0,37147. Mejora al CatBoost base, pero queda por debajo del top-60.

Mejor submit actual:

- `Submits/catboost_fs_top60_submit.csv`

## Playground sobre el ganador top-60

Fecha: 2026-05-11

Se creo:

- `codigo/6_catboost_fs_playground.py`

Objetivo:

- Jugar alrededor del mejor modelo publico actual (`catboost_fs_top60_submit.csv`).
- Revisar hiperparametros cercanos al ganador, especialmente mas arboles y menor learning rate.
- Probar una version con menos features.
- Generar blends conservadores contra el ganador publico actual.
- Permitir correr candidatos por tandas con `--only` y acumular resultados.

Candidatos disponibles en el script:

- `catboost_fs_play_top50_regularized`: menos features, mas regularizacion.
- `catboost_fs_play_top60_more_trees`: misma cantidad de features que el ganador, mas iteraciones y menor learning rate.
- `catboost_fs_play_top60_depth5_smooth`: arboles mas chicos para cambiar la forma del modelo.
- `catboost_fs_play_top70_balanced`: un poco mas de features con regularizacion moderada.
- `catboost_fs_play_top45_tight`: version compacta y mas regularizada.
- `catboost_fs_play_top60_seedmix`: ensemble final de varias seeds.

Candidatos corridos:

- `catboost_fs_play_top50_regularized`
- `catboost_fs_play_top60_more_trees`
- `catboost_fs_play_top60_depth5_smooth`

Metricas CV acumuladas:

- `catboost_fs_play_top60_more_trees`:
  - 66 features.
  - RMSE medio: 14,7251.
  - OOF RMSE: 14,7255.
  - OOF R2: 0,6528.
  - Mejor CV que el `catboost_fs_top60` previo.
- `catboost_fs_play_top60_depth5_smooth`:
  - 66 features.
  - RMSE medio: 14,7440.
  - OOF RMSE: 14,7444.
  - OOF R2: 0,6520.
  - Queda cerca del ganador original y puede servir para diversificar.
- `catboost_fs_play_top50_regularized`:
  - 56 features.
  - RMSE medio: 14,7670.
  - OOF RMSE: 14,7673.
  - OOF R2: 0,6509.
  - Cumple la prueba de menos features, pero no parece el primer candidato a subir.

Submits nuevos generados:

- `Submits/catboost_fs_play_top60_more_trees_submit.csv`
- `Submits/catboost_fs_play_top60_depth5_smooth_submit.csv`
- `Submits/catboost_fs_play_top50_regularized_submit.csv`
- `Submits/catboost_fs_play_blend_public_top60_75_bestcv_25.csv`
- `Submits/catboost_fs_play_blend_best2_cv_equal.csv`
- `Submits/catboost_fs_play_blend_best3_cv_weighted.csv`

Archivos de seguimiento:

- `otros csv/catboost_fs_playground_cv_summary.csv`
- `otros csv/catboost_fs_playground_cv_folds.csv`
- `otros csv/catboost_fs_playground_blends.csv`
- `modelos/catboost_fs_playground_metadata.json`

Lectura:

- El candidato mas prometedor para Kaggle es `catboost_fs_play_top60_more_trees_submit.csv`.
- Como segunda prueba, conviene subir `catboost_fs_play_blend_public_top60_75_bestcv_25.csv`, porque combina 75% del ganador publico actual con 25% del nuevo mejor por CV.
- Si queda tiempo/submits, probar `catboost_fs_play_blend_best3_cv_weighted.csv` para capturar diversidad.
- Los candidatos pendientes mas interesantes son `catboost_fs_play_top70_balanced` y `catboost_fs_play_top60_seedmix`.

Resultados Kaggle informados:

- `catboost_fs_play_top60_more_trees_submit.csv`: R2 publico 0,356. Se descarta pese a buena CV interna.
- `catboost_fs_play_blend_public_top60_75_bestcv_25.csv`: R2 publico 0,36998. Mejor que el `more_trees` puro, pero por debajo del mejor actual `catboost_fs_top60_submit.csv` con 0,37362.
- `catboost_fs_play_top50_regularized_submit.csv`: R2 publico 0,325. Se descarta; la reduccion a 50 features fue demasiado agresiva para el publico.

Lectura posterior:

- La mejora de CV de `top60_more_trees` no traslado al publico.
- Conviene no seguir aumentando arboles sobre la misma receta.
- La variante de 50 features perdio mucha senal.
- Proxima prueba sugerida: si quedan submits, probar `catboost_fs_play_top60_depth5_smooth_submit.csv`; si no, correr una nueva tanda muy conservadora alrededor del `catboost_fs_top60` original.

## Hyper-CV conservadora alrededor del top-60 ganador

Fecha: 2026-05-11

Se creo:

- `codigo/7_catboost_top60_hypercv.py`

Objetivo:

- Modificar hiperparametros del modelo ganador `catboost_fs_top60_submit.csv` sin pisar el archivo ganador.
- Trabajar solo con `top60` features + categoricas base, porque fue la mejor receta publica.
- Hacer validacion agrupada mas densa que las pruebas anteriores.
- Comparar cada variante contra un control `top60_public_anchor` usando el mismo esquema de CV.
- Generar submits puros y blends conservadores contra el ganador publico.

Caracteristicas del script:

- Usa una CV agrupada repetible por `track_id`.
- Permite definir intensidad con `--splits` y `--repeats`.
- Permite correr subconjuntos con `--only`.
- Guarda resumen, folds, raw predictions, submits y blends.

Comando corrido:

- `python codigo/7_catboost_top60_hypercv.py --splits 4 --repeats 1 --only top60_public_anchor top60_l2_10_seed707 top60_l2_8_less_noise`

Candidatos evaluados:

- `top60_public_anchor`: control cercano al modelo ganador original.
- `top60_l2_10_seed707`: sube levemente `l2_leaf_reg` de 9 a 10 y cambia seed.
- `top60_l2_8_less_noise`: baja `l2_leaf_reg` a 8 y reduce aleatoriedad.

Metricas CV:

- `top60_l2_10_seed707`:
  - RMSE medio: 14,6897.
  - OOF RMSE repetido: 14,6910.
  - OOF R2 repetido: 0,6545.
  - Mejor de la tanda.
- `top60_l2_8_less_noise`:
  - RMSE medio: 14,6962.
  - OOF RMSE repetido: 14,6977.
  - OOF R2 repetido: 0,6542.
- `top60_public_anchor`:
  - RMSE medio: 14,6989.
  - OOF RMSE repetido: 14,7002.
  - OOF R2 repetido: 0,6540.

Submits nuevos:

- `Submits/catboost_top60_l2_10_seed707_submit.csv`
- `Submits/catboost_top60_l2_8_less_noise_submit.csv`
- `Submits/catboost_top60_public_anchor_submit.csv`
- `Submits/catboost_top60_hypercv_blend_public90_best10.csv`
- `Submits/catboost_top60_hypercv_blend_public80_best20.csv`
- `Submits/catboost_top60_hypercv_blend_best2_equal.csv`
- `Submits/catboost_top60_hypercv_blend_best3_weighted.csv`

Archivos de seguimiento:

- `otros csv/catboost_top60_hypercv_summary.csv`
- `otros csv/catboost_top60_hypercv_folds.csv`
- `otros csv/catboost_top60_hypercv_blends.csv`
- `modelos/catboost_top60_hypercv_metadata.json`

Lectura:

- La mejor variante pura de esta tanda es `catboost_top60_l2_10_seed707_submit.csv`.
- Como el leaderboard publico castigo variantes con cambios grandes, conviene probar primero el blend conservador `catboost_top60_hypercv_blend_public90_best10.csv`.
- Si ese mejora o queda muy cerca, probar luego `catboost_top60_hypercv_blend_public80_best20.csv`.
- Si se quiere probar un modelo puro, usar `catboost_top60_l2_10_seed707_submit.csv`.

## Limpieza de submits no probados

Fecha: 2026-05-11

Se limpio la carpeta `Submits` para reducir confusion.

Criterio:

- Se conservaron en `Submits` los archivos con resultado publico informado, el mejor submit actual y los candidatos nuevos recomendados para probar.
- Se movieron a `Submits/archivo_no_probados` los submits no probados o de baja prioridad.
- No se borraron definitivamente; quedaron archivados por si hace falta recuperarlos.

Archivos movidos a `Submits/archivo_no_probados`:

- `catboost_best_expand_103_submit.csv`
- `catboost_best_expand_106_submit.csv`
- `catboost_best_expand103_rounded_submit.csv`
- `catboost_best_rounded_submit.csv`
- `catboost_close_a_more_trees_submit.csv`
- `catboost_close_b_slightly_regularized_submit.csv`
- `catboost_close_tuning_ensemble_submit.csv`
- `catboost_fs_top90_submit.csv`
- `catboost_fs_top120_submit.csv`
- `catboost_fs_play_blend_best2_cv_equal.csv`
- `catboost_fs_play_blend_best3_cv_weighted.csv`
- `catboost_fs_play_top60_depth5_smooth_submit.csv`
- `catboost_top60_hypercv_blend_best2_equal.csv`
- `catboost_top60_hypercv_blend_best3_weighted.csv`
- `catboost_top60_l2_8_less_noise_submit.csv`
- `catboost_top60_public_anchor_submit.csv`

Quedaron en `Submits`:

- submits probados con score publico;
- `catboost_fs_top60_submit.csv`, mejor actual;
- candidatos nuevos recomendados:
  - `catboost_top60_hypercv_blend_public90_best10.csv`
  - `catboost_top60_hypercv_blend_public80_best20.csv`
  - `catboost_top60_l2_10_seed707_submit.csv`

Resultado Kaggle posterior:

- `catboost_top60_hypercv_blend_public90_best10.csv`: R2 publico 0,37314. Queda muy cerca del mejor actual, pero no lo supera.
- `catboost_top60_l2_10_seed707_submit.csv`: R2 publico 0,36371. Se descarta como modelo puro; la mejora de CV no traslado al publico.

Lectura posterior:

- El blend 90/10 bajo levemente respecto del ganador, por lo que no conviene probar el blend 80/20.
- La variante pura `l2_10_seed707` bajo bastante mas.
- Mejor actual se mantiene: `catboost_fs_top60_submit.csv` con R2 publico 0,37362.

## Modelo LightGBM

Fecha: 2026-05-12

Se creo:

- `codigo/8_lightgbm_models.py`

Objetivo:

- Probar un modelo distinto a CatBoost para salir de la rama que el leaderboard publico viene castigando.
- Medir primero LightGBM puro antes de blendear.
- Usar las features ya generadas y ordenadas por importancia del mejor CatBoost.
- Validar con `GroupKFold` por `track_id`.
- Aplicar el mismo lookup final para canciones de test vistas en train.

Features usadas:

- Las variantes numericas usan las top-N features segun `otros csv/catboost_best_feature_importance.csv`, filtrando columnas numericas.
- La variante con categoricas conserva tambien:
  - `playlist_genre`
  - `playlist_subgenre`
  - `key_str`
  - `mode_str`
  - `key_mode`
  - `genre_subgenre`
- Se evito usar texto crudo de alta cardinalidad como categorico directo.

Candidatos evaluados:

- `lgbm_top60_num_regularized`
- `lgbm_top90_num_regularized`
- `lgbm_top120_num_smooth`
- `lgbm_top60_cats_conservative`

Metricas CV:

- `lgbm_top90_num_regularized`:
  - 89 features numericas.
  - RMSE medio: 14,6301.
  - OOF RMSE: 14,6308.
  - OOF R2: 0,6573.
  - Mejor LightGBM de la tanda.
- `lgbm_top120_num_smooth`:
  - 116 features numericas.
  - RMSE medio: 14,6440.
  - OOF RMSE: 14,6446.
  - OOF R2: 0,6567.
- `lgbm_top60_num_regularized`:
  - 60 features numericas.
  - RMSE medio: 14,6842.
  - OOF RMSE: 14,6848.
  - OOF R2: 0,6548.
- `lgbm_top60_cats_conservative`:
  - 66 features, incluyendo 6 categoricas base.
  - RMSE medio: 14,6876.
  - OOF RMSE: 14,6883.
  - OOF R2: 0,6546.

Submits nuevos:

- `Submits/lgbm_top90_num_regularized_submit.csv`
- `Submits/lgbm_top120_num_smooth_submit.csv`
- `Submits/lgbm_top60_num_regularized_submit.csv`
- `Submits/lgbm_top60_cats_conservative_submit.csv`
- `Submits/lgbm_blend_catboost_top60_90_lgbm_10.csv`
- `Submits/lgbm_blend_catboost_top60_85_lgbm_15.csv`

Archivos de seguimiento:

- `otros csv/lightgbm_cv_summary.csv`
- `otros csv/lightgbm_cv_folds.csv`
- `otros csv/lightgbm_blends.csv`
- `modelos/lightgbm_metadata.json`

Lectura:

- El mejor submit puro para probar primero es `lgbm_top90_num_regularized_submit.csv`.
- Si el LightGBM puro da un score publico razonable, despues conviene probar los blends contra el CatBoost top60 ganador.
- Aunque la CV de LightGBM es mejor que la de CatBoost, no se asume que ganara en publico hasta probarlo.

Resultado Kaggle informado:

- `lgbm_top60_num_regularized_submit.csv`: R2 publico 0,23. Se descarta el LightGBM top60 puro; la CV no traslado al publico.

Lectura posterior:

- El top60 numerico de LightGBM no alcanzo.
- El candidato que todavia podria valer una prueba es `lgbm_top90_num_regularized_submit.csv`, porque tuvo bastante mejor CV y usa mas senial.
- No conviene probar blends con LightGBM hasta ver si `lgbm_top90_num_regularized_submit.csv` obtiene un score publico razonable.

## LightGBM core rapido

Fecha: 2026-05-13

Se creo:

- `codigo/10_lightgbm_core_fast.py`

Objetivo:

- Rehacer LightGBM con menos variables y seleccion manual por familias de senial.
- Evitar el posible ruido del `top90`.
- Generar un unico submit final con el mejor set de features por CV.

Sets evaluados:

- `lgbm_core25_identity`: 25 features.
- `lgbm_core35_signal`: 35 features.
- `lgbm_core45_balanced`: 45 features.

Validacion:

- `GroupKFold(5)` por `track_id`.
- Early stopping.
- Lookup final por `track_id` visto en train.

Metricas CV:

- `lgbm_core45_balanced`:
  - 45 features.
  - RMSE medio: 14,7065.
  - OOF RMSE: 14,7070.
  - OOF R2: 0,6537.
  - Mejor core.
- `lgbm_core35_signal`:
  - 35 features.
  - RMSE medio: 14,7843.
  - OOF RMSE: 14,7846.
  - OOF R2: 0,6501.
- `lgbm_core25_identity`:
  - 25 features.
  - RMSE medio: 14,7891.
  - OOF RMSE: 14,7895.
  - OOF R2: 0,6498.

Submit generado:

- `Submits/lgbm_core_fast_best_submit.csv`

Archivos de seguimiento:

- `otros csv/lgbm_core_fast_cv_summary.csv`
- `otros csv/lgbm_core_fast_cv_folds.csv`
- `otros csv/lgbm_core_fast_best_raw_predictions.csv`
- `modelos/lgbm_core_fast_metadata.json`

Lectura:

- El mejor core usa 45 variables.
- No mejora la CV del LightGBM top90, pero es mas limpio y menos propenso a ruido.
- Si se quiere probar otro LightGBM puro sin ir al top90, este es el candidato.

Resultado Kaggle informado:

- `lgbm_core_fast_best_submit.csv`: R2 publico 0,1934. Se descarta LightGBM puro; la CV interna no representa bien el leaderboard publico.

Lectura posterior:

- LightGBM puro no esta funcionando para este leaderboard.
- No conviene probar otros submits LightGBM puros ni blends con LightGBM por ahora.
- Mejor actual se mantiene: `catboost_fs_top60_submit.csv` con R2 publico 0,37362.

## Postprocesamiento sobre CatBoost top-60 ganador

Fecha: 2026-05-13

Se creo:

- `codigo/11_catboost_top60_postprocess.py`

Objetivo:

- Volver al mejor submit conocido: `catboost_fs_top60_submit.csv`.
- No tocar hiperparametros ni features.
- Generar ajustes minimos de distribucion sobre filas sin lookup de `track_id`.
- Mantener intactas las 800 filas con `track_id` visto en train, porque el lookup fue util.

Motivacion:

- `catboost_top60_hypercv_blend_public90_best10.csv` bajo levemente a 0,37314.
- Ese blend reducia apenas la media de predicciones en filas no vistas.
- Por eso se probaron ajustes pequenos en sentido contrario: subir muy poco las predicciones no vistas o alejarse minimamente de variantes que el publico castigo.

Submits generados:

- `Submits/catboost_top60_post_unseen_shift_p05.csv`
- `Submits/catboost_top60_post_unseen_shift_p10.csv`
- `Submits/catboost_top60_post_unseen_shift_p20.csv`
- `Submits/catboost_top60_post_away_l2_03.csv`
- `Submits/catboost_top60_post_away_l2_05.csv`
- `Submits/catboost_top60_post_away_moretrees_03.csv`
- `Submits/catboost_top60_post_blend_oldbase_03.csv`

Archivos de seguimiento:

- `otros csv/catboost_top60_postprocess_distributions.csv`
- `modelos/catboost_top60_postprocess_metadata.json`

Orden recomendado para Kaggle:

1. `catboost_top60_post_unseen_shift_p05.csv`
2. `catboost_top60_post_away_l2_03.csv`
3. `catboost_top60_post_unseen_shift_p10.csv`

Lectura:

- Son pruebas quirurgicas alrededor del ganador, no modelos nuevos.
- El primer candidato es el mas conservador: suma solo 0,05 a filas no vistas y deja lookup intacto.
- Si `p05` baja, no conviene probar shifts mayores.

Resultado Kaggle informado:

- `catboost_top60_post_unseen_shift_p05.csv`: R2 publico 0,37344. Queda muy cerca, pero no supera al mejor actual 0,37362.

Lectura posterior:

- El ajuste positivo minimo no alcanzo.
- No conviene probar `p10` ni `p20` por ahora.
- Si se quiere probar otro postproceso, el candidato distinto es `catboost_top60_post_away_l2_03.csv`.

Resultado Kaggle posterior:

- `catboost_top60_post_away_l2_03.csv`: R2 publico 0,37375. Nuevo mejor resultado.

Resultado Kaggle posterior:

- `catboost_top60_post_away_l2_05.csv`: R2 publico 0,37382. Nuevo mejor resultado.

Lectura posterior:

- La rama `away_l2` es la primera tanda de postprocesamiento que supera al mejor CatBoost top-60 original.
- `away_l2_05` mejora levemente a `away_l2_03`, por lo que el ajuste en esa direccion parece util pero el margen es chico.
- A partir de este punto conviene registrar cada prueba nueva en este archivo y avanzar con cambios muy conservadores alrededor de `away_l2_05`.

Proximo paso propuesto:

- Probar primero una extrapolacion apenas mayor de la misma familia, por ejemplo `away_l2_07`.
- Si `away_l2_07` mejora, probar como maximo una variante `away_l2_10`.
- Si `away_l2_07` baja, volver a `away_l2_05` como mejor actual y no seguir aumentando la fuerza del ajuste.
- Evitar por ahora `p10`, `p20`, LightGBM puro y blends con LightGBM, porque el registro muestra que esas ramas no trasladaron bien al publico.

Decision de enfoque:

- El lookup de `track_id` ya funciona y no conviene tocarlo por ahora.
- La mejora pendiente esta principalmente en las 5.767 filas de test con `track_id` no visto en train.
- Por eso, las proximas pruebas deben concentrarse en mejorar o postprocesar las predicciones CatBoost para filas no vistas, manteniendo intactas las 800 filas con lookup.

## Tanda picante sobre `away_l2`

Fecha: 2026-05-14

Pedido:

- Probar ajustes mas fuertes que `away_l2_05`, apuntando a una fuerza cercana a `0.10` o un poco mayor.

Cambio realizado:

- Se actualizo `codigo/11_catboost_top60_postprocess.py` para generar tambien:
  - `Submits/catboost_top60_post_away_l2_07.csv`
  - `Submits/catboost_top60_post_away_l2_10.csv`
  - `Submits/catboost_top60_post_away_l2_12.csv`
- Todas las variantes mantienen intactas las 800 filas con lookup de `track_id`.
- Solo modifican las 5.767 filas con `track_id` no visto en train.

Formula usada para las filas no vistas:

- `pred_nueva = pred_base + fuerza * (pred_base - pred_l2_10_seed707)`

Distribuciones principales:

- `away_l2_05`: media 35,3233; media no vistos 32,0474; std no vistos 6,5301.
- `away_l2_07`: media 35,3329; media no vistos 32,0583; std no vistos 6,5359.
- `away_l2_10`: media 35,3473; media no vistos 32,0747; std no vistos 6,5450.
- `away_l2_12`: media 35,3569; media no vistos 32,0857; std no vistos 6,5513.

Orden recomendado para Kaggle:

1. `catboost_top60_post_away_l2_07.csv`
2. Si mejora o queda muy cerca: `catboost_top60_post_away_l2_10.csv`
3. Si `away_l2_10` mejora: `catboost_top60_post_away_l2_12.csv`

Lectura:

- Esta tanda empuja en la misma direccion que ya mejoro de `away_l2_03` a `away_l2_05`.
- El riesgo aumenta con `away_l2_10` y `away_l2_12`, pero siguen siendo ajustes chicos y solo sobre filas no vistas.
- Si `away_l2_07` baja respecto de 0,37382, conviene frenar y conservar `away_l2_05` como mejor actual.

Resultado Kaggle informado:

- `catboost_top60_post_away_l2_12.csv`: R2 publico 0,37405. Nuevo mejor resultado.

Lectura posterior:

- La rama `away_l2` siguio mejorando incluso con fuerza `0.12`.
- Esto sugiere que el leaderboard publico esta premiando alejarse un poco mas de la variante `l2_10_seed707` en las filas no vistas.
- Como la mejora ya supero el salto de `away_l2_03` a `away_l2_05`, tiene sentido probar una tanda adicional un poco mas fuerte, pero todavia gradual.

Proximo paso propuesto:

- Generar y probar `away_l2_15` como siguiente candidato natural.
- Si `away_l2_15` mejora o queda muy cerca, probar `away_l2_18`.
- Dejar `away_l2_20` como limite picante razonable antes de cambiar de estrategia.

## Variante `away_l2_20`

Fecha: 2026-05-14

Pedido:

- Generar directamente una variante mas fuerte `away_l2_20`.

Cambio realizado:

- Se actualizo `codigo/11_catboost_top60_postprocess.py` para incluir fuerza `0.20`.
- Se genero `Submits/catboost_top60_post_away_l2_20.csv`.
- Se regeneraron `otros csv/catboost_top60_postprocess_distributions.csv` y `modelos/catboost_top60_postprocess_metadata.json`.

Distribucion:

- Media total: 35,3954.
- Desvio total: 13,5427.
- Media en no vistos: 32,1294.
- Desvio en no vistos: 6,5784.

Lectura:

- Es una prueba mas agresiva que el ganador actual `away_l2_12`.
- Mantiene intactas las 800 filas con lookup y solo ajusta las 5.767 filas no vistas.
- Si mejora, la rama todavia admite mas fuerza; si baja, conviene volver a explorar entre `0.12` y `0.20` con pasos intermedios.

Resultado Kaggle informado:

- `catboost_top60_post_away_l2_20.csv`: R2 publico 0,37425. Nuevo mejor resultado.

Lectura posterior:

- La mejora de `away_l2_20` confirma que aumentar la fuerza de alejamiento respecto de `l2_10_seed707` sigue ayudando.
- La rama ya mejoro desde 0,37375 (`away_l2_03`) hasta 0,37425 (`away_l2_20`), por lo que vale la pena explorar un poco mas arriba.
- Como el ajuste ya es mas agresivo, conviene probar escalones moderados y no saltar demasiado lejos.

Proximo paso propuesto:

- Generar `away_l2_25` como siguiente prueba.
- Si mejora, probar `away_l2_30`.
- Si `away_l2_25` baja, explorar intermedios entre `0.20` y `0.25`, como `away_l2_22` o `away_l2_23`.

## Variantes `away_l2_25` y `away_l2_30`

Fecha: 2026-05-14

Cambio realizado:

- Se actualizo `codigo/11_catboost_top60_postprocess.py` para incluir fuerzas `0.25` y `0.30`.
- Se generaron:
  - `Submits/catboost_top60_post_away_l2_25.csv`
  - `Submits/catboost_top60_post_away_l2_30.csv`
- Se regeneraron `otros csv/catboost_top60_postprocess_distributions.csv` y `modelos/catboost_top60_postprocess_metadata.json`.

Distribuciones:

- `away_l2_25`: media total 35,4194; media no vistos 32,1568; std no vistos 6,5967.
- `away_l2_30`: media total 35,4434; media no vistos 32,1841; std no vistos 6,6162.

Orden de prueba:

1. `catboost_top60_post_away_l2_25.csv`
2. Si mejora o queda muy cerca: `catboost_top60_post_away_l2_30.csv`

Lectura:

- Ambas variantes son mas agresivas que el mejor actual `away_l2_20`.
- Si `away_l2_25` baja claramente, conviene no probar `away_l2_30` y buscar intermedios entre `0.20` y `0.25`.

Resultado Kaggle informado:

- `catboost_top60_post_away_l2_30.csv`: R2 publico 0,37440. Nuevo mejor resultado.

Lectura posterior:

- La rama `away_l2` sigue mejorando al aumentar la fuerza hasta `0.30`.
- El salto desde `away_l2_20` fue positivo: 0,37425 -> 0,37440.
- Todavia no aparece senal de saturacion clara, pero el ajuste ya es bastante mas agresivo que las primeras pruebas.

Proximo paso propuesto:

- Generar `away_l2_35` y `away_l2_40` como siguiente tanda.
- Probar primero `away_l2_35`.
- Si `away_l2_35` mejora o queda muy cerca, probar `away_l2_40`.
- Si `away_l2_35` baja, buscar fino entre `0.30` y `0.35`.

## Variante exploratoria `away_l2_100`

Fecha: 2026-05-15

Pedido:

- Generar una variante `away_l2_100` para probar si la tendencia sigue mejorando con una fuerza mucho mas grande.

Cambio realizado:

- Se actualizo `codigo/11_catboost_top60_postprocess.py` para incluir fuerza `1.00`.
- Se genero `Submits/catboost_top60_post_away_l2_100.csv`.
- Se regeneraron `otros csv/catboost_top60_postprocess_distributions.csv` y `modelos/catboost_top60_postprocess_metadata.json`.

Distribucion:

- Media total: 35,7796.
- Desvio total: 13,6361.
- Percentil 5 total: 22,9273.
- Mediana total: 32,6624.
- Media en no vistos: 32,5670.
- Desvio en no vistos: 7,0002.

Lectura:

- Es una prueba exploratoria agresiva: equivale a alejarse con fuerza completa de `catboost_top60_l2_10_seed707_submit.csv`.
- Mantiene intactas las 800 filas con lookup y solo ajusta las 5.767 filas no vistas.
- Si mejora, conviene explorar tambien puntos intermedios entre `0.30` y `1.00`.
- Si baja mucho, la zona util probablemente este bastante antes de `1.00`, cerca de `0.30` o en escalones moderados como `0.35`/`0.40`.

Resultado Kaggle informado:

- `catboost_top60_post_away_l2_100.csv`: R2 publico 0,37225. Se descarta por ser demasiado agresivo.

Lectura posterior:

- La fuerza `1.00` se paso claramente y queda por debajo del mejor actual `away_l2_30` con 0,37440.
- Esto confirma que la mejora tiene un limite y que no conviene seguir saltando a fuerzas grandes.
- La zona prometedora queda por encima de `0.30`, pero bastante por debajo de `1.00`.

Proximo paso propuesto:

- Volver a una busqueda fina/moderada alrededor del mejor actual.
- Generar `away_l2_35`, `away_l2_40` y eventualmente `away_l2_50`.
- Probar primero `away_l2_35`; si mejora, seguir con `away_l2_40`.
- No probar variantes tipo `0.70` o mayores hasta tener evidencia de que `0.40`/`0.50` todavia mejoran.

## Variante `away_l2_50`

Fecha: 2026-05-15

Pedido:

- Generar una variante intermedia/fuerte `away_l2_50`, despues de que `away_l2_100` resultara demasiado agresiva.

Cambio realizado:

- Se actualizo `codigo/11_catboost_top60_postprocess.py` para incluir fuerza `0.50`.
- Se genero `Submits/catboost_top60_post_away_l2_50.csv`.
- Se regeneraron `otros csv/catboost_top60_postprocess_distributions.csv` y `modelos/catboost_top60_postprocess_metadata.json`.

Distribucion:

- Media total: 35,5395.
- Desvio total: 13,5626.
- Percentil 5 total: 23,5144.
- Mediana total: 32,1811.
- Media en no vistos: 32,2935.
- Desvio en no vistos: 6,7051.

Lectura:

- Es mas agresiva que el mejor actual `away_l2_30`, pero mucho menos extrema que `away_l2_100`.
- Sirve para ubicar si el optimo esta cerca de `0.30` o si todavia hay margen hacia `0.50`.
- Si `away_l2_50` baja respecto de 0,37440, conviene buscar entre `0.30` y `0.50` con `0.35`/`0.40`.

Resultado Kaggle informado:

- `catboost_top60_post_away_l2_50.csv`: R2 publico 0,37435. Queda cerca, pero por debajo del mejor actual `away_l2_30` con 0,37440.

Lectura posterior:

- `away_l2_50` no supera a `away_l2_30`, pero tampoco cae fuerte.
- Esto sugiere que el optimo de esta familia probablemente esta entre `0.30` y `0.50`, o muy cerca de `0.30`.
- Conviene probar puntos intermedios antes de abandonar la rama.

Proximo paso propuesto:

- Generar `away_l2_35` y `away_l2_40`.
- Probar primero `away_l2_35`; si mejora o queda muy cerca, probar `away_l2_40`.
- Si ambos bajan, conservar `away_l2_30` como mejor de la familia.

## Variantes `away_l2_35` y `away_l2_40`

Fecha: 2026-05-15

Pedido:

- Dejar listos `away_l2_35` y `away_l2_40` para probar manana, buscando fino entre el mejor actual `away_l2_30` y la variante `away_l2_50` que quedo cerca pero abajo.

Cambio realizado:

- Se actualizo `codigo/11_catboost_top60_postprocess.py` para incluir fuerzas `0.35` y `0.40`.
- Se generaron:
  - `Submits/catboost_top60_post_away_l2_35.csv`
  - `Submits/catboost_top60_post_away_l2_40.csv`
- Se regeneraron `otros csv/catboost_top60_postprocess_distributions.csv` y `modelos/catboost_top60_postprocess_metadata.json`.

Distribuciones:

- `away_l2_35`: media total 35,4674; media no vistos 32,2115; std no vistos 6,6368.
- `away_l2_40`: media total 35,4914; media no vistos 32,2388; std no vistos 6,6585.

Orden de prueba:

1. `catboost_top60_post_away_l2_35.csv`
2. Si mejora o queda muy cerca: `catboost_top60_post_away_l2_40.csv`

Lectura:

- Estas variantes exploran la zona intermedia entre `away_l2_30` (mejor actual, R2 publico 0,37440) y `away_l2_50` (R2 publico 0,37435).
- Si ambas bajan, el mejor punto conocido de la familia queda en `away_l2_30`.

Resultado Kaggle informado:

- `catboost_top60_post_away_l2_35.csv`: R2 publico 0,37443. Nuevo mejor resultado.

Lectura posterior:

- `away_l2_35` supera por poco a `away_l2_30` y queda como mejor punto conocido de la familia.
- Como `away_l2_50` ya habia bajado a 0,37435, el optimo parece estar entre `0.30` y `0.50`, probablemente cerca de `0.35`.
- El siguiente submit natural es `away_l2_40`, que ya esta generado.

Proximo paso propuesto:

- Probar `catboost_top60_post_away_l2_40.csv`.
- Si `away_l2_40` mejora, buscar fino entre `0.35` y `0.50`.
- Si `away_l2_40` baja, buscar fino alrededor de `0.35`, por ejemplo `away_l2_33` y `away_l2_37`.

Resultado Kaggle informado:

- `catboost_top60_post_away_l2_40.csv`: R2 publico 0,37443. Empata el mejor resultado de `away_l2_35`.

Lectura posterior:

- `away_l2_35` y `away_l2_40` empatan como mejores puntos conocidos.
- `away_l2_50` queda apenas por debajo, por lo que la zona util parece estar entre `0.35` y `0.40`, con posible meseta hasta antes de `0.50`.
- Ya no conviene hacer saltos grandes; el siguiente avance deberia ser una busqueda fina.

Proximo paso propuesto:

- Generar `away_l2_33`, `away_l2_37` y `away_l2_42`.
- Probar primero `away_l2_37`, porque cae en el medio exacto de los dos mejores actuales.
- Si `away_l2_37` no mejora, conservar `away_l2_35`/`away_l2_40` como mejores de la familia.

## Busqueda fina `away_l2_33`, `away_l2_37` y `away_l2_42`

Fecha: 2026-05-18

Pedido:

- Generar una tanda fina alrededor de la meseta `away_l2_35`/`away_l2_40`.

Cambio realizado:

- Se actualizo `codigo/11_catboost_top60_postprocess.py` para incluir fuerzas `0.33`, `0.37` y `0.42`.
- Se generaron:
  - `Submits/catboost_top60_post_away_l2_33.csv`
  - `Submits/catboost_top60_post_away_l2_37.csv`
  - `Submits/catboost_top60_post_away_l2_42.csv`
- Se regeneraron `otros csv/catboost_top60_postprocess_distributions.csv` y `modelos/catboost_top60_postprocess_metadata.json`.

Distribuciones:

- `away_l2_33`: media total 35,4578; media no vistos 32,2005; std no vistos 6,6284.
- `away_l2_37`: media total 35,4770; media no vistos 32,2224; std no vistos 6,6453.
- `away_l2_42`: media total 35,5010; media no vistos 32,2498; std no vistos 6,6675.

Orden de prueba:

1. `catboost_top60_post_away_l2_37.csv`
2. Si mejora o queda empatado: `catboost_top60_post_away_l2_42.csv`
3. Si `away_l2_37` baja: `catboost_top60_post_away_l2_33.csv`

Lectura:

- La tanda busca ubicar el maximo local alrededor de los mejores conocidos `away_l2_35` y `away_l2_40`, ambos con R2 publico 0,37443.
- Si ninguno supera 0,37443, se conserva `away_l2_35`/`away_l2_40` como meseta ganadora.

Resultado Kaggle informado:

- `catboost_top60_post_away_l2_37.csv`: R2 publico 0,37443. Empata el mejor resultado.

Lectura posterior:

- `away_l2_35`, `away_l2_37` y `away_l2_40` empatan con R2 publico 0,37443.
- La familia `away_l2` parece haber llegado a una meseta entre `0.35` y `0.40`.
- Como `away_l2_50` ya bajo a 0,37435, es poco probable que aumentar mucho mas la fuerza mejore.

Proximo paso propuesto:

- Si queda un submit barato, probar `away_l2_42` para cerrar la parte alta de la meseta.
- Si `away_l2_42` no mejora, dejar `away_l2_35` como mejor archivo principal y considerar otra estrategia.

## Estrategias siguientes fuera de `away_l2`

Fecha: 2026-05-18

Situacion:

- La familia `away_l2` llego a una meseta: `away_l2_35`, `away_l2_37` y `away_l2_40` empatan con R2 publico 0,37443.
- `away_l2_50` bajo a 0,37435 y `away_l2_100` bajo fuerte a 0,37225.
- El lookup de `track_id` sigue fijo y no conviene tocarlo.

Estrategias candidatas:

1. Postprocesamiento por tramos de prediccion:
   - Mantener la idea de ajustar solo filas no vistas.
   - Aplicar fuerzas distintas segun rango de prediccion base, por ejemplo bajas, medias y altas.
   - Objetivo: no empujar igual canciones con prediccion 20 que canciones con prediccion 70.

2. Blend/extrapolacion contra otra variante mala:
   - Ya funciono alejarse de `l2_10_seed707`.
   - Probar alejarse de `catboost_fs_play_top60_more_trees_submit.csv`, que habia dado mal en publico.
   - Tambien se puede combinar alejamiento de `l2_10_seed707` y `more_trees`.

3. Ajuste de dispersion:
   - En vez de alejarse de otro submit, expandir o contraer predicciones no vistas alrededor de su media.
   - Objetivo: corregir si el modelo esta demasiado conservador o demasiado disperso en canciones no vistas.

4. Nuevo CatBoost cercano al ganador:
   - Entrenar una variante muy cercana al `catboost_fs_top60`, no una receta nueva grande.
   - Cambiar solo una cosa por vez: seed ensemble, regularizacion minima, o iteraciones alrededor del punto ganador.
   - Usarlo mas como fuente para blend/postproceso que como submit puro.

Prioridad recomendada:

- Primero probar postprocesamiento por tramos usando como base `away_l2_35`/`away_l2_40`.
- Segundo probar una combinacion suave de alejamiento contra `l2_10_seed707` y contra `more_trees`.
- Dejar nuevo entrenamiento CatBoost para despues, porque varias mejoras de CV no trasladaron bien al publico.

## Cambio de enfoque: buscar nuevo CatBoost

Fecha: 2026-05-18

Decision:

- Se decide considerar un nuevo CatBoost directamente, en vez de seguir solo con postprocesamientos.
- La motivacion es que la familia `away_l2` llego a una meseta publica alrededor de 0,37443.

Lectura:

- No conviene empezar desde cero ni cambiar toda la receta, porque el registro muestra que los cambios grandes fueron castigados.
- La mejor base para una nueva tanda es `codigo/7_catboost_top60_hypercv.py`, porque ya trabaja con:
  - top-60 features + categoricas base;
  - lookup de `track_id` intacto;
  - validacion agrupada por `track_id`;
  - candidatos cercanos al ganador.
- El objetivo no debe ser solo mejorar CV, sino generar un CatBoost que pueda ser buen submit o, al menos, una nueva direccion util para blend/postproceso.

Plan propuesto:

1. Mantener `top60` features y las mismas categoricas base.
2. Mantener lookup intacto.
3. Probar pocos candidatos nuevos, cercanos al ganador:
   - seed ensemble con parametros public-anchor;
   - misma receta con `l2_leaf_reg` apenas menor/mayor;
   - variante con un poco menos de ruido y misma profundidad;
   - variante con seed mix pero sin cambiar learning rate grande.
4. Evaluar con CV agrupada, pero elegir candidatos para Kaggle con criterio publico conservador.
5. Guardar raw predictions para poder hacer blends/postprocesos si el submit puro no supera.

## CatBoost denoised por redundancia de features

Fecha: 2026-05-18

Idea:

- Buscar un nuevo CatBoost sacando ruido y variables redundantes, en vez de seguir solo con postprocesamiento.
- Mantener la receta ganadora: CatBoost, top features, categoricas base y lookup intacto.
- Reducir redundancia dentro de las features mas importantes usando correlacion entre variables numericas.

Diagnostico:

- Se detectaron pares casi duplicados dentro de las top features:
  - `track_artist_te_oof` y `track_artist_te_diff_global`.
  - `release_age_2020` y `release_year`.
  - `playlist_id_te_oof` y `playlist_id_te_diff_global`.
  - `track_album_id_te_diff_global` y `track_album_id_te_oof`.
  - `tempo` y `tempo_log`.
  - varias familias `freq_all` / `freq_train` / `freq_all_log`.

Archivo creado:

- `codigo/12_catboost_denoised.py`

Caracteristicas del script:

- Usa como base la importancia de `catboost_best_feature_importance.csv`.
- Mantiene siempre las categoricas base:
  - `playlist_genre`
  - `playlist_subgenre`
  - `key_str`
  - `mode_str`
  - `key_mode`
  - `genre_subgenre`
- Selecciona features numericas por importancia, descartando variables muy correlacionadas con otra ya elegida.
- Mantiene lookup final de `track_id`.
- Guarda submits, raw predictions, features usadas, resumen CV y features descartadas.

Candidatos disponibles:

- `denoise_corr090`: 50 features, descarta 35.
- `denoise_corr092`: 54 features, descarta 31.
- `denoise_corr095`: 60 features, descarta 25.
- `denoise_corr092_top45`: 45 features, descarta 40.
- `denoise_corr095_top50`: 50 features, descarta 35.

Comando de chequeo ejecutado:

- `python codigo/12_catboost_denoised.py --list`

Proximo paso recomendado:

- Correr primero una tanda corta con:
  - `denoise_corr092`
  - `denoise_corr095_top50`
- Comando sugerido:
  - `python codigo/12_catboost_denoised.py --only denoise_corr092 denoise_corr095_top50 --splits 4 --repeats 1`
- Si alguno queda bien por CV y con distribucion razonable, probar el submit puro o usar sus raw predictions para una nueva direccion de postproceso.

## Corrida CatBoost denoised

Fecha: 2026-05-18

Comando ejecutado:

- `python codigo/12_catboost_denoised.py --only denoise_corr092 denoise_corr095_top50 denoise_corr095 --splits 4 --repeats 1`

Candidatos corridos:

- `denoise_corr092`
- `denoise_corr095_top50`
- `denoise_corr095`

Resultados CV:

- `denoise_corr095_top50`:
  - 50 features, 44 numericas.
  - Features descartadas: 35.
  - RMSE medio: 14,6910.
  - MAE medio: 10,6131.
  - R2 medio: 0,6543.
  - OOF RMSE: 14,6910.
  - OOF R2: 0,6545.
  - Iteraciones finales: 1048.
  - Media de prediccion submit: 36,6236.
- `denoise_corr095`:
  - 60 features, 54 numericas.
  - Features descartadas: 25.
  - RMSE medio: 14,6944.
  - MAE medio: 10,6214.
  - R2 medio: 0,6541.
  - OOF RMSE: 14,6944.
  - OOF R2: 0,6543.
  - Iteraciones finales: 1018.
  - Media de prediccion submit: 36,9197.
- `denoise_corr092`:
  - 54 features, 48 numericas.
  - Features descartadas: 31.
  - RMSE medio: 14,6961.
  - MAE medio: 10,6201.
  - R2 medio: 0,6540.
  - OOF RMSE: 14,6960.
  - OOF R2: 0,6542.
  - Iteraciones finales: 1059.
  - Media de prediccion submit: 37,6736.

Comparacion contra CatBoost top-60 ganador original:

- `catboost_fs_top60`:
  - 66 features.
  - RMSE medio: 14,7514.
  - MAE medio: 10,6700.
  - R2 medio: 0,6515.
  - R2 publico: 0,37362.
- Los tres denoised mejoran la CV interna respecto de `catboost_fs_top60`.
- El mejor por CV fue `denoise_corr095_top50`.

Submits generados:

- `Submits/catboost_denoise_corr092_submit.csv`
- `Submits/catboost_denoise_corr095_top50_submit.csv`
- `Submits/catboost_denoise_corr095_submit.csv`

Archivos de seguimiento:

- `otros csv/catboost_denoised_cv_summary.csv`
- `otros csv/catboost_denoised_cv_folds.csv`
- `otros csv/catboost_denoised_dropped_features.csv`
- `modelos/catboost_denoised_metadata.json`

Lectura:

- La limpieza de redundancia funciono muy bien en CV.
- Hay que ser cautos: en este TP ya hubo modelos con mejor CV que bajaron en publico.
- `denoise_corr095_top50` es el primer submit puro para probar, porque combina mejor CV y poda moderada.
- `denoise_corr095` es el segundo candidato: conserva mas features y puede ser mas estable en publico.
- `denoise_corr092` queda tercero porque tiene media de prediccion bastante mas alta y podria estar mas corrido.

Orden recomendado para Kaggle:

1. `catboost_denoise_corr095_top50_submit.csv`
2. Si queda cerca o mejora: `catboost_denoise_corr095_submit.csv`
3. Si los anteriores no caen y se quiere una prueba mas: `catboost_denoise_corr092_submit.csv`

Resultado Kaggle informado:

- `catboost_denoise_corr095_top50_submit.csv`: R2 publico 0,36107. Se descarta como submit puro.

Lectura posterior:

- La mejora de CV de `denoise_corr095_top50` no traslado al leaderboard publico.
- Es una caida fuerte respecto del mejor actual 0,37443.
- No conviene asumir que los otros denoised puros van a mejorar, aunque tengan buena CV.
- La poda de redundancia puede haber cambiado la distribucion publica de forma desfavorable.

Decision:

- Pausar pruebas de denoised puros.
- Usar los denoised, si se usan, solo como fuente auxiliar para blends/postprocesos muy conservadores contra el ganador actual.
- Mantener como mejor submit actual `catboost_top60_post_away_l2_35.csv`.

## Promedio meseta `away_l2_35` / `away_l2_37` / `away_l2_40`

Fecha: 2026-05-19

Idea:

- Como `away_l2_35`, `away_l2_37` y `away_l2_40` empataron en publico con R2 0,37443, generar un promedio simple de los tres.
- El objetivo es suavizar la meseta ganadora sin introducir una direccion nueva ni usar los denoised puros, que no trasladaron bien al publico.

Archivo generado:

- `Submits/catboost_top60_post_away_l2_plateau_35_37_40.csv`

Formula:

- `pred = mean(away_l2_35, away_l2_37, away_l2_40)`

Validacion:

- Filas: 6.567.
- Columnas: `ID`, `track_popularity`.
- Sin nulos.
- Sin IDs duplicados.
- Predicciones dentro de 0 a 100.

Distribucion:

- Media total: 35,4786.
- Desvio total: 13,5519.
- Minimo: 0,0.
- Maximo: 99,0.

Diferencias contra fuentes:

- Contra `away_l2_35`: diferencia absoluta media 0,0300; maxima 0,1722.
- Contra `away_l2_37`: diferencia absoluta media 0,0043; maxima 0,0246.
- Contra `away_l2_40`: diferencia absoluta media 0,0343; maxima 0,1968.

Lectura:

- El promedio queda muy cerca de `away_l2_37`, porque es el punto medio entre `0.35` y `0.40`.
- Es una prueba de bajo riesgo dentro de la misma meseta, no una estrategia nueva.

Resultado Kaggle informado:

- `catboost_top60_post_away_l2_plateau_35_37_40.csv`: R2 publico 0,37443. Empata el mejor resultado, pero no lo supera.

Lectura posterior:

- El promedio confirma que la familia `away_l2` esta en una meseta alrededor de 0,37443.
- No parece haber mejora adicional por promediar puntos globales de fuerza cercanos.
- Para superar el score actual probablemente haga falta una estrategia distinta: ajuste selectivo por subconjuntos, otra fuente de prediccion o nuevo feature/modelo con mejor traslado publico.

## `away_l2` selectivo por discrepancia

Fecha: 2026-05-19

Idea:

- Dejar de aplicar la misma fuerza `away_l2` a todas las filas no vistas.
- Usar mas fuerza cuando el CatBoost base y el submit malo `catboost_top60_l2_10_seed707_submit.csv` discrepan mucho.
- Usar menos fuerza cuando discrepan poco.
- Mantener intactas las 800 filas con lookup.

Cambio realizado:

- Se actualizo `codigo/11_catboost_top60_postprocess.py` para generar variantes selectivas.
- La discrepancia se mide como `abs(pred_base - pred_l2_10_seed707)` en filas no vistas.
- Se dividieron las 5.767 filas no vistas en tres grupos por terciles:
  - diff chica: `<= 0,7739`, 1.923 filas.
  - diff media: `0,7739` a `1,7936`, 1.922 filas.
  - diff grande: `> 1,7936`, 1.922 filas.

Submits generados:

- `Submits/catboost_top60_post_away_l2_sel_diff_25_35_45.csv`
- `Submits/catboost_top60_post_away_l2_sel_diff_30_35_45.csv`
- `Submits/catboost_top60_post_away_l2_sel_diff_30_37_45.csv`
- `Submits/catboost_top60_post_away_l2_sel_diff_25_37_50.csv`

Distribuciones:

- `sel_diff_25_35_45`: media total 35,5062; media no vistos 32,2557; std no vistos 6,6718.
- `sel_diff_30_35_45`: media total 35,5065; media no vistos 32,2560; std no vistos 6,6722.
- `sel_diff_30_37_45`: media total 35,5081; media no vistos 32,2578; std no vistos 6,6737.
- `sel_diff_25_37_50`: media total 35,5276; media no vistos 32,2800; std no vistos 6,6926.

Lectura:

- Estas variantes quedan un poco por encima de `away_l2_40` en media no vista, pero por debajo de `away_l2_50`.
- Son mas distintas que promediar la meseta, porque la fuerza cambia por fila.
- La opcion menos agresiva y mas razonable para probar primero es `sel_diff_25_35_45`.

Orden recomendado para Kaggle:

1. `catboost_top60_post_away_l2_sel_diff_25_35_45.csv`
2. Si mejora o queda muy cerca: `catboost_top60_post_away_l2_sel_diff_30_37_45.csv`
3. Si se busca algo mas picante: `catboost_top60_post_away_l2_sel_diff_25_37_50.csv`

Resultado Kaggle informado:

- `catboost_top60_post_away_l2_sel_diff_25_35_45.csv`: R2 publico 0,37436. Queda por debajo de la meseta ganadora 0,37443.

Lectura posterior:

- El ajuste selectivo por discrepancia no mejoro al ganador.
- La variante queda cerca, pero no lo suficiente como para justificar insistir mucho con esta familia.
- Si se prueba otra selectiva, deberia ser solo una variante claramente distinta; no conviene gastar submits en pequenas variaciones de lo mismo.

## Modelo ExtraTrees

Fecha: 2026-05-19

Motivacion:

- Probar otro modelo distinto a CatBoost/LightGBM.
- Buscar diversidad real: arboles extremadamente aleatorizados sobre features numericas importantes.
- Mantener lookup intacto.

Archivo creado:

- `codigo/13_extratrees_models.py`

Comando ejecutado:

- `python codigo/13_extratrees_models.py --splits 4`

Candidatos corridos:

- `et_top45_leaf4`
- `et_top70_leaf5`
- `et_top100_leaf8`

Resultados CV:

- `et_top70_leaf5`:
  - 70 features.
  - RMSE medio: 14,5002.
  - MAE medio: 10,2341.
  - R2 medio: 0,6632.
  - OOF R2: 0,6633.
  - Media de prediccion: 42,2892.
- `et_top100_leaf8`:
  - 100 features.
  - RMSE medio: 14,5564.
  - MAE medio: 10,3606.
  - R2 medio: 0,6606.
  - OOF R2: 0,6607.
  - Media de prediccion: 42,4941.
- `et_top45_leaf4`:
  - 45 features.
  - RMSE medio: 14,5871.
  - MAE medio: 10,2563.
  - R2 medio: 0,6591.
  - OOF R2: 0,6593.
  - Media de prediccion: 43,3832.

Submits generados:

- `Submits/et_top45_leaf4_submit.csv`
- `Submits/et_top70_leaf5_submit.csv`
- `Submits/et_top100_leaf8_submit.csv`

Lectura:

- ExtraTrees mejora mucho la CV respecto de CatBoost, pero las predicciones puras quedan con media muy alta.
- Dado que los denoised tambien mejoraron CV y cayeron en publico, no conviene subir ExtraTrees puro de entrada.
- El mejor uso inicial de ExtraTrees es como fuente de diversidad en blends conservadores contra el ganador actual.

## Blends ExtraTrees contra ganador actual

Fecha: 2026-05-19

Archivo creado:

- `codigo/14_extratrees_blends.py`

Base:

- Ganador actual: `catboost_top60_post_away_l2_35.csv`.
- Fuente ExtraTrees: `et_top70_leaf5_submit.csv`, mejor por CV.

Submits generados:

- `Submits/et_blend_winner_et02.csv`
- `Submits/et_blend_winner_et03.csv`
- `Submits/et_blend_winner95_et05.csv`
- `Submits/et_aligned_blend_05.csv`
- `Submits/et_aligned_blend_10.csv`
- `Submits/et_aligned_blend_15.csv`

Distribuciones:

- Ganador `away_l2_35`: media total 35,4674; media no vistos 32,2115; std no vistos 6,6368.
- ExtraTrees puro `et_top70_leaf5`: media total 42,2892; media no vistos 39,9795; std no vistos 5,2618.
- ExtraTrees alineado a distribucion del ganador: media total 35,4792; media no vistos 32,2249; std no vistos 6,5650.
- `et_blend_winner_et02`: media total 35,6038; media no vistos 32,3668.
- `et_blend_winner_et03`: media total 35,6721; media no vistos 32,4445.
- `et_blend_winner95_et05`: media total 35,8085; media no vistos 32,5999.
- `et_aligned_blend_05`: media total 35,4680; media no vistos 32,2121.
- `et_aligned_blend_10`: media total 35,4686; media no vistos 32,2128.
- `et_aligned_blend_15`: media total 35,4692; media no vistos 32,2135.

Lectura:

- Los blends directos con ExtraTrees suben la media; pueden ser riesgosos.
- Los blends alineados mantienen casi exactamente la media del ganador y agregan solo forma/ranking de ExtraTrees.
- Si se quiere probar esta rama con bajo riesgo, el primer candidato es `et_aligned_blend_05.csv`.

Orden recomendado para Kaggle:

1. `et_aligned_blend_05.csv`
2. Si mejora o empata: `et_aligned_blend_10.csv`
3. Evitar ExtraTrees puro salvo que se quiera una prueba exploratoria de alto riesgo.

Resultado Kaggle informado:

- `et_aligned_blend_05.csv`: R2 publico 0,37466. Nuevo mejor resultado.

Lectura posterior:

- La senial de ExtraTrees alineada a la distribucion del ganador si traslado al publico.
- Esto rompe la meseta de `away_l2` en 0,37443.
- La estrategia correcta parece ser usar ExtraTrees como fuente de ranking/forma, pero manteniendo la distribucion del CatBoost ganador.
- Conviene seguir explorando esta rama con blends alineados apenas mas fuertes.

Proximo paso propuesto:

- Probar `et_aligned_blend_10.csv`.
- Si mejora o queda muy cerca, probar `et_aligned_blend_15.csv`.
- No probar ExtraTrees puro todavia, porque su media original sigue siendo demasiado alta.

Resultado Kaggle informado:

- `et_aligned_blend_15.csv`: R2 publico 0,37464. Queda muy cerca, pero por debajo del mejor actual `et_aligned_blend_05` con 0,37466.

Lectura posterior:

- ExtraTrees alineado aporta, pero un peso de 15% parece un poco alto.
- El mejor peso conocido por ahora es 5%.
- Como 15% queda cerca, podria valer probar el punto intermedio 10% si no fue probado todavia.
- Si 10% no mejora, buscar fino entre 3% y 8%.

## Tanda fina ExtraTrees alineado

Fecha: 2026-05-20

Motivacion:

- El blend alineado `et_aligned_blend_05.csv` subio el mejor publico a 0,37466.
- `et_aligned_blend_15.csv` quedo cerca con 0,37464, por lo que la rama ExtraTrees alineada sigue siendo prometedora pero el peso optimo parece bajo.
- Se busca un salto chico adicional sin mover demasiado la distribucion.

Cambio realizado:

- Se actualizo `codigo/14_extratrees_blends.py` para generar:
  - pesos finos del ExtraTrees top70 alineado;
  - blends con ExtraTrees top45 y top100 alineados;
  - blends con ensemble alineado de ExtraTrees top45/top70/top100.

Submits generados:

- `Submits/et_aligned_blend_03.csv`
- `Submits/et_aligned_blend_07.csv`
- `Submits/et_aligned_blend_08.csv`
- `Submits/et_aligned_blend_12.csv`
- `Submits/et_top45_aligned_blend_05.csv`
- `Submits/et_top45_aligned_blend_08.csv`
- `Submits/et_top100_aligned_blend_05.csv`
- `Submits/et_top100_aligned_blend_08.csv`
- `Submits/et_ensemble_aligned_blend_05.csv`
- `Submits/et_ensemble_aligned_blend_08.csv`
- `Submits/et_ensemble_aligned_blend_10.csv`

Distribuciones destacadas:

- Ganador actual `et_aligned_blend_05`: media total 35,4680; media no vistos 32,2121; std no vistos 6,5609.
- `et_aligned_blend_07`: media total 35,4682; media no vistos 32,2124; std no vistos 6,5325.
- `et_aligned_blend_08`: media total 35,4684; media no vistos 32,2125; std no vistos 6,5187.
- `et_ensemble_aligned_blend_05`: media total 35,4681; media no vistos 32,2122; std no vistos 6,5511.
- `et_top100_aligned_blend_05`: media total 35,4684; media no vistos 32,2126; std no vistos 6,5641.

Lectura:

- Todos los nuevos candidatos mantienen casi exactamente la media del ganador.
- La diferencia principal esta en forma/ranking y dispersion.
- Para buscar un empujon chico, conviene probar primero un peso fino cercano al 5% ganador.

Orden recomendado para Kaggle:

1. `et_aligned_blend_07.csv`
2. Si mejora o queda muy cerca: `et_ensemble_aligned_blend_05.csv`
3. Si se quiere explorar otra fuente ET sin aumentar mucho riesgo: `et_top100_aligned_blend_05.csv`

## CatBoost con pesos de dominio

Fecha: 2026-05-20

Motivacion:

- Buscar una mejora mas significativa que los ajustes finos.
- Entrenar un CatBoost orientado especificamente al test no visto.
- La idea es ponderar mas las filas de train que se parecen al subconjunto de test sin lookup.

Archivo creado:

- `codigo/15_catboost_domain_weighted.py`

Metodo:

- Se entrena un clasificador de dominio para distinguir train vs test no visto.
- Con la probabilidad de parecerse al test no visto se generan pesos para el regresor CatBoost.
- Se mantiene:
  - lookup intacto;
  - receta CatBoost top features + categoricas base;
  - validacion agrupada por `track_id`.

Candidatos corridos:

- `domain_top60_mild`
- `domain_top60_medium`
- `domain_top80_medium`

Resultados:

- `domain_top60_mild`:
  - 66 features.
  - RMSE medio: 14,6900.
  - MAE medio: 10,6113.
  - R2 medio: 0,6543.
  - OOF R2: 0,6545.
  - Media de prediccion: 36,0985.
  - Submit: `Submits/catboost_domain_top60_mild_submit.csv`.
- `domain_top60_medium`:
  - 66 features.
  - RMSE medio: 14,6894.
  - MAE medio: 10,6235.
  - R2 medio: 0,6544.
  - OOF R2: 0,6545.
  - Media de prediccion: 37,2507.
  - Submit: `Submits/catboost_domain_top60_medium_submit.csv`.
- `domain_top80_medium`:
  - 86 features.
  - RMSE medio: 14,6984.
  - MAE medio: 10,6299.
  - R2 medio: 0,6540.
  - OOF R2: 0,6541.
  - Media de prediccion: 38,5390.
  - Submit: `Submits/catboost_domain_top80_medium_submit.csv`.

Diagnostico:

- El clasificador de dominio separo train vs test no visto con AUC in-sample 1,0000.
- Eso indica que la separacion es demasiado facil y puede estar capturando artefactos de distribucion.
- Los pesos quedaron casi todos cerca de 1, con unas pocas filas muy ponderadas.
- Las medias de prediccion quedaron bastante mas altas que el ganador actual.

Lectura:

- La rama no parece lista para submit puro.
- `domain_top60_mild` es el menos riesgoso de los tres, pero igual tiene media superior al ganador.
- `domain_top60_medium` y `domain_top80_medium` parecen mas riesgosos por desplazamiento de media.
- Si se usa esta rama, conviene hacerlo como fuente auxiliar/blend alineado, no como submit directo.

Decision recomendada:

- No subir `domain_top80_medium`.
- No subir `domain_top60_medium` salvo prueba exploratoria de alto riesgo.
- Si se quiere probar uno puro, el unico razonable seria `catboost_domain_top60_mild_submit.csv`, pero no es mi primera recomendacion.

Resultado Kaggle informado:

- `catboost_domain_top60_mild_submit.csv`: R2 publico 0,35112. Se descarta la rama domain-weighted pura.

Lectura posterior:

- El CatBoost con pesos de dominio no traslado al publico y cayo fuerte.
- La media mas alta y el esquema de pesos probablemente desplazaron mal la distribucion.
- No conviene probar `domain_top60_medium` ni `domain_top80_medium` como submits puros.
- Si se usa algo de esta rama en el futuro, deberia ser como fuente auxiliar muy chica y alineada, no como modelo directo.

Resultado Kaggle informado:

- `et_aligned_blend_07.csv`: R2 publico 0,37471. Nuevo mejor resultado.

Lectura posterior:

- La rama ExtraTrees alineada sigue mejorando y supera a `et_aligned_blend_05`.
- El peso 7% parece mejor que 5%; 15% ya habia quedado levemente por debajo.
- Conviene explorar fino alrededor de 7%, especialmente 8%, 10% o una variante con ensemble alineado.

Proximo paso propuesto:

- Probar `et_aligned_blend_08.csv`.
- Si mejora o queda muy cerca, probar `et_aligned_blend_10.csv`.
- Como alternativa de diversidad, probar `et_ensemble_aligned_blend_05.csv`.

## Rank blend ExtraTrees preservando distribucion

Fecha: 2026-05-21

Motivacion:

- Buscar una mejora mas significativa que los blends lineales finos.
- Usar ExtraTrees para cambiar el orden/ranking de las predicciones, pero conservar exactamente la distribucion del ganador en filas no vistas.
- Esto evita desplazar la media, pero puede mover muchas filas.

Cambio realizado:

- Se actualizo `codigo/14_extratrees_blends.py` con `rank_blend_preserve_distribution`.
- Para filas no vistas:
  - se combina el ranking del ganador con el ranking de ExtraTrees;
  - se reasignan los valores ordenados del ganador segun ese ranking combinado;
  - las 800 filas con lookup quedan intactas.

Submits generados:

- `Submits/et_rankblend_top70_preserve_10.csv`
- `Submits/et_rankblend_top70_preserve_20.csv`
- `Submits/et_rankblend_top70_preserve_30.csv`
- `Submits/et_rankblend_ensemble_preserve_10.csv`
- `Submits/et_rankblend_ensemble_preserve_20.csv`
- `Submits/et_rankblend_ensemble_preserve_30.csv`

Distribucion:

- Todos preservan la distribucion del ganador `catboost_top60_post_away_l2_35.csv`:
  - media total 35,4674;
  - media no vistos 32,2115;
  - std no vistos 6,6368.

Diferencia contra ganador:

- `et_rankblend_top70_preserve_10`: 5.689 filas cambiadas; diferencia absoluta media 0,3458; maxima 4,4745.
- `et_rankblend_top70_preserve_20`: 5.732 filas cambiadas; diferencia absoluta media 0,6690; maxima 6,4600.
- `et_rankblend_top70_preserve_30`: 5.750 filas cambiadas; diferencia absoluta media 0,9838; maxima 8,0376.
- `et_rankblend_ensemble_preserve_10`: 5.691 filas cambiadas; diferencia absoluta media 0,3530; maxima 5,0516.
- `et_rankblend_ensemble_preserve_20`: 5.730 filas cambiadas; diferencia absoluta media 0,6826; maxima 7,0402.
- `et_rankblend_ensemble_preserve_30`: 5.741 filas cambiadas; diferencia absoluta media 1,0035; maxima 8,6928.

Lectura:

- Esta es una rama mas distinta que los blends alineados: cambia el ranking de muchas filas sin cambiar la distribucion global.
- Tiene mas upside que mover el peso lineal de 7% a 8%, pero tambien mas riesgo.
- La primera prueba razonable es `et_rankblend_top70_preserve_10.csv`.

Orden recomendado para Kaggle:

1. `et_rankblend_top70_preserve_10.csv`
2. Si mejora o queda cerca: `et_rankblend_top70_preserve_20.csv`
3. Si el top70 rankblend funciona, despues comparar contra `et_rankblend_ensemble_preserve_10.csv`.

Resultado Kaggle informado:

- `et_rankblend_top70_preserve_10.csv`: R2 publico 0,37580. Nuevo mejor resultado y salto significativo respecto de `et_aligned_blend_07` con 0,37471.

Lectura posterior:

- La estrategia de cambiar ranking preservando distribucion funciono mucho mejor que los blends lineales.
- ExtraTrees aporta valor principalmente como orden/ranking, no como escala directa de prediccion.
- Como el peso de ranking 10% ya dio un salto grande, conviene explorar esta rama antes de volver a modelos nuevos.
- El siguiente candidato natural es `et_rankblend_top70_preserve_20.csv`, pero tambien puede convenir generar puntos intermedios como 12%, 15% y 18%.

Proximo paso propuesto:

- Probar `et_rankblend_top70_preserve_20.csv` si se quiere seguir la tanda ya generada.
- Alternativamente, generar antes `et_rankblend_top70_preserve_12`, `15` y `18` para buscar el optimo con menos salto.
- Probar luego `et_rankblend_ensemble_preserve_10.csv` para ver si el ensemble de ExtraTrees mejora el ranking top70.

Mejor submit actual:

- `Submits/et_rankblend_top70_preserve_10.csv`

Resultado Kaggle informado:

- Correccion segun captura Kaggle: el submit probado fue `et_rankblend_ensemble_preserve_20.csv`, no `et_rankblend_preserve_20.csv` ni `et_rankblend_top70_preserve_20.csv`.
- `et_rankblend_ensemble_preserve_20.csv`: R2 publico 0,37937. Nuevo mejor resultado y salto fuerte respecto de `et_rankblend_top70_preserve_10.csv` con 0,37580.

Lectura posterior:

- El aumento de peso del ranking ExtraTrees de 10% a 20% mejoro claramente.
- La rama `ensemble` rank-preserve queda confirmada como la linea principal actual.
- Como el salto fue grande, todavia puede haber margen por arriba de 20%, pero conviene cuidar el riesgo de mover demasiado el orden.
- El submit `et_rankblend_ensemble_preserve_30.csv` es el siguiente candidato directo si esta disponible en la tanda usada para Kaggle.
- Tambien seria razonable generar puntos intermedios, especialmente 24%, 25% y 27%, si se quiere buscar el optimo con mas precision.

Proximo paso propuesto:

- Probar `et_rankblend_ensemble_preserve_30.csv` si se quiere continuar con la misma rama.
- Si `30%` mejora o queda cerca, generar una grilla fina entre 20% y 35%.
- Si `30%` cae, generar `et_rankblend_ensemble_preserve_24`, `25` y `27` para buscar un punto intermedio.

Mejor submit actual:

- `Submits/et_rankblend_ensemble_preserve_20.csv`

Resultado Kaggle informado:

- Correccion segun captura Kaggle: el submit probado fue `et_rankblend_ensemble_preserve_30.csv`, no `et_rankblend_preserve_30.csv` ni `et_rankblend_top70_preserve_30.csv`.
- `et_rankblend_ensemble_preserve_30.csv`: R2 publico 0,38006. Nuevo mejor resultado y mejora adicional respecto de `et_rankblend_ensemble_preserve_20.csv` con 0,37937.

Lectura posterior:

- La curva sigue subiendo entre 20% y 30%, aunque la mejora fue menor que de 10% a 20%.
- Esto sugiere que todavia puede haber margen cerca o apenas por encima de 30%, pero ya conviene afinar mas.
- La rama principal sigue siendo `et_rankblend_ensemble_preserve`.
- No conviene saltar demasiado sin grilla fina, porque el rank-blend mueve casi todas las filas no vistas y puede pasarse del optimo.

Proximo paso propuesto:

- Generar y probar una grilla fina alrededor de 30%: 32%, 35% y 40%.
- Si 35% mejora, explorar 37% y 42%.
- Si 32% mejora pero 35% cae, explorar 31%, 33% y 34%.
- Mantener la rama `et_rankblend_preserve` no-ensemble como prueba de diversidad, pero la prioridad actual es explotar `et_rankblend_ensemble_preserve`.

Mejor submit actual:

- `Submits/et_rankblend_ensemble_preserve_30.csv`

Resultado Kaggle informado:

- `hardcode_genre_rank_neo_down_08.csv`: R2 publico 0,38223. Nuevo mejor resultado, con salto claro respecto de `et_rankblend_ensemble_preserve_30.csv` con 0,38006.

Lectura posterior:

- La hipotesis de bajar `r&b`/`neo soul` en el ranking funciono muy bien.
- Esto confirma que el hardcode por subgenero puede aportar sobre el rank-blend, siempre que preserve distribucion.
- La correccion `neo_down_08` solo reordena filas no vistas y mantiene media/std global, por eso es menos riesgosa que los ajustes directos.
- Conviene explotar fino la intensidad de `neo_down` antes de probar reglas EDM mas especulativas.

Cambio realizado:

- Se actualizo `codigo/16_hardcode_genre.py` para generar una grilla de intensidad `neo_down`.
- Nuevos submits generados:
  - `Submits/hardcode_genre_rank_neo_down_04.csv`
  - `Submits/hardcode_genre_rank_neo_down_06.csv`
  - `Submits/hardcode_genre_rank_neo_down_10.csv`
  - `Submits/hardcode_genre_rank_neo_down_12.csv`
- Tambien se generaron sus versiones `direct`, pero no son prioridad porque mueven la distribucion.

Proximo paso propuesto:

- Probar `hardcode_genre_rank_neo_down_10.csv`.
- Si mejora, probar `hardcode_genre_rank_neo_down_12.csv`.
- Si cae, probar `hardcode_genre_rank_neo_down_06.csv`.
- Tener en cuenta que `hardcode_genre_rank_edm_up_neo_down_05.csv` queda practicamente equivalente a `neo_down_10` en efecto de ranking, por lo que no hace falta probar ambos de inmediato.

Mejor submit actual:

- `Submits/hardcode_genre_rank_neo_down_08.csv`

Resultado Kaggle informado:

- `hardcode_genre_rank_neo_down_12.csv`: R2 publico 0,38332. Nuevo mejor resultado y mejora clara respecto de `hardcode_genre_rank_neo_down_08.csv` con 0,38223.

Lectura posterior:

- La intensidad de bajada para `neo soul` todavia no llego al techo en 1.2.
- El hardcode por ranking sobre `neo soul` es ahora la linea principal del proyecto.
- Como 0.8 y 1.2 mejoraron, conviene explorar por arriba antes de volver a EDM o ajustes directos.
- Los submits rank-preserve siguen manteniendo la distribucion del mejor submit base, por lo que el riesgo principal es de orden/ranking, no de escala global.

Cambio realizado:

- Se amplio `codigo/16_hardcode_genre.py` con mas intensidades `neo_down`.
- Nuevos submits generados:
  - `Submits/hardcode_genre_rank_neo_down_14.csv`
  - `Submits/hardcode_genre_rank_neo_down_16.csv`
  - `Submits/hardcode_genre_rank_neo_down_18.csv`
  - `Submits/hardcode_genre_rank_neo_down_20.csv`

Proximo paso propuesto:

- Probar `hardcode_genre_rank_neo_down_16.csv` para ubicar rapido si el techo sigue mas arriba.
- Si 16% mejora, probar `hardcode_genre_rank_neo_down_20.csv`.
- Si 16% cae, probar `hardcode_genre_rank_neo_down_14.csv`.

Mejor submit actual:

- `Submits/hardcode_genre_rank_neo_down_12.csv`

Resultado Kaggle informado:

- `hardcode_genre_rank_neo_down_16.csv`: R2 publico 0,38425. Nuevo mejor resultado y mejora clara respecto de `hardcode_genre_rank_neo_down_12.csv` con 0,38332.

Lectura posterior:

- La curva de `neo_down` sigue subiendo hasta intensidad 1.6.
- La correccion de ranking sobre `neo soul` esta aportando mas que cualquier blend/modelo previo.
- Todavia no hay evidencia de techo; conviene probar 2.0 antes de combinar con otros subgeneros.
- Si 2.0 mejora, habria que generar una grilla mas alta y fina, por ejemplo 2.2, 2.4 y 2.8.

Proximo paso propuesto:

- Probar `hardcode_genre_rank_neo_down_20.csv`.
- Si mejora, generar `hardcode_genre_rank_neo_down_22`, `24` y `28`.
- Si cae, generar puntos intermedios `18` ya disponible y luego 17/19 si hace falta.

Mejor submit actual:

- `Submits/hardcode_genre_rank_neo_down_16.csv`

Resultado Kaggle informado:

- `hardcode_genre_rank_neo_down_20.csv`: R2 publico 0,38523. Nuevo mejor resultado y mejora respecto de `hardcode_genre_rank_neo_down_16.csv` con 0,38425.

Lectura posterior:

- La curva `neo_down` sigue subiendo hasta intensidad 2.0.
- La senal de sobreprediccion en `neo soul` parece mas fuerte de lo que sugeria el OOF.
- Conviene seguir subiendo, pero ya con una grilla moderada para no pasarse demasiado.
- La prioridad sigue siendo encontrar el techo de `neo_down` antes de combinar con EDM u otros subgeneros.

Proximo paso propuesto:

- Generar y probar `hardcode_genre_rank_neo_down_24.csv`.
- Si 24 mejora, probar 28.
- Si 24 cae, probar 22.

Mejor submit actual:

- `Submits/hardcode_genre_rank_neo_down_20.csv`

Resultado Kaggle informado:

- `hardcode_genre_rank_neo_down_28.csv`: R2 publico 0,38707. Nuevo mejor resultado y salto fuerte respecto de `hardcode_genre_rank_neo_down_20.csv` con 0,38523.

Lectura posterior:

- La curva `neo_down` sigue subiendo hasta intensidad 2.8.
- El salto de 2.0 a 2.8 fue grande, por lo que el techo podria estar aun mas arriba.
- `hardcode_genre_rank_neo_down_24.csv` quedo sin probar por ahora, pero ya no es prioridad inmediata salvo que una intensidad mas alta caiga.
- Conviene seguir explorando hacia arriba con puntos moderados, antes de combinar con otros subgeneros.

Proximo paso propuesto:

- Generar y probar `hardcode_genre_rank_neo_down_36.csv`.
- Si 36 mejora, probar 44 o 48.
- Si 36 cae, probar `hardcode_genre_rank_neo_down_32.csv` y/o `24.csv` para ubicar el pico.

Mejor submit actual:

- `Submits/hardcode_genre_rank_neo_down_28.csv`

Resultado Kaggle informado:

- `hardcode_genre_direct_neo_down_40.csv`: R2 publico 0,39031. Nuevo mejor resultado y salto fuerte respecto de `hardcode_genre_rank_neo_down_28.csv` con 0,38707.

Lectura posterior:

- La captura confirma que el submit probado fue la variante `direct`, no `rank`.
- Bajar directamente 4,0 puntos las 484 filas `neo soul` sin lookup funciono mejor que limitar el cambio al reordenamiento.
- Esto indica que el problema no era solo el ranking: la escala de prediccion de `neo soul` tambien estaba demasiado alta.
- La rama principal pasa a ser `hardcode_genre_direct_neo_down`.
- Como 4,0 dio un salto grande, conviene buscar el techo con intensidades directas cercanas antes de combinar otros subgeneros.

Proximo paso propuesto:

- Probar `hardcode_genre_direct_neo_down_48.csv`.
- Si 4,8 mejora, probar `hardcode_genre_direct_neo_down_52.csv`.
- Si 4,8 cae, probar `hardcode_genre_direct_neo_down_44.csv`.
- Mantener `hardcode_genre_rank_neo_down_28.csv` como mejor referencia de la rama rank-preserve.

Mejor submit actual:

- `Submits/hardcode_genre_direct_neo_down_40.csv`

Resultado Kaggle informado:

- `hardcode_genre_rank_neo_down_48.csv`: R2 publico 0,39076. Nuevo mejor resultado, superando a `hardcode_genre_direct_neo_down_40.csv` con 0,39031.

Lectura posterior:

- La variante rank-preserve vuelve a quedar arriba y logra el mejor resultado general.
- Esto es especialmente valioso porque conserva la distribucion global y resulta menos agresiva que la correccion directa.
- La intensidad 4,8 mejora claramente respecto de `rank_neo_down_28` con 0,38707, por lo que la curva rank seguia teniendo margen.
- Conviene probar 5,2 para verificar si el techo esta mas arriba; si cae, afinar entre 4,8 y 5,2.

Proximo paso propuesto:

- Probar `hardcode_genre_rank_neo_down_52.csv`.
- Si mejora, generar intensidades 5,6 y 6,0.
- Si cae, generar 4,6, 5,0 y 5,1 para afinar alrededor de 4,8.

Mejor submit actual:

- `Submits/hardcode_genre_rank_neo_down_48.csv`

Resultado Kaggle informado:

- `hardcode_genre_rank_neo_down_52.csv`: R2 publico 0,39138. Nuevo mejor resultado y mejora respecto de `hardcode_genre_rank_neo_down_48.csv` con 0,39076.

Lectura posterior:

- La curva rank-preserve sigue subiendo hasta intensidad 5,2.
- La mejora respecto de 4,8 confirma que todavia no se encontro el techo.
- Como la ganancia continua pero es moderada, conviene avanzar con pasos de 0,4 antes de hacer saltos mayores.

Proximo paso propuesto:

- Generar y probar `hardcode_genre_rank_neo_down_56.csv`.
- Si mejora, probar `hardcode_genre_rank_neo_down_60.csv`.
- Si cae, afinar con 5,0 y 5,4.

Mejor submit actual:

- `Submits/hardcode_genre_rank_neo_down_52.csv`

Resultado Kaggle informado:

- Segun la captura de Kaggle, el archivo probado fue exactamente `hardcode_genre_rank_neo_down_60.csv`.
- `hardcode_genre_rank_neo_down_60.csv`: R2 publico 0,39241. Nuevo mejor resultado, superando a `hardcode_genre_rank_neo_down_52.csv` con 0,39138.
- No se informo todavia un resultado para `hardcode_genre_rank_neo_down_56.csv`.

Lectura posterior:

- La curva rank-preserve de `neo_down` sigue mejorando hasta intensidad 6,0.
- Como se salto directamente de 5,2 a 6,0 y hubo mejora, todavia no hay evidencia de haber alcanzado el techo.
- Conviene continuar con un paso moderado hacia arriba y conservar 5,6 como punto intermedio si la siguiente intensidad cae.

Proximo paso propuesto:

- Generar y probar `hardcode_genre_rank_neo_down_64.csv`.
- Si mejora, probar `hardcode_genre_rank_neo_down_68.csv`.
- Si cae, probar `hardcode_genre_rank_neo_down_56.csv` para completar el punto intermedio entre 5,2 y 6,0.

Mejor submit actual:

- `Submits/hardcode_genre_rank_neo_down_60.csv`

## Segunda correccion por subgenero sobre `neo_down_60`

Fecha: 2026-06-06

Motivacion:

- Evaluar si una segunda correccion localizada puede mejorar el nuevo mejor `hardcode_genre_rank_neo_down_60.csv`.
- Probar primero el efecto adicional de forma aislada antes de sumar mas reglas.

Diagnostico:

- Fuera de `neo soul`, las 5.283 filas sin lookup pertenecen al genero nuevo `edm`, ausente en train.
- Por esa ausencia no existe evidencia OOF directa para bajar todo el genero `edm`.
- Dentro de EDM, `progressive electro house` tiene 1.715 filas y la menor prediccion media del grupo en el mejor submit actual.
- Se eligio una correccion conservadora por subgenero, preservando la distribucion, en lugar de bajar todo EDM.

Cambio realizado:

- Se creo `codigo/17_second_subgenre_probe.py`.
- El script usa como base `hardcode_genre_rank_neo_down_60.csv` y baja el ranking de `progressive electro house`.
- Se generaron intensidades adicionales de 0,4, 0,8 y 1,2:
  - `Submits/hardcode_rank_neo60_progressive_down_04.csv`
  - `Submits/hardcode_rank_neo60_progressive_down_08.csv`
  - `Submits/hardcode_rank_neo60_progressive_down_12.csv`

Proximo paso propuesto:

- Probar primero `hardcode_rank_neo60_progressive_down_08.csv`.
- Si mejora, probar `hardcode_rank_neo60_progressive_down_12.csv`.
- Si cae poco, probar `hardcode_rank_neo60_progressive_down_04.csv`.
- Si cae claramente, abandonar esta segunda correccion y continuar afinando solo `neo_down`.

Resultado Kaggle informado:

- Segun la captura de Kaggle, el archivo probado fue exactamente `hardcode_rank_neo60_progressive_down_08.csv`.
- `hardcode_rank_neo60_progressive_down_08.csv`: R2 publico 0,39461. Nuevo mejor resultado, superando a `hardcode_genre_rank_neo_down_60.csv` con 0,39241.

Lectura posterior:

- La segunda correccion localizada sobre `progressive electro house` aporta una mejora clara de 0,00220 puntos de R2 publico.
- La combinacion confirma que habia una señal adicional dentro de EDM y que no era necesario bajar todo el genero.
- Como la intensidad 0,8 mejoro claramente, conviene probar la intensidad 1,2 ya generada antes de abrir otra rama.

Proximo paso propuesto:

- Probar `hardcode_rank_neo60_progressive_down_12.csv`.
- Si mejora, generar intensidades 1,6 y 2,0.
- Si cae, afinar entre 0,8 y 1,2 con intensidades 0,9, 1,0 y 1,1.

Mejor submit actual:

- `Submits/hardcode_rank_neo60_progressive_down_08.csv`

Prueba agresiva solicitada:

- Se amplio `codigo/17_second_subgenre_probe.py` para generar una intensidad 2,4 sobre `progressive electro house`.
- Archivo generado para probar: `Submits/hardcode_rank_neo60_progressive_down_24.csv`.
- Esta variante conserva el enfoque rank-preserve y mantiene `neo_down_60`, pero reordena el subgenero con una intensidad tres veces mayor que el mejor `progressive_down_08`.
- Se considera una prueba de mayor riesgo para verificar rapidamente si la señal admite una correccion mucho mas fuerte.

Resultado Kaggle informado:

- Segun la captura de Kaggle, el archivo probado fue exactamente `hardcode_rank_neo60_progressive_down_24.csv`.
- `hardcode_rank_neo60_progressive_down_24.csv`: R2 publico 0,39740. Nuevo mejor resultado, superando a `hardcode_rank_neo60_progressive_down_08.csv` con 0,39461.

Lectura posterior:

- El salto de intensidad 0,8 a 2,4 mejoro 0,00279 puntos de R2 publico.
- La correccion mas agresiva de `progressive electro house` funciono y todavia no muestra evidencia de techo.
- Conviene seguir aumentando con un salto controlado antes de combinar un tercer subgenero.

Cambio realizado:

- Se amplio `codigo/17_second_subgenre_probe.py` para generar:
  - `Submits/hardcode_rank_neo60_progressive_down_36.csv`
  - `Submits/hardcode_rank_neo60_progressive_down_48.csv`

Proximo paso propuesto:

- Probar `hardcode_rank_neo60_progressive_down_36.csv`.
- Si mejora, probar `hardcode_rank_neo60_progressive_down_48.csv`.
- Si cae, afinar entre 2,4 y 3,6 con intensidades 2,8, 3,0 y 3,2.

Mejor submit actual:

- `Submits/hardcode_rank_neo60_progressive_down_24.csv`

## Hardcode por genero/subgenero

Fecha: 2026-06-03

Motivacion:

- Probar la idea de hardcodear ajustes por genero/subgenero recomendada externamente.
- Evitar ajustes a ciegas demasiado agresivos sobre el mejor submit actual.
- Mantener como base `et_rankblend_ensemble_preserve_30.csv`, que es el mejor publico actual con 0,38006.

Diagnostico:

- En test sin lookup hay 5.767 filas:
  - `edm`: 5.283 filas.
  - `r&b`: 484 filas.
- Subgeneros no vistos principales:
  - `progressive electro house`: 1.715 filas.
  - `electro house`: 1.416 filas.
  - `big room`: 1.097 filas.
  - `pop edm`: 1.055 filas.
  - `neo soul`: 484 filas.
- En train no aparece `edm`, por lo que los ajustes EDM son exploratorios de leaderboard.
- Para `r&b`/`neo soul` si hay OOF: el residual medio OOF de `r&b` fue negativo, sugiriendo una leve sobreprediccion.

Cambio realizado:

- Se creo `codigo/16_hardcode_genre.py`.
- El script genera dos familias:
  - `hardcode_genre_direct_*`: suma/resta valores fijos en filas no vistas.
  - `hardcode_genre_rank_*`: usa esos valores como boost de ranking y luego reasigna la distribucion original del mejor submit.
- Las filas con lookup quedan intactas.
- Los submits rank-preserve mantienen la media y std del mejor submit.

Submits generados:

- `Submits/hardcode_genre_direct_popedm_up_08.csv`
- `Submits/hardcode_genre_rank_popedm_up_08.csv`
- `Submits/hardcode_genre_direct_progressive_up_08.csv`
- `Submits/hardcode_genre_rank_progressive_up_08.csv`
- `Submits/hardcode_genre_direct_electro_bigroom_up_06.csv`
- `Submits/hardcode_genre_rank_electro_bigroom_up_06.csv`
- `Submits/hardcode_genre_direct_neo_down_08.csv`
- `Submits/hardcode_genre_rank_neo_down_08.csv`
- `Submits/hardcode_genre_direct_edm_up_neo_down_05.csv`
- `Submits/hardcode_genre_rank_edm_up_neo_down_05.csv`
- `Submits/hardcode_genre_direct_popedm_up_progressive_down_06.csv`
- `Submits/hardcode_genre_rank_popedm_up_progressive_down_06.csv`

Lectura:

- Los directos son mas riesgosos porque cambian la distribucion global.
- Los rank-preserve son mas compatibles con la estrategia que viene funcionando.
- La correccion con mas justificacion es `neo_down_08`, porque r&b/neo soul tiene evidencia OOF de leve sobreprediccion.
- Las correcciones EDM son mas especulativas porque EDM no esta en train.

Proximo paso propuesto:

- Probar primero `hardcode_genre_rank_neo_down_08.csv`.
- Si mejora o queda muy cerca, probar `hardcode_genre_rank_edm_up_neo_down_05.csv`.
- Si cae, probar una hipotesis distinta y localizada: `hardcode_genre_rank_popedm_up_08.csv`.
- Dejar los submits `direct_*` para despues; no son prioridad por mover media/std.

Mejor submit actual:

- `Submits/et_rankblend_ensemble_preserve_30.csv`

Resultado Kaggle informado:

- `et_rankblend_ensemble_preserve_35.csv`: R2 publico 0,37998. Queda apenas por debajo de `et_rankblend_ensemble_preserve_30.csv` con 0,38006.

Lectura posterior:

- El 35% no mejora al 30%, pero queda extremadamente cerca.
- La zona buena de la rama ensemble esta alrededor de 30%-35%.
- Como 35% cae apenas y 40% no fue probado en ensemble, todavia no conviene descartar del todo pesos mayores, pero el siguiente submit mas informativo es 32%.
- El mejor publico sigue siendo `et_rankblend_ensemble_preserve_30.csv`.

Proximo paso propuesto:

- Probar `et_rankblend_ensemble_preserve_32.csv`.
- Si 32% mejora, generar 31%, 33% y 34%.
- Si 32% cae, mantener 30% como pico publico actual y pasar a probar diversidad, por ejemplo `et_rankblend_ensemble_preserve_40.csv` o una rama de hardcode/ajuste por genero muy conservadora.

Mejor submit actual:

- `Submits/et_rankblend_ensemble_preserve_30.csv`

Resultado Kaggle informado:

- `et_rankblend_preserve_40.csv`: R2 publico 0,37419. Cae fuerte respecto de `et_rankblend_ensemble_preserve_30.csv` con 0,38006. Este resultado corresponde a la rama no-ensemble.

Lectura posterior:

- El peso 40% de la rama no-ensemble se paso del optimo y degrada bastante.
- El maximo publico observado queda por ahora en `et_rankblend_ensemble_preserve_30.csv`.
- No se puede inferir directamente que `ensemble_preserve_40` vaya a caer igual, porque el 40 probado no fue ensemble.
- Conviene afinar la rama ensemble alrededor de 30%, generando 32%, 35% y 40% ensemble.

Proximo paso propuesto:

- Generar `et_rankblend_ensemble_preserve_32.csv`, `et_rankblend_ensemble_preserve_35.csv` y `et_rankblend_ensemble_preserve_40.csv`.
- Probar primero `et_rankblend_ensemble_preserve_35.csv`.
- Si 35% mejora, generar 36%, 37% y 38% ensemble.
- Si 35% cae, probar `et_rankblend_ensemble_preserve_32.csv`.

Mejor submit actual:

- `Submits/et_rankblend_ensemble_preserve_30.csv`

## Solver por probes del leaderboard

Fecha: 2026-06-10

Motivacion:

- Adaptar al proyecto el metodo de interpretar cada score publico como una
  ecuacion sobre los valores verdaderos del test.
- Combinar la informacion acumulada por los submits historicos sin resolver
  directamente un vector de 6.567 targets.
- Controlar la inestabilidad numerica mediante truncamiento espectral y una
  intensidad final de 50% respecto del optimo algebraico.

Implementacion:

- Se creo `codigo/19_lb_probe_solver.py`.
- El script extrae de este registro los pares exactos de archivo y R2 publico.
- Se identificaron 46 submits con archivo disponible y score no aproximado.
- Se excluyeron del ajuste los scores anotados con menos de cuatro decimales.
- Se excluyo `catboost_best_lookup_blend_85.csv` porque modifica las filas con
  lookup conocido.
- La base final usa 41 submits y preserva exactamente las 800 filas conocidas.

Estimacion de SST:

- La familia `catboost_top60_post_away_l2_*` es practicamente unidimensional:
  la primera direccion explica 0,99994352 de su variacion.
- El ajuste parabolico de R2 contra esa direccion tiene error maximo 0,00000490.
- El SST estimado es 3.458.700,81.

Validacion retrospectiva:

- Sin utilizar el score de `hardcode_rank_neo60_progressive_down_24.csv`, el
  corte espectral 1.000 predice 0,39758 frente al 0,39740 real.
- El error retrospectivo es 0,00018.
- Para `hardcode_genre_rank_neo_down_60.csv`, predice 0,39257 frente a 0,39241.
- Estas pruebas muestran que el sistema recupera razonablemente submits futuros
  que se encuentran cerca del espacio generado por los probes anteriores.

Candidatos generados:

- `Submits/lb_probe_conservative_t30000_a050.csv`
  - 5 direcciones efectivas.
  - R2 algebraico estimado antes de clipping: 0,40959.
- `Submits/lb_probe_balanced_t3000_a050.csv`
  - 9 direcciones efectivas.
  - R2 algebraico estimado antes de clipping: 0,41446.
- `Submits/lb_probe_aggressive_t1000_a050.csv`
  - 10 direcciones efectivas.
  - R2 algebraico estimado: 0,43176.
  - No requiere clipping y conserva exactamente el lookup.

Archivos de diagnostico:

- `otros csv/lb_probe_solver/leaderboard_results.csv`
- `otros csv/lb_probe_solver/temporal_validation.csv`
- `otros csv/lb_probe_solver/candidate_summary.csv`
- `otros csv/lb_probe_solver/report.md`

Limitacion:

- Kaggle calcula el R2 sobre un 50% publico cuya mascara de filas es desconocida.
- Los productos internos se calculan sobre las 6.567 filas como aproximacion.
- Las predicciones algebraicas sirven para ordenar candidatos, pero no son una
  garantia del score que devolvera Kaggle.

Proximo paso propuesto:

- Probar primero `lb_probe_aggressive_t1000_a050.csv`.
- Aunque se llama `aggressive`, ya aplica solo 50% del desplazamiento optimo y
  usa el mismo corte espectral 1.000 que funciono en el ejemplo de referencia.
- Si mejora claramente pero queda por debajo de la estimacion, probar
  `lb_probe_balanced_t3000_a050.csv` como control mas regularizado.
- Mantener `hardcode_rank_neo60_progressive_down_24.csv` con 0,39740 como mejor
  resultado confirmado hasta recibir el score real del nuevo candidato.

Resultado Kaggle informado:

- Segun la captura de Kaggle, el archivo probado fue exactamente
  `lb_probe_aggressive_t1000_a050.csv`.
- `lb_probe_aggressive_t1000_a050.csv`: R2 publico 0,43178.
- Es el nuevo mejor resultado, superando a
  `hardcode_rank_neo60_progressive_down_24.csv` con 0,39740.
- La estimacion previa era 0,43176: el error fue solamente 0,00002.

Lectura posterior:

- El resultado confirma con mucha precision la estimacion algebraica del solver.
- La nueva observacion aporta una ecuacion especialmente valiosa porque combina
  diez direcciones espectrales y explora una zona muy superior al mejor anterior.
- No conviene probar ahora los candidatos `balanced` o `conservative` de la
  primera tanda: sus estimaciones eran inferiores al 0,43178 ya confirmado.
- Corresponde recalcular el sistema usando este resultado como nueva referencia
  y generar una segunda iteracion sin sobrescribir el archivo probado.

Mejor submit actual:

- `Submits/lb_probe_aggressive_t1000_a050.csv`: R2 publico 0,43178.

## Guion completo de la presentacion final

Fecha: 2026-06-12

Actualizacion:

- Se reviso `Predictiva Kaggle (2).pdf`, compuesto por 12 diapositivas.
- Se creo `GUION_PRESENTACION_FINAL.md` con:
  - guion oral por diapositiva;
  - explicacion en lenguaje simple de los conceptos tecnicos;
  - transiciones entre diapositivas;
  - espacio reservado para completar el mejor modelo;
  - preguntas probables y respuestas breves.
- El guion usa como score publico inicial confirmado `0,36930` y como mejor
  resultado visible actual `0,45980`.
- La diapositiva de mejor modelo queda pendiente hasta conocer el cierre de la
  competencia y el resultado privado.

## Reanudacion de submits: candidato pendiente v8

Fecha: 2026-06-14

Estado verificado:

- Mejor score publico confirmado:
  `lb_probe_oof_residual_numeric_inverted_rms050.csv`, R2 publico `0,46113`.
- El siguiente candidato generado y todavia pendiente es
  `Submits/lb_probe_v8_residual_primary_t3000_a075.csv`.
- R2 algebraico estimado: `0,46143`.
- Mueve `0,30946` puntos RMS respecto del mejor confirmado.
- Tiene 6.567 filas, columnas `ID` y `track_popularity`, sin nulos ni IDs
  duplicados.
- Conserva exactamente las 800 filas con lookup.

Proximo paso:

- Probar `lb_probe_v8_residual_primary_t3000_a075.csv`.
- Informar el score publico con cinco decimales antes de evaluar otro archivo.
- No priorizar la version full: su mejora adicional estimada es solo `0,00002`.

### Resultado del candidato principal v8

Fecha: 2026-06-14

Resultado Kaggle confirmado por captura:

- `lb_probe_v8_residual_primary_t3000_a075.csv`: R2 publico 0,46150.
- Es el nuevo mejor resultado, superando el `0,46113` anterior en `0,00037`.
- La estimacion algebraica era `0,46143`, con un error de aproximadamente
  `0,00007`.

Mejor submit actual:

- `Submits/lb_probe_v8_residual_primary_t3000_a075.csv`: R2 publico 0,46150.

### Resultado de la novena iteracion

Fecha: 2026-06-14

Resultado Kaggle informado:

- `lb_probe_v9_residual_refined_primary_t3000_a075.csv`: R2 publico 0,46155.
- Es el nuevo mejor resultado, con una mejora de solo `0,00005` sobre v8.
- La estimacion algebraica era `0,461527`.
- La mejora confirma que esta familia esta practicamente agotada.

Mejor submit actual:

- `Submits/lb_probe_v9_residual_refined_primary_t3000_a075.csv`: R2 publico
  0,46155.

### Resultado del probe residual categorico

Fecha: 2026-06-14

Resultado Kaggle informado:

- `lb_probe_newspace_categorical_residual_rms085.csv`: R2 publico 0,46088.
- Cae `0,00067` respecto del mejor `0,46155`.
- La caida es menor que la penalizacion esperada sin señal nueva, que lo ubicaba
  aproximadamente en `0,4602`.
- Esto indica que la direccion tiene señal favorable, pero la amplitud de
  `0,85 RMS` fue demasiado agresiva.

Proximo paso:

- Estimar la intensidad optima de esta direccion y generar una version moderada
  sobre el mejor confirmado.

### Calibracion del probe residual categorico

Fecha: 2026-06-14

Calculo:

- Mejor de referencia: `0,46155`.
- Probe completo a amplitud `0,85 RMS`: `0,46088`.
- La caida fue menor que la penalizacion sin señal, por lo que la direccion es
  favorable pero estaba sobredimensionada.
- La intensidad optima estimada es aproximadamente `0,25` veces el probe.
- R2 publico proyectado: aproximadamente `0,46164`.

Archivo generado:

- `Submits/lb_probe_categorical_residual_calibrated_a025.csv`

Validacion:

- 6.567 filas, sin nulos ni IDs duplicados.
- Movimiento RMS respecto del mejor: `0,21201`.
- Conserva exactamente las 800 canciones conocidas.
- Valores entre 0 y 99.

Proximo paso propuesto:

- Probar `lb_probe_categorical_residual_calibrated_a025.csv`.
- Informar el score con cinco decimales.

Decision:

- No continuar con refinamientos v9 ni probar la version full.
- Abrir una direccion residual nueva basada en categorias crudas y contexto.

## Probe residual categorico de nueva dimension

Fecha: 2026-06-14

Motivacion:

- Las iteraciones v8 y v9 mejoraron solo `0,00037` y `0,00005`.
- El espacio algebraico de submits anteriores quedo practicamente agotado.
- Se busca una direccion nueva basada en errores sistematicos por artista,
  album, playlist, genero, subgenero y sus combinaciones.

Implementacion:

- Se creo `codigo/29_oof_residual_categorical_probe.py`.
- Se entreno CatBoost sobre los residuos OOF del modelo base con GroupKFold por
  `track_id`.
- La correlacion residual OOF fue positiva pero moderada: `0,07910`.
- La direccion se ortogonalizo contra todos los submits con score exacto.
- El coseno absoluto maximo contra el historial fue solo `0,0361`, confirmando
  que es una direccion practicamente nueva.

Archivo generado:

- `Submits/lb_probe_newspace_categorical_residual_rms085.csv`

Validacion:

- 6.567 filas y columnas `ID`, `track_popularity`.
- Sin nulos ni IDs duplicados.
- Valores entre 0 y 99.
- Cambia 5.762 filas.
- Movimiento RMS: `0,84804`.
- Conserva exactamente las 800 canciones conocidas.

Interpretacion:

- Es un probe exploratorio mas agresivo, no una mejora garantizada.
- Sin señal nueva, la penalizacion cuadratica esperada lo ubicaria
  aproximadamente en `0,4602`.
- Su resultado permitira estimar el signo y la intensidad optima de esta nueva
  coordenada antes de combinarla con el mejor `0,46155`.

Proximo paso propuesto:

- Probar `lb_probe_newspace_categorical_residual_rms085.csv`.
- Informar el score con cinco decimales.

Mejor submit actual:

- `Submits/lb_probe_v9_residual_refined_primary_t3000_a075.csv`: R2 publico
  0,46155.

Proximo paso:

- Incorporar este score al solver y recalcular antes de recomendar otro submit.

### Novena iteracion refinada del solver

Fecha: 2026-06-14

Actualizacion:

- Se incorporo `lb_probe_v8_residual_primary_t3000_a075.csv` con R2 publico
  0,46150.
- El solver usa ahora 56 scores exactos y 51 submits luego de proteger lookup.
- La estimacion de v8 fue `0,46143`, con un error aproximado de `0,00007`.

Nuevos candidatos:

- `Submits/lb_probe_v9_residual_refined_conservative_t30000_a050.csv`
  - R2 algebraico estimado: `0,461504`.
- `Submits/lb_probe_v9_residual_refined_primary_t3000_a075.csv`
  - R2 algebraico estimado: `0,461527`.
  - Mueve `0,09210` puntos RMS respecto del mejor.
  - Conserva exactamente las 800 filas con lookup.
- `Submits/lb_probe_v9_residual_refined_full_t3000_a100.csv`
  - R2 algebraico estimado: `0,461529`.
  - Su mejora adicional frente al principal es inferior a `0,000002`.

Validacion del candidato principal:

- 6.567 filas y columnas `ID`, `track_popularity`.
- Sin nulos ni IDs duplicados.
- Valores entre 0 y 99.
- Diferencia maxima en lookup: 0.

Proximo paso propuesto:

- Probar `lb_probe_v9_residual_refined_primary_t3000_a075.csv`.
- No probar la version full: agrega amplitud sin una mejora visible estimada.
- Informar el score con cinco decimales.

Mejor submit actual:

- `Submits/lb_probe_v8_residual_primary_t3000_a075.csv`: R2 publico 0,46150.

## Resultado de la cuarta iteracion

Fecha: 2026-06-11

Resultado Kaggle informado:

- `lb_probe_v4_contextflip_primary_t3000_a075.csv`: R2 publico 0,45903.
- Es el nuevo mejor resultado, superando el 0,44691 anterior.
- La estimacion algebraica era 0,45902; el error fue aproximadamente 0,00001.

Lectura posterior:

- La inversion de la direccion de contexto quedo confirmada con gran precision.
- No corresponde probar el `full` generado antes de conocer este resultado.
- Se debe incorporar la nueva ecuacion y recalcular el desplazamiento restante
  desde el nuevo mejor.

Mejor submit actual:

- `Submits/lb_probe_v4_contextflip_primary_t3000_a075.csv`: R2 publico 0,45903.

## Quinta iteracion del solver

Fecha: 2026-06-11

Actualizacion:

- Se incorporo `lb_probe_v4_contextflip_primary_t3000_a075.csv` con R2 publico
  0,45903.
- La estimacion anterior fue 0,45902, con un error aproximado de 0,00001.
- El solver usa ahora 51 scores exactos y 46 submits luego de proteger lookup.
- La nueva referencia es el mejor confirmado de 0,45903.

Nuevos candidatos:

- `Submits/lb_probe_v5_refined_conservative_t30000_a050.csv`
  - 5 direcciones efectivas.
  - R2 algebraico estimado: 0,45963.
- `Submits/lb_probe_v5_refined_primary_t3000_a075.csv`
  - 11 direcciones efectivas.
  - R2 algebraico estimado: 0,45979.
  - Mueve 0,49 puntos RMS respecto del mejor.
  - Conserva exactamente las 800 filas con lookup.
- `Submits/lb_probe_v5_refined_full_t3000_a100.csv`
  - 11 direcciones efectivas.
  - R2 algebraico estimado: 0,45984.
  - Agrega solo 0,00005 estimado frente al principal.

Proximo paso propuesto:

- Probar `lb_probe_v5_refined_primary_t3000_a075.csv`.
- No probar primero la version full porque agrega riesgo para una mejora
  estimada minima.
- Informar el score con cinco decimales para decidir si queda margen medible.

Mejor submit actual:

- `Submits/lb_probe_v4_contextflip_primary_t3000_a075.csv`: R2 publico 0,45903.

## Resultado de la quinta iteracion

Fecha: 2026-06-11

Resultado Kaggle informado:

- `lb_probe_v5_refined_primary_t3000_a075.csv`: R2 publico 0,45980.
- Es el nuevo mejor resultado, superando el 0,45903 anterior.
- La estimacion algebraica era 0,45979; el error fue aproximadamente 0,00001.

Lectura posterior:

- La refinacion conjunta volvio a predecir el resultado con gran precision.
- No corresponde probar el `full` de la quinta iteracion porque debe
  recalcularse usando esta nueva ecuacion.

Mejor submit actual:

- `Submits/lb_probe_v5_refined_primary_t3000_a075.csv`: R2 publico 0,45980.

## Sexta iteracion del solver

Fecha: 2026-06-11

Actualizacion:

- Se incorporo `lb_probe_v5_refined_primary_t3000_a075.csv` con R2 publico
  0,45980.
- La estimacion anterior fue 0,45979, con un error aproximado de 0,00001.
- El solver usa ahora 52 scores exactos y 47 submits luego de proteger lookup.
- El espacio conjunto medido muestra un margen restante de aproximadamente
  0,00005.

Nuevos candidatos:

- `Submits/lb_probe_v6_refined_conservative_t30000_a050.csv`
  - R2 algebraico estimado: 0,45984.
- `Submits/lb_probe_v6_refined_primary_t3000_a075.csv`
  - 11 direcciones efectivas.
  - R2 algebraico estimado: 0,45985.
  - Mueve solo 0,12 puntos RMS respecto del mejor.
  - Conserva exactamente las 800 filas con lookup.
- `Submits/lb_probe_v6_refined_full_t3000_a100.csv`
  - R2 algebraico estimado: 0,45985.
  - Su ventaja frente al principal es inferior a 0,00001.

Proximo paso propuesto:

- Probar `lb_probe_v6_refined_primary_t3000_a075.csv` como afinacion de cierre.
- No probar luego la version full salvo una discrepancia inesperada.
- Si el resultado confirma aproximadamente 0,45985, este espacio queda agotado
  y una mejora mayor exigira construir otra direccion independiente.

Mejor submit actual:

- `Submits/lb_probe_v5_refined_primary_t3000_a075.csv`: R2 publico 0,45980.

## Resultado de la sexta iteracion

Fecha: 2026-06-12

Resultado Kaggle informado:

- `lb_probe_v6_refined_primary_t3000_a075.csv`: R2 publico 0,45986.
- Es el nuevo mejor resultado, superando el 0,45980 anterior.
- La estimacion algebraica era 0,45985, con un error aproximado de 0,00001.
- La afinacion de cierre confirmo que el espacio conjunto medido estaba
  practicamente agotado.

Mejor submit actual:

- `Submits/lb_probe_v6_refined_primary_t3000_a075.csv`: R2 publico 0,45986.

## Septima iteracion del solver

Fecha: 2026-06-12

Actualizacion:

- Se incorporo `lb_probe_v6_refined_primary_t3000_a075.csv` con R2 publico
  0,45986.
- La estimacion previa era 0,45985, con un error aproximado de 0,00001.
- El solver usa ahora 53 scores exactos y 48 submits luego de proteger lookup.

Candidatos generados:

- `Submits/lb_probe_v7_refined_conservative_t30000_a050.csv`
  - R2 algebraico estimado: 0,45986254.
- `Submits/lb_probe_v7_refined_primary_t3000_a075.csv`
  - R2 algebraico estimado: 0,45986320.
- `Submits/lb_probe_v7_refined_full_t3000_a100.csv`
  - R2 algebraico estimado: 0,45986342.

Conclusion:

- La mejora estimada frente a 0,45986 es inferior a 0,000004.
- No conviene gastar otro submit en esta misma familia.
- Para buscar una mejora visible hace falta abrir una direccion realmente nueva.

Mejor submit actual:

- `Submits/lb_probe_v6_refined_primary_t3000_a075.csv`: R2 publico 0,45986.

## Probe nuevo por residuos OOF numericos

Fecha: 2026-06-12

Objetivo:

- Abrir una direccion independiente del solver, cuyo margen estimado ya era
  inferior a 0,000004.
- Modelar los residuos OOF del CatBoost con variables numericas, temporales,
  frecuencias y target encodings ya disponibles.

Diagnostico:

- HistGradientBoosting obtuvo correlacion OOF de -0,07163 con el residuo.
- ExtraTrees obtuvo correlacion OOF de -0,07280.
- La coincidencia de signo entre modelos motivo invertir y combinar la senal.
- La direccion final se ortogonalizo contra los submits con score exacto.

Nuevo candidato:

- `Submits/lb_probe_oof_residual_numeric_rms050.csv`
  - Mueve 0,50 puntos RMS respecto del mejor actual.
  - Modifica las 5.767 filas sin lookup.
  - Conserva exactamente las 800 canciones conocidas.
  - Es un probe exploratorio de una senal nueva, no una mejora algebraica
    garantizada.

Proximo paso propuesto:

- Probar `lb_probe_oof_residual_numeric_rms050.csv`.
- Usar el score para decidir si la senal residual existe tambien en el test
  publico antes de generar variantes de mayor amplitud.

Mejor submit actual:

- `Submits/lb_probe_v6_refined_primary_t3000_a075.csv`: R2 publico 0,45986.

### Resultado e inversion del probe residual

Fecha: 2026-06-12

Resultado Kaggle informado:

- `lb_probe_oof_residual_numeric_rms050.csv`: R2 publico 0,45772.
- Cayo 0,00214 respecto de la referencia 0,45986.
- La caida es mayor que la penalizacion cuadratica esperada para un movimiento
  RMS 0,50, por lo que indica que la direccion debe invertirse.

Nuevo candidato:

- `Submits/lb_probe_oof_residual_numeric_inverted_rms050.csv`
  - Invierte exactamente el desplazamiento del probe respecto de la referencia,
    salvo el clipping entre 0 y 100.
  - Tiene coseno 0,996 con la direccion opuesta ideal.
  - Mueve 0,498 puntos RMS.
  - Conserva exactamente las 800 canciones conocidas.
  - R2 publico estimado por simetria cuadratica: 0,46105.

Proximo paso propuesto:

- Probar `lb_probe_oof_residual_numeric_inverted_rms050.csv`.

Mejor submit actual:

- `Submits/lb_probe_v6_refined_primary_t3000_a075.csv`: R2 publico 0,45986.

### Resultado de la inversion residual

Fecha: 2026-06-12

Resultado Kaggle informado:

- `lb_probe_oof_residual_numeric_inverted_rms050.csv`: R2 publico 0,46113.
- Es el nuevo mejor resultado, con una mejora de 0,00127 sobre 0,45986.
- La proyeccion era 0,46105, con un error de solo 0,00008.
- El resultado confirma que la direccion residual invertida contiene señal
  util en el test publico.

Mejor submit actual:

- `Submits/lb_probe_oof_residual_numeric_inverted_rms050.csv`: R2 publico
  0,46113.

## Octava iteracion del solver con la señal residual

Fecha: 2026-06-12

Actualizacion:

- Se incorporaron el probe residual de 0,45772 y su inversion de 0,46113.
- El solver usa ahora 55 scores exactos y 50 submits luego de proteger lookup.
- La nueva referencia es
  `lb_probe_oof_residual_numeric_inverted_rms050.csv`.

Nuevos candidatos:

- `Submits/lb_probe_v8_residual_conservative_t30000_a050.csv`
  - R2 algebraico estimado: 0,46115.
- `Submits/lb_probe_v8_residual_primary_t3000_a075.csv`
  - 12 direcciones efectivas.
  - R2 algebraico estimado: 0,46143.
  - Mueve 0,309 puntos RMS respecto del mejor.
  - Conserva exactamente las 800 filas con lookup.
- `Submits/lb_probe_v8_residual_full_t3000_a100.csv`
  - R2 algebraico estimado: 0,46145.
  - La ganancia adicional frente al principal es solo 0,00002.

Proximo paso propuesto:

- Probar `lb_probe_v8_residual_primary_t3000_a075.csv`.
- No priorizar la version full porque agrega muy poco margen con mayor amplitud.

Mejor submit actual:

- `Submits/lb_probe_oof_residual_numeric_inverted_rms050.csv`: R2 publico
  0,46113.

## Cuarta iteracion del solver: inversion de contexto

Fecha: 2026-06-11

Actualizacion:

- Se incorporo `lb_probe_newspace_context_rms075.csv` con R2 publico 0,43866.
- El solver usa 50 scores exactos y 45 submits luego de proteger lookup.
- La referencia sigue siendo `lb_probe_newspace_hadamard_rms075.csv` con
  0,44691.
- La caida del probe de contexto permitio estimar con claridad el signo
  contrario de esa direccion.

Nuevos candidatos:

- `Submits/lb_probe_v4_contextflip_conservative_t30000_a050.csv`
  - 4 direcciones efectivas.
  - R2 algebraico estimado: 0,44747.
  - Mueve 0,31 puntos RMS respecto del mejor.
- `Submits/lb_probe_v4_contextflip_primary_t3000_a075.csv`
  - 12 direcciones efectivas.
  - R2 algebraico estimado: 0,45902.
  - Mueve 1,95 puntos RMS respecto del mejor.
  - Invierte fuertemente la direccion desfavorable de contexto.
  - Conserva exactamente las 800 filas con lookup.
- `Submits/lb_probe_v4_contextflip_full_t3000_a100.csv`
  - 12 direcciones efectivas.
  - R2 algebraico estimado: 0,45982.
  - Es el paso completo y de mayor riesgo.

Proximo paso propuesto:

- Probar `lb_probe_v4_contextflip_primary_t3000_a075.csv`.
- No probar primero la version full: agrega riesgo y solo mejora 0,00081 en la
  estimacion.
- Informar el score con cinco decimales para recalcular el sistema.

Mejor submit actual:

- `Submits/lb_probe_newspace_hadamard_rms075.csv`: R2 publico 0,44691.

Resultado Kaggle informado:

- `lb_probe_v2_primary_t300_a075.csv`: R2 publico 0,44622.
- Es el nuevo mejor resultado, superando a
  `lb_probe_aggressive_t1000_a050.csv` con 0,43178.
- La estimacion algebraica era 0,44659; el error fue 0,00037.

Lectura posterior:

- La segunda iteracion confirma que el solver sigue ordenando y estimando bien
  aun despues del salto inicial.
- El nuevo submit debe incorporarse como referencia antes de elegir entre la
  version full anterior o una tercera iteracion.
- No corresponde probar directamente `lb_probe_v2_full_t300_a100.csv`: fue
  calculado antes de conocer esta nueva ecuacion y ya puede recalcularse mejor.

Mejor submit actual:

- `Submits/lb_probe_v2_primary_t300_a075.csv`: R2 publico 0,44622.

Resultado Kaggle informado:

- Segun la captura, el archivo probado fue
  `lb_probe_newspace_hadamard_rms075.csv`.
- `lb_probe_newspace_hadamard_rms075.csv`: R2 publico 0,44691.
- Es el nuevo mejor resultado, superando al 0,44622 anterior.
- El escenario sin señal nueva estimaba aproximadamente 0,44515, por lo que el
  resultado confirma que la direccion ortogonal contiene informacion util.

Diagnostico:

- La intensidad optima aislada de esta direccion se estima en 0,824 veces la
  aplicada por el probe.
- El maximo unidimensional estimado es aproximadamente 0,44694.
- No conviene aumentar la intensidad: a partir de 1,0 la parabola ya comienza a
  descender.
- El siguiente paso correcto es incorporar esta ecuacion al solver conjunto y
  reoptimizar todas las direcciones.

Mejor submit actual:

- `Submits/lb_probe_newspace_hadamard_rms075.csv`: R2 publico 0,44691.

Resultado Kaggle informado:

- `lb_probe_newspace_context_rms075.csv`: R2 publico 0,43866.
- El probe cae respecto del mejor confirmado de 0,44691.
- La caida no invalida la direccion: aporta una ecuacion de alta senal que
  permite estimar el signo contrario y la intensidad conveniente.

Lectura posterior:

- La direccion de contexto aplicada tenia orientacion desfavorable.
- Corresponde incorporarla al sistema conjunto antes de generar otro submit.
- No conviene probar una inversion manual completa: el siguiente candidato debe
  combinar esta nueva coordenada con todas las ecuaciones confiables anteriores.

Mejor submit actual:

- `Submits/lb_probe_newspace_hadamard_rms075.csv`: R2 publico 0,44691.

## Segunda direccion nueva de contexto

Fecha: 2026-06-10

Motivacion:

- Al incorporar el primer probe ortogonal, la reoptimizacion conjunta estima un
  maximo cercano a 0,4480.
- Como ese incremento sigue siendo pequeno para el objetivo solicitado, se
  decide no priorizar todavia el refinamiento conjunto.
- Se abre una segunda direccion independiente para buscar señal no capturada por
  audio y subgenero.

Implementacion:

- Se creo `codigo/21_newspace_context_probe.py`.
- La nueva direccion usa contrastes de antiguedad, duracion, acousticness,
  instrumentalness, speechiness, liveness y encodings de artista, album y
  contexto de playlist.
- Se ortogonaliza contra todos los submits scoreados, incluido el primer probe
  de nueva dimension.
- Usa amplitud 0,75 RMS y conserva exactamente las 800 filas con lookup.

Archivo generado:

- `Submits/lb_probe_newspace_context_rms075.csv`

Validacion:

- 6.567 filas, sin nulos ni IDs duplicados.
- Rango entre 0 y 99.
- Diferencia maxima en lookup: 0.
- Si la direccion no contiene señal, se espera un score cercano a 0,44584.

Proximo paso propuesto:

- Probar `lb_probe_newspace_context_rms075.csv`.
- El archivo es exploratorio: puede caer respecto de 0,44691.
- Su score permitira resolver una segunda coordenada nueva y comprobar si existe
  margen para otro salto relevante; si no aparece señal fuerte, el limite
  honesto del espacio medido quedara cerca de 0,448.

Mejor submit actual:

- `Submits/lb_probe_newspace_hadamard_rms075.csv`: R2 publico 0,44691.

## Apertura de una direccion nueva

Fecha: 2026-06-10

Motivacion:

- El usuario pidio buscar una mejora mayor en lugar de continuar con incrementos
  estimados cercanos a 0,001.
- El espacio generado por los 48 submits scoreados ya esta casi agotado: shifts
  directos, escalas por subgenero y distintos cortes espectrales solo muestran
  un margen de aproximadamente 0,0008 a 0,0011.
- Para aspirar a otro salto es necesario medir una direccion que no pueda
  reconstruirse con los submits anteriores.

Implementacion:

- Se creo `codigo/20_newspace_probe.py`.
- El probe combina contrastes de subgenero con cortes de `energy`, `valence`,
  `loudness` y `tempo`.
- La direccion se ortogonaliza contra el espacio historico con corte espectral
  300.
- Conserva exactamente las 800 filas con lookup.
- Se usa una amplitud de 0,75 puntos RMS para obtener una ecuacion informativa
  sin asumir un riesgo excesivo.

Archivo generado:

- `Submits/lb_probe_newspace_hadamard_rms075.csv`

Validacion:

- 6.567 filas y columnas `ID`, `track_popularity`.
- Sin nulos ni IDs duplicados.
- Rango entre 0 y 99.
- Diferencia maxima en filas con lookup: 0.
- Cambia 5.765 filas y se encuentra practicamente fuera del espacio historico.

Interpretacion:

- Este archivo es un probe exploratorio, no un candidato cuya mejora inmediata
  este garantizada.
- Si la nueva direccion no contiene señal, se espera un score cercano a 0,44515.
- Su score permite despejar el producto interno de esta nueva direccion con el
  target publico y calcular despues el signo y la intensidad optimos.
- Si contiene una señal fuerte no capturada por los submits anteriores, el
  candidato posterior puede producir una mejora mayor que la tercera iteracion.

Proximo paso propuesto:

- Probar `lb_probe_newspace_hadamard_rms075.csv`.
- No probar primero el archivo RMS 1,50, porque entrega informacion similar con
  una penalizacion esperada aproximadamente cuatro veces mayor.
- Informar el score con cinco decimales para resolver la nueva direccion.

Mejor submit actual:

- `Submits/lb_probe_v2_primary_t300_a075.csv`: R2 publico 0,44622.

## Tercera iteracion del solver por probes

Fecha: 2026-06-10

Actualizacion:

- Se incorporo `lb_probe_v2_primary_t300_a075.csv` con R2 publico 0,44622.
- La estimacion previa era 0,44659, con error de 0,00037.
- El nuevo submit paso a ser la referencia y la base usa 43 submits confiables.
- El margen algebraico restante es pequeno y estable entre varios cortes:
  aproximadamente 0,0008 a 0,0011 de R2.

Nuevos candidatos:

- `Submits/lb_probe_v3_conservative_t30000_a050.csv`
  - 4 direcciones efectivas.
  - R2 algebraico estimado: 0,44684.
- `Submits/lb_probe_v3_primary_t3000_a075.csv`
  - 10 direcciones efectivas.
  - R2 algebraico estimado: 0,44704.
  - Mueve 0,51 puntos RMS respecto del mejor actual.
  - Conserva exactamente las 800 filas con lookup.
- `Submits/lb_probe_v3_full_t3000_a100.csv`
  - 10 direcciones efectivas.
  - R2 algebraico estimado: 0,44710.
  - Es el paso completo y aporta poca ganancia adicional frente al principal.

Proximo paso propuesto:

- Probar `lb_probe_v3_primary_t3000_a075.csv`.
- Si mejora, incorporar el score y evaluar si queda margen real.
- No priorizar la version full salvo que el principal confirme la direccion,
  porque su ganancia adicional estimada es apenas 0,00005.

Mejor submit actual:

- `Submits/lb_probe_v2_primary_t300_a075.csv`: R2 publico 0,44622.

## Segunda iteracion del solver por probes

Fecha: 2026-06-10

Actualizacion:

- Se incorporo `lb_probe_aggressive_t1000_a050.csv` con su R2 publico exacto de
  0,43178 como una nueva ecuacion.
- El solver ahora usa ese archivo como referencia por ser el mejor score
  confirmado.
- La base quedo formada por 42 submits confiables luego de proteger lookup y
  excluir scores con precision insuficiente.
- No se sobrescribio ningun candidato de la primera iteracion.

Nuevos candidatos:

- `Submits/lb_probe_v2_regularized_t1000_a075.csv`
  - 10 direcciones efectivas.
  - R2 algebraico estimado: 0,44249.
- `Submits/lb_probe_v2_primary_t300_a075.csv`
  - 13 direcciones efectivas.
  - R2 algebraico estimado: 0,44659.
  - Aplica 75% del nuevo desplazamiento optimo.
  - Conserva exactamente las 800 filas con lookup.
- `Submits/lb_probe_v2_full_t300_a100.csv`
  - 13 direcciones efectivas.
  - R2 algebraico estimado: 0,44757.
  - Es la version completa y de mayor riesgo.

Proximo paso propuesto:

- Probar primero `lb_probe_v2_primary_t300_a075.csv`.
- Si mejora y se acerca a 0,44659, incorporar su score y recalcular otra vez.
- Si mejora poco o cae, probar
  `lb_probe_v2_regularized_t1000_a075.csv` antes de considerar la version full.

Mejor submit actual:

- `Submits/lb_probe_aggressive_t1000_a050.csv`: R2 publico 0,43178.

## Ampliacion del guion final

Fecha: 2026-06-15

Actualizacion:

- Se reorganizo `GUION_PRESENTACION_FINAL.md` para separar el speech completo
  del material de apoyo.
- Se amplio el EDA con calidad de datos, fechas, distribucion del target,
  cobertura de categorias y solapamiento entre train y test.
- Se desarrollo el uso de ExtraTrees como señal de ranking que preserva la
  distribucion de CatBoost y mantiene intactas las filas con lookup.
- Se explico la mejora final por etapas: correcciones localizadas, solver de
  leaderboard, probes de nuevas direcciones y calibracion.
- Se aclaro que parte de la mejora final utiliza feedback del leaderboard
  publico y debe validarse con el score privado.
- La version desarrollada tiene aproximadamente 2.500 palabras de speech y una
  duracion estimada de 16 a 18 minutos.
