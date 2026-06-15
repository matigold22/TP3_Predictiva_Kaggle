# Guion breve para la presentacion de modelos

## Diapositiva 1 - Titulo
Voy a presentar el proceso de modelado para predecir `track_popularity`. Separo dos resultados: el modelo final reproducible, que es la entrega principal, y el mejor resultado de competencia, que fue una optimizacion posterior usando feedback del leaderboard.

## Diapositiva 2 - Problema y datos
El problema es de regresion: queremos predecir popularidad entre 0 y 100. Train tiene 26.266 filas y test 6.567. El punto clave fue que train y test no tenian la misma composicion de generos y subgeneros, por eso una buena validacion interna no siempre se traslado al leaderboard.

## Diapositiva 3 - Modelo baseline
Como baseline use un CatBoost inicial. Lo elegi porque permite entrenar un modelo fuerte rapidamente sobre variables numericas y categoricas. Uso 157 features procesadas: audio, duracion, fechas, frecuencias, target encoding y contexto de playlist. La validacion fue GroupKFold por `track_id`. El baseline obtuvo R2 publico 0,36930.

## Diapositiva 4 - Features
La senal no estaba concentrada en una sola variable. Combine audio, tiempo, contexto, frecuencias y target encoding. Tambien use lookup para canciones del test que ya aparecian en train: eran 800 filas.

## Diapositiva 5 - Que se probo
Este grafico resume el recorrido real de submits. Primero probe CatBoost y variantes de features. Despues ajuste hiperparametros y postprocesos. Luego sume LightGBM y ExtraTrees como diversidad. Mas adelante aparecieron reglas por subgenero y, al final, probes de leaderboard.

## Diapositiva 6 - Estrategia de seleccion
La estrategia combino validacion interna y evidencia externa. La validacion interna servia para evitar modelos claramente malos, pero el leaderboard mostraba si el modelo trasladaba bien al test. Por eso los cambios fueron graduales: una hipotesis por vez, registrando cada submit con nombre exacto y R2 publico.

## Diapositiva 7 - Modelo final seleccionado
El modelo final reproducible es un pipeline compuesto: CatBoost top 60, ExtraTrees y postprocesamiento deterministico por subgenero. CatBoost top 60 fue el mejor modelo puro. ExtraTrees aporto una mirada distinta para reordenar filas no vistas. Despues se aplicaron dos ajustes localizados: `neo soul` y `progressive electro house`. El resultado publico fue 0,39740.

## Diapositiva 8 - Mejor resultado de competencia
Adicionalmente, el mejor resultado publico fue 0,46155. No lo presento como modelo principal porque usa feedback del leaderboard publico y combina submits historicos. Sirve como experimento avanzado de competencia, pero tiene mas riesgo de sobreajustar al 50% publico del test.

## Diapositiva 9 - Limitaciones y mejoras
La principal limitacion fue el drift entre train y test. Tambien el leaderboard publico no representa todo el test, por lo que optimizarlo demasiado puede ser riesgoso. Con mas tiempo, mejoraria la validacion, buscaria datos externos y automatizaria reglas por subgrupo con validacion mas honesta.

## Diapositiva 10 - Cierre
El desafio no fue solo entrenar un modelo con buena CV, sino construir una solucion robusta frente a un test distinto. Por eso separo el modelo final reproducible, con R2 0,39740, del mejor resultado de competencia, con R2 0,46155.
