# Otros CSV

Archivos auxiliares generados por el pipeline:

- `test_features.csv`: features procesadas para la base de validacion/test.
- `track_id_lookup.csv`: tabla de lookup por `track_id`.
- `feature_columns.csv`: columnas generadas por el feature engineering.
- `lb_probe_solver/`: scores publicos, diagnostico y candidatos del solver de
  leaderboard.

Estos archivos permiten reproducir resultados sin tener que inspeccionar toda la
bitacora historica.
