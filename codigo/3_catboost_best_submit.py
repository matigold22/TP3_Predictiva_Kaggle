import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


RANDOM_STATE = 42
TARGET = "track_popularity"
ID_COL = "ID"
N_SPLITS = 3


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    cwd = Path.cwd()
    project_root = cwd.parent if cwd.name.lower() in ["codigo", "codigos"] else cwd
    feature_dir = project_root / "otros csv"
    submit_dir = project_root / "Submits"
    results_dir = project_root / "otros csv"
    model_dir = project_root / "modelos"
    submit_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    train = pd.read_csv(feature_dir / "train_features.csv")
    test = pd.read_csv(feature_dir / "test_features.csv")
    lookup = pd.read_csv(feature_dir / "track_id_lookup.csv")

    # CatBoost puede usar categoricas, pero las de alta cardinalidad directa
    # hacen la corrida muy lenta. Las dejamos representadas por frecuencia y TE.
    categorical_cols = [
        "playlist_genre",
        "playlist_subgenre",
        "key_str",
        "mode_str",
        "key_mode",
        "genre_subgenre",
    ]

    numeric_cols = [
        col
        for col in train.columns
        if col not in [ID_COL, TARGET]
        and col not in categorical_cols
        and is_numeric_dtype(train[col])
    ]

    feature_cols = categorical_cols + numeric_cols
    feature_cols = [col for col in feature_cols if col in train.columns and col in test.columns]

    X = train[feature_cols].copy()
    y = train[TARGET].copy()
    X_test = test[feature_cols].copy()
    groups = train["track_id"].copy()

    cat_cols = [col for col in categorical_cols if col in X.columns]
    cat_features = [X.columns.get_loc(col) for col in cat_cols]
    for col in cat_cols:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1200,
        "learning_rate": 0.035,
        "depth": 6,
        "l2_leaf_reg": 9,
        "random_strength": 1.2,
        "bagging_temperature": 0.5,
        "subsample": 0.85,
        "rsm": 0.9,
        "min_data_in_leaf": 35,
        "random_seed": RANDOM_STATE,
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": False,
    }

    print("Train:", train.shape, "Test:", test.shape)
    print("Features usadas:", len(feature_cols))
    print("Categoricas directas:", cat_cols)
    print("CV:", f"GroupKFold({N_SPLITS}) por track_id")

    cv = GroupKFold(n_splits=N_SPLITS)
    fold_rows = []
    oof_pred = np.zeros(len(X))
    best_iterations = []
    start = time.time()

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        train_pool = Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_features)
        valid_pool = Pool(X.iloc[val_idx], y.iloc[val_idx], cat_features=cat_features)

        model = CatBoostRegressor(**params)
        fold_start = time.time()
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=120,
            verbose=False,
        )

        pred = np.clip(model.predict(valid_pool), 0, 100)
        oof_pred[val_idx] = pred
        best_iter = model.get_best_iteration()
        best_iterations.append(best_iter)

        row = {
            "fold": fold,
            "rmse": rmse(y.iloc[val_idx], pred),
            "mae": mean_absolute_error(y.iloc[val_idx], pred),
            "r2": r2_score(y.iloc[val_idx], pred),
            "best_iteration": best_iter,
            "train_rows": len(tr_idx),
            "valid_rows": len(val_idx),
            "elapsed_seconds": time.time() - fold_start,
        }
        fold_rows.append(row)
        print(
            f"fold {fold}: RMSE={row['rmse']:.4f} "
            f"MAE={row['mae']:.4f} R2={row['r2']:.4f} "
            f"best_iter={best_iter} time={row['elapsed_seconds']:.1f}s"
        )

    fold_metrics = pd.DataFrame(fold_rows)
    summary = {
        "cv_type": f"GroupKFold_{N_SPLITS}_track_id",
        "features_used": len(feature_cols),
        "categorical_features_used": len(cat_cols),
        "rmse_mean": fold_metrics["rmse"].mean(),
        "rmse_std": fold_metrics["rmse"].std(),
        "mae_mean": fold_metrics["mae"].mean(),
        "mae_std": fold_metrics["mae"].std(),
        "r2_mean": fold_metrics["r2"].mean(),
        "oof_rmse": rmse(y, oof_pred),
        "oof_mae": mean_absolute_error(y, oof_pred),
        "oof_r2": r2_score(y, oof_pred),
        "best_iteration_mean": float(np.mean(best_iterations)),
        "best_iteration_median": float(np.median(best_iterations)),
        "elapsed_seconds": time.time() - start,
    }

    print("\nResumen CV:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Entrenamiento final: usa una cantidad de iteraciones basada en CV, con margen.
    final_iterations = int(max(250, round(summary["best_iteration_median"] + 80)))
    final_params = params.copy()
    final_params["iterations"] = final_iterations
    print(f"\nEntrenando final con iterations={final_iterations}")

    full_pool = Pool(X, y, cat_features=cat_features)
    test_pool = Pool(X_test, cat_features=cat_features)
    final_model = CatBoostRegressor(**final_params)
    final_model.fit(full_pool, verbose=False)

    raw_pred = np.clip(final_model.predict(test_pool), 0, 100)

    # Regla final de lookup para canciones ya vistas en train.
    lookup_map = dict(zip(lookup["track_id"], lookup["track_id_target_mean"]))
    seen_track_mask = test["track_id"].isin(lookup_map)
    final_pred = raw_pred.copy()
    final_pred[seen_track_mask.values] = test.loc[seen_track_mask, "track_id"].map(lookup_map).values
    final_pred = np.clip(final_pred, 0, 100)

    submit = pd.DataFrame(
        {
            "ID": test[ID_COL].astype(int),
            "track_popularity": final_pred,
        }
    )

    submit_path = submit_dir / "catboost_best_submit.csv"
    raw_pred_path = results_dir / "catboost_best_test_predictions_raw.csv"
    fold_path = results_dir / "catboost_best_cv_folds.csv"
    summary_path = results_dir / "catboost_best_cv_summary.csv"
    oof_path = results_dir / "catboost_best_oof_predictions.csv"
    model_path = model_dir / "catboost_best.cbm"
    metadata_path = model_dir / "catboost_best_metadata.json"

    submit.to_csv(submit_path, index=False)
    pd.DataFrame({"ID": test[ID_COL].astype(int), "track_popularity_raw": raw_pred}).to_csv(
        raw_pred_path, index=False
    )
    fold_metrics.to_csv(fold_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    pd.DataFrame({"ID": train[ID_COL].astype(int), TARGET: y, "pred_oof": oof_pred}).to_csv(
        oof_path, index=False
    )
    final_model.save_model(model_path)

    metadata = {
        "params_cv": params,
        "params_final": final_params,
        "summary": summary,
        "feature_cols": feature_cols,
        "categorical_cols": cat_cols,
        "lookup_rows_applied": int(seen_track_mask.sum()),
        "submit_path": str(submit_path),
        "model_path": str(model_path),
        "prediction_mean": float(submit["track_popularity"].mean()),
        "prediction_min": float(submit["track_popularity"].min()),
        "prediction_max": float(submit["track_popularity"].max()),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nArchivos guardados:")
    print(submit_path)
    print(raw_pred_path)
    print(fold_path)
    print(summary_path)
    print(oof_path)
    print(model_path)
    print(metadata_path)
    print("\nPreview submit:")
    print(submit.head().to_string(index=False))
    print("\nPredicciones:")
    print(submit["track_popularity"].describe().to_string())
    print("Lookup aplicado a filas:", int(seen_track_mask.sum()))


if __name__ == "__main__":
    main()
