import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


RANDOM_STATE = 505
TARGET = "track_popularity"
ID_COL = "ID"
N_SPLITS = 3
TOP_NS = [60, 90, 120]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def apply_lookup(test, lookup, pred):
    lookup_map = dict(zip(lookup["track_id"], lookup["track_id_target_mean"]))
    seen = test["track_id"].isin(lookup_map).values
    out = pred.copy()
    out[seen] = test.loc[seen, "track_id"].map(lookup_map).values
    return np.clip(out, 0, 100), int(seen.sum())


def load_best_feature_order(project_root, X, y, cat_features):
    model_path = project_root / "modelos" / "catboost_best.cbm"
    metadata_path = project_root / "modelos" / "catboost_best_metadata.json"

    if model_path.exists():
        model = CatBoostRegressor()
        model.load_model(model_path)
        importances = model.get_feature_importance(Pool(X, y, cat_features=cat_features))
        order = (
            pd.DataFrame({"feature": X.columns, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        return order

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return pd.DataFrame({"feature": metadata["feature_cols"], "importance": np.nan})

    raise FileNotFoundError("No encontre modelo ni metadata del mejor CatBoost.")


def main():
    cwd = Path.cwd()
    project_root = cwd.parent if cwd.name.lower() in ["codigo", "codigos"] else cwd
    feature_dir = project_root / "otros csv"
    submit_dir = project_root / "Submits"
    model_dir = project_root / "modelos"
    submit_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    train = pd.read_csv(feature_dir / "train_features.csv")
    test = pd.read_csv(feature_dir / "test_features.csv")
    lookup = pd.read_csv(feature_dir / "track_id_lookup.csv")

    with open(model_dir / "catboost_best_metadata.json", "r", encoding="utf-8") as f:
        best_metadata = json.load(f)

    base_feature_cols = [c for c in best_metadata["feature_cols"] if c in train.columns and c in test.columns]
    base_cat_cols = [c for c in best_metadata["categorical_cols"] if c in base_feature_cols]

    X_base = train[base_feature_cols].copy()
    X_test_base = test[base_feature_cols].copy()
    y = train[TARGET].copy()
    groups = train["track_id"].copy()

    for col in base_cat_cols:
        X_base[col] = X_base[col].astype(str)
        X_test_base[col] = X_test_base[col].astype(str)
    base_cat_features = [X_base.columns.get_loc(c) for c in base_cat_cols]

    feature_importance = load_best_feature_order(project_root, X_base, y, base_cat_features)
    feature_importance.to_csv(feature_dir / "catboost_best_feature_importance.csv", index=False)

    print("Feature selection sobre mejor CatBoost")
    print("Features base:", len(base_feature_cols), "Cats base:", base_cat_cols)
    print("Top 20 features:")
    print(feature_importance.head(20).to_string(index=False))

    params = best_metadata["params_cv"].copy()
    params.update(
        {
            "iterations": 1000,
            "learning_rate": 0.04,
            "random_seed": RANDOM_STATE,
            "allow_writing_files": False,
            "thread_count": -1,
            "verbose": False,
        }
    )

    cv = GroupKFold(n_splits=N_SPLITS)
    fold_rows = []
    summary_rows = []
    lookup_preds = {}

    for top_n in TOP_NS:
        model_name = f"catboost_fs_top{top_n}"
        top_features = feature_importance.head(top_n)["feature"].tolist()

        # Keep core categorical descriptors even if their split importance is low.
        feature_cols = []
        for col in base_cat_cols + top_features:
            if col in base_feature_cols and col not in feature_cols:
                feature_cols.append(col)

        X = train[feature_cols].copy()
        X_test = test[feature_cols].copy()
        cat_cols = [c for c in base_cat_cols if c in feature_cols]
        for col in cat_cols:
            X[col] = X[col].astype(str)
            X_test[col] = X_test[col].astype(str)
        cat_features = [X.columns.get_loc(c) for c in cat_cols]

        print("\n" + "=" * 80)
        print(model_name, "features:", len(feature_cols), "cats:", len(cat_cols))
        oof = np.zeros(len(X))
        best_iters = []
        start = time.time()

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
            model = CatBoostRegressor(**params)
            model.fit(
                Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_features),
                eval_set=Pool(X.iloc[val_idx], y.iloc[val_idx], cat_features=cat_features),
                use_best_model=True,
                early_stopping_rounds=100,
                verbose=False,
            )
            pred = np.clip(model.predict(Pool(X.iloc[val_idx], cat_features=cat_features)), 0, 100)
            oof[val_idx] = pred
            best_iter = model.get_best_iteration()
            best_iters.append(best_iter)
            row = {
                "model": model_name,
                "fold": fold,
                "features_used": len(feature_cols),
                "rmse": rmse(y.iloc[val_idx], pred),
                "mae": mean_absolute_error(y.iloc[val_idx], pred),
                "r2": r2_score(y.iloc[val_idx], pred),
                "best_iteration": best_iter,
            }
            fold_rows.append(row)
            print(
                f"fold {fold}: RMSE={row['rmse']:.4f} MAE={row['mae']:.4f} "
                f"R2={row['r2']:.4f} best_iter={best_iter}"
            )

        model_rows = [r for r in fold_rows if r["model"] == model_name]
        summary = {
            "model": model_name,
            "features_used": len(feature_cols),
            "rmse_mean": float(np.mean([r["rmse"] for r in model_rows])),
            "rmse_std": float(np.std([r["rmse"] for r in model_rows], ddof=1)),
            "mae_mean": float(np.mean([r["mae"] for r in model_rows])),
            "r2_mean": float(np.mean([r["r2"] for r in model_rows])),
            "oof_rmse": rmse(y, oof),
            "oof_mae": mean_absolute_error(y, oof),
            "oof_r2": r2_score(y, oof),
            "best_iteration_median": float(np.median(best_iters)),
            "elapsed_seconds": time.time() - start,
        }
        summary_rows.append(summary)

        final_params = params.copy()
        final_params["iterations"] = int(max(250, round(summary["best_iteration_median"] + 70)))
        print("Entrenando final", model_name, "iterations", final_params["iterations"])
        final_model = CatBoostRegressor(**final_params)
        final_model.fit(Pool(X, y, cat_features=cat_features), verbose=False)
        raw_pred = np.clip(final_model.predict(Pool(X_test, cat_features=cat_features)), 0, 100)
        lookup_pred, lookup_rows = apply_lookup(test, lookup, raw_pred)
        lookup_preds[model_name] = lookup_pred

        submit = pd.DataFrame({"ID": test[ID_COL].astype(int), "track_popularity": lookup_pred})
        submit_path = submit_dir / f"{model_name}_submit.csv"
        submit.to_csv(submit_path, index=False)
        pd.DataFrame({"ID": test[ID_COL].astype(int), "track_popularity_raw": raw_pred}).to_csv(
            feature_dir / f"{model_name}_raw_predictions.csv", index=False
        )
        final_model.save_model(model_dir / f"{model_name}.cbm")
        print("Submit:", submit_path, "lookup rows:", lookup_rows)

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("rmse_mean").reset_index(drop=True)
    fold_df.to_csv(feature_dir / "catboost_feature_selection_cv_folds.csv", index=False)
    summary_df.to_csv(feature_dir / "catboost_feature_selection_cv_summary.csv", index=False)

    # Conservative ensemble: current best gets double weight.
    ensemble_sources = []
    best_raw_path = feature_dir / "catboost_best_test_predictions_raw.csv"
    if best_raw_path.exists():
        prev_raw = pd.read_csv(best_raw_path)["track_popularity_raw"].values
        prev_lookup, _ = apply_lookup(test, lookup, np.clip(prev_raw, 0, 100))
        ensemble_sources.extend([prev_lookup, prev_lookup])
    for model_name in summary_df.head(2)["model"]:
        ensemble_sources.append(lookup_preds[model_name])

    ensemble_pred = np.mean(np.vstack(ensemble_sources), axis=0)
    ensemble_submit = pd.DataFrame(
        {"ID": test[ID_COL].astype(int), "track_popularity": np.clip(ensemble_pred, 0, 100)}
    )
    ensemble_submit.to_csv(submit_dir / "catboost_fs_ensemble_submit.csv", index=False)

    with open(model_dir / "catboost_feature_selection_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cv": f"GroupKFold_{N_SPLITS}_track_id",
                "top_ns": TOP_NS,
                "base_cat_cols": base_cat_cols,
                "params": params,
                "summary": summary_df.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nResumen feature selection:")
    print(summary_df.round(4).to_string(index=False))
    print("\nSubmits guardados en:", submit_dir)


if __name__ == "__main__":
    main()
