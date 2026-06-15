import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


RANDOM_STATE = 707
TARGET = "track_popularity"
ID_COL = "ID"
TOP_N = 60


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def project_root_from_cwd():
    cwd = Path.cwd()
    return cwd.parent if cwd.name.lower() in ["codigo", "codigos"] else cwd


def apply_lookup(test, lookup, pred):
    lookup_map = dict(zip(lookup["track_id"], lookup["track_id_target_mean"]))
    seen = test["track_id"].isin(lookup_map).values
    out = pred.copy()
    out[seen] = test.loc[seen, "track_id"].map(lookup_map).values
    return np.clip(out, 0, 100), int(seen.sum())


def repeated_group_kfold(groups, n_splits=5, n_repeats=2, random_state=RANDOM_STATE):
    """Repeated GroupKFold aproximado: baraja grupos unicos y reparte grupos por folds."""
    unique_groups = pd.Series(groups).drop_duplicates().to_numpy()
    for repeat in range(1, n_repeats + 1):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
        for fold, (_, valid_group_idx) in enumerate(kf.split(unique_groups), start=1):
            valid_groups = set(unique_groups[valid_group_idx])
            valid_mask = pd.Series(groups).isin(valid_groups).to_numpy()
            valid_idx = np.flatnonzero(valid_mask)
            train_idx = np.flatnonzero(~valid_mask)
            yield repeat, fold, train_idx, valid_idx


def validate_submit(submit, expected_rows):
    if list(submit.columns) != [ID_COL, TARGET]:
        raise ValueError(f"Columnas invalidas: {submit.columns.tolist()}")
    if len(submit) != expected_rows:
        raise ValueError(f"Filas invalidas: {len(submit)} != {expected_rows}")
    if submit[ID_COL].duplicated().any():
        raise ValueError("IDs duplicados.")
    if submit[TARGET].isna().any():
        raise ValueError("Predicciones nulas.")
    if not submit[TARGET].between(0, 100).all():
        raise ValueError("Predicciones fuera de rango.")


def save_submit(path, ids, pred):
    submit = pd.DataFrame({ID_COL: ids.astype(int), TARGET: np.clip(pred, 0, 100)})
    validate_submit(submit, len(ids))
    submit.to_csv(path, index=False)
    print("Submit:", path)


def make_top60_features(feature_order, base_feature_cols, base_cat_cols):
    top_features = feature_order.head(TOP_N)["feature"].tolist()
    feature_cols = []
    for col in base_cat_cols + top_features:
        if col in base_feature_cols and col not in feature_cols:
            feature_cols.append(col)
    return feature_cols


def candidate_configs():
    # El ganador publico top60 salio de estos parametros base:
    # iterations=1000, learning_rate=0.04, depth=6, l2=9,
    # random_strength=1.2, bagging_temperature=0.5, rsm=0.9,
    # min_data_in_leaf=35, final ~= best_iter_median + 70.
    #
    # Aca jugamos cerca. Nada de movimientos grandes: el leaderboard publico
    # ya castigo mas arboles/learning rate bajo y poda fuerte.
    base = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1100,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 9,
        "random_strength": 1.2,
        "bagging_temperature": 0.5,
        "subsample": 0.85,
        "rsm": 0.9,
        "min_data_in_leaf": 35,
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": False,
    }
    return {
        "top60_public_anchor": {
            "params": {**base, "iterations": 1000, "random_seed": 505},
            "early_stopping_rounds": 100,
            "final_padding": 70,
            "final_seeds": [505],
        },
        "top60_l2_10_seed707": {
            "params": {**base, "l2_leaf_reg": 10, "random_seed": 707},
            "early_stopping_rounds": 110,
            "final_padding": 70,
            "final_seeds": [707],
        },
        "top60_l2_8_less_noise": {
            "params": {
                **base,
                "l2_leaf_reg": 8,
                "random_strength": 1.0,
                "bagging_temperature": 0.45,
                "random_seed": 708,
            },
            "early_stopping_rounds": 110,
            "final_padding": 70,
            "final_seeds": [708],
        },
        "top60_l2_11_more_noise": {
            "params": {
                **base,
                "l2_leaf_reg": 11,
                "random_strength": 1.45,
                "bagging_temperature": 0.62,
                "rsm": 0.88,
                "random_seed": 709,
            },
            "early_stopping_rounds": 110,
            "final_padding": 70,
            "final_seeds": [709],
        },
        "top60_leaf45_rsm088": {
            "params": {
                **base,
                "l2_leaf_reg": 9.5,
                "random_strength": 1.35,
                "bagging_temperature": 0.55,
                "rsm": 0.88,
                "min_data_in_leaf": 45,
                "random_seed": 710,
            },
            "early_stopping_rounds": 110,
            "final_padding": 70,
            "final_seeds": [710],
        },
        "top60_leaf28_rsm092": {
            "params": {
                **base,
                "l2_leaf_reg": 8.5,
                "random_strength": 1.10,
                "bagging_temperature": 0.48,
                "rsm": 0.92,
                "min_data_in_leaf": 28,
                "random_seed": 711,
            },
            "early_stopping_rounds": 110,
            "final_padding": 70,
            "final_seeds": [711],
        },
        "top60_depth5_publicish": {
            "params": {
                **base,
                "iterations": 1250,
                "learning_rate": 0.042,
                "depth": 5,
                "l2_leaf_reg": 7,
                "random_strength": 1.0,
                "bagging_temperature": 0.45,
                "rsm": 0.94,
                "min_data_in_leaf": 25,
                "random_seed": 712,
            },
            "early_stopping_rounds": 120,
            "final_padding": 80,
            "final_seeds": [712],
        },
        "top60_seed_bag_public_params": {
            "params": {**base, "iterations": 1000, "random_seed": 713},
            "early_stopping_rounds": 100,
            "final_padding": 70,
            "final_seeds": [713, 813, 913],
        },
    }


def evaluate_candidate(
    name,
    config,
    X,
    y,
    groups,
    X_test,
    test,
    lookup,
    cat_features,
    n_splits,
    n_repeats,
):
    params = config["params"].copy()
    rows = []
    oof_sum = np.zeros(len(X))
    oof_count = np.zeros(len(X))
    best_iters = []
    start = time.time()

    print("\n" + "=" * 90)
    print(name)
    print(
        "params:",
        {
            k: params[k]
            for k in [
                "iterations",
                "learning_rate",
                "depth",
                "l2_leaf_reg",
                "random_strength",
                "bagging_temperature",
                "rsm",
                "min_data_in_leaf",
                "random_seed",
            ]
        },
    )
    print(f"CV: repeated grouped {n_splits} folds x {n_repeats} repeats")

    for repeat, fold, tr_idx, val_idx in repeated_group_kfold(
        groups, n_splits=n_splits, n_repeats=n_repeats
    ):
        model = CatBoostRegressor(**params)
        model.fit(
            Pool(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_features),
            eval_set=Pool(X.iloc[val_idx], y.iloc[val_idx], cat_features=cat_features),
            use_best_model=True,
            early_stopping_rounds=config["early_stopping_rounds"],
            verbose=False,
        )
        pred = np.clip(model.predict(Pool(X.iloc[val_idx], cat_features=cat_features)), 0, 100)
        oof_sum[val_idx] += pred
        oof_count[val_idx] += 1
        best_iter = model.get_best_iteration()
        best_iters.append(best_iter)
        row = {
            "model": name,
            "repeat": repeat,
            "fold": fold,
            "valid_rows": len(val_idx),
            "rmse": rmse(y.iloc[val_idx], pred),
            "mae": mean_absolute_error(y.iloc[val_idx], pred),
            "r2": r2_score(y.iloc[val_idx], pred),
            "best_iteration": best_iter,
        }
        rows.append(row)
        print(
            f"repeat {repeat} fold {fold}: RMSE={row['rmse']:.4f} "
            f"MAE={row['mae']:.4f} R2={row['r2']:.4f} best_iter={best_iter}"
        )

    oof = oof_sum / np.maximum(oof_count, 1)
    final_iterations = int(max(300, round(float(np.median(best_iters)) + config["final_padding"])))
    raw_preds = []
    lookup_preds = []
    lookup_rows = 0

    print("Entrenando final iterations:", final_iterations, "seeds:", config["final_seeds"])
    for seed in config["final_seeds"]:
        final_params = params.copy()
        final_params["iterations"] = final_iterations
        final_params["random_seed"] = seed
        final_model = CatBoostRegressor(**final_params)
        final_model.fit(Pool(X, y, cat_features=cat_features), verbose=False)
        raw_pred = np.clip(final_model.predict(Pool(X_test, cat_features=cat_features)), 0, 100)
        lookup_pred, lookup_rows = apply_lookup(test, lookup, raw_pred)
        raw_preds.append(raw_pred)
        lookup_preds.append(lookup_pred)

    fold_df = pd.DataFrame(rows)
    summary = {
        "model": name,
        "top_n": TOP_N,
        "features_used": X.shape[1],
        "cv_folds_total": int(n_splits * n_repeats),
        "rmse_mean": float(fold_df["rmse"].mean()),
        "rmse_std": float(fold_df["rmse"].std()),
        "mae_mean": float(fold_df["mae"].mean()),
        "r2_mean": float(fold_df["r2"].mean()),
        "oof_rmse_repeated_mean": rmse(y, oof),
        "oof_mae_repeated_mean": mean_absolute_error(y, oof),
        "oof_r2_repeated_mean": r2_score(y, oof),
        "best_iteration_median": float(np.median(best_iters)),
        "best_iteration_mean": float(np.mean(best_iters)),
        "final_iterations": final_iterations,
        "final_seed_count": len(config["final_seeds"]),
        "lookup_rows_applied": lookup_rows,
        "prediction_mean": float(np.mean(np.vstack(lookup_preds), axis=0).mean()),
        "prediction_min": float(np.mean(np.vstack(lookup_preds), axis=0).min()),
        "prediction_max": float(np.mean(np.vstack(lookup_preds), axis=0).max()),
        "elapsed_seconds": time.time() - start,
    }
    return rows, summary, np.mean(np.vstack(raw_preds), axis=0), np.mean(np.vstack(lookup_preds), axis=0)


def build_blends(project_root, submit_dir, test_ids, preds, summary_df):
    rows = []
    public_path = project_root / "Submits" / "catboost_fs_top60_submit.csv"
    if public_path.exists():
        public_pred = pd.read_csv(public_path)[TARGET].values
        best_name = summary_df.sort_values("rmse_mean").iloc[0]["model"]
        best_pred = preds[best_name]
        recipes = {
            "catboost_top60_hypercv_blend_public90_best10.csv": (0.90 * public_pred + 0.10 * best_pred, f"0.90 public_top60 + 0.10 {best_name}"),
            "catboost_top60_hypercv_blend_public80_best20.csv": (0.80 * public_pred + 0.20 * best_pred, f"0.80 public_top60 + 0.20 {best_name}"),
        }
        for filename, (pred, recipe) in recipes.items():
            path = submit_dir / filename
            save_submit(path, test_ids, pred)
            rows.append({"blend": filename, "recipe": recipe})

    top_models = summary_df.sort_values("rmse_mean").head(3)["model"].tolist()
    if len(top_models) >= 2:
        pred = np.mean(np.vstack([preds[m] for m in top_models[:2]]), axis=0)
        filename = "catboost_top60_hypercv_blend_best2_equal.csv"
        save_submit(submit_dir / filename, test_ids, pred)
        rows.append({"blend": filename, "recipe": " + ".join([f"0.50 {m}" for m in top_models[:2]])})
    if len(top_models) >= 3:
        pred = 0.50 * preds[top_models[0]] + 0.30 * preds[top_models[1]] + 0.20 * preds[top_models[2]]
        filename = "catboost_top60_hypercv_blend_best3_weighted.csv"
        save_submit(submit_dir / filename, test_ids, pred)
        rows.append(
            {
                "blend": filename,
                "recipe": f"0.50 {top_models[0]} + 0.30 {top_models[1]} + 0.20 {top_models[2]}",
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="Lista candidatos y termina.")
    parser.add_argument("--only", nargs="*", default=None, help="Candidatos a correr.")
    parser.add_argument("--splits", type=int, default=5, help="Folds por repeticion.")
    parser.add_argument("--repeats", type=int, default=2, help="Repeticiones de CV agrupada.")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = candidate_configs()
    if args.list:
        print("\n".join(configs.keys()))
        return
    if args.only:
        missing = sorted(set(args.only) - set(configs))
        if missing:
            raise ValueError(f"Candidatos inexistentes: {missing}")
        configs = {name: configs[name] for name in args.only}

    project_root = project_root_from_cwd()
    feature_dir = project_root / "otros csv"
    submit_dir = project_root / "Submits"
    model_dir = project_root / "modelos"
    submit_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    train = pd.read_csv(feature_dir / "train_features.csv")
    test = pd.read_csv(feature_dir / "test_features.csv")
    lookup = pd.read_csv(feature_dir / "track_id_lookup.csv")
    feature_order = pd.read_csv(feature_dir / "catboost_best_feature_importance.csv")

    with open(model_dir / "catboost_best_metadata.json", "r", encoding="utf-8") as f:
        best_metadata = json.load(f)

    base_feature_cols = [c for c in best_metadata["feature_cols"] if c in train.columns and c in test.columns]
    base_cat_cols = [c for c in best_metadata["categorical_cols"] if c in base_feature_cols]
    feature_cols = make_top60_features(feature_order, base_feature_cols, base_cat_cols)

    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    y = train[TARGET].copy()
    groups = train["track_id"].copy()
    cat_cols = [c for c in base_cat_cols if c in feature_cols]
    for col in cat_cols:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    cat_features = [X.columns.get_loc(c) for c in cat_cols]

    print("CatBoost top60 hyper-CV")
    print("Train:", train.shape, "Test:", test.shape)
    print("Features:", len(feature_cols), "Cats:", cat_cols)
    print(f"CV pedida: {args.splits} folds x {args.repeats} repeats =", args.splits * args.repeats, "fits por candidato")
    print("Candidatos:", ", ".join(configs.keys()))

    fold_rows = []
    summaries = []
    preds = {}
    raw_preds = {}
    for name, config in configs.items():
        rows, summary, raw_pred, pred = evaluate_candidate(
            name=name,
            config=config,
            X=X,
            y=y,
            groups=groups,
            X_test=X_test,
            test=test,
            lookup=lookup,
            cat_features=cat_features,
            n_splits=args.splits,
            n_repeats=args.repeats,
        )
        fold_rows.extend(rows)
        summaries.append(summary)
        preds[name] = pred
        raw_preds[name] = raw_pred
        save_submit(submit_dir / f"catboost_{name}_submit.csv", test[ID_COL], pred)
        pd.DataFrame({ID_COL: test[ID_COL].astype(int), "track_popularity_raw": raw_pred}).to_csv(
            feature_dir / f"catboost_{name}_raw_predictions.csv", index=False
        )

    summary_cols = [
        "model",
        "top_n",
        "features_used",
        "cv_folds_total",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "r2_mean",
        "oof_rmse_repeated_mean",
        "oof_mae_repeated_mean",
        "oof_r2_repeated_mean",
        "best_iteration_median",
        "final_iterations",
        "final_seed_count",
        "lookup_rows_applied",
        "prediction_mean",
        "prediction_min",
        "prediction_max",
        "elapsed_seconds",
    ]
    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summaries).sort_values("rmse_mean").reset_index(drop=True)
    fold_df.to_csv(feature_dir / "catboost_top60_hypercv_folds.csv", index=False)
    summary_df[summary_cols].to_csv(feature_dir / "catboost_top60_hypercv_summary.csv", index=False)
    blend_rows = build_blends(project_root, submit_dir, test[ID_COL], preds, summary_df)
    pd.DataFrame(blend_rows).to_csv(feature_dir / "catboost_top60_hypercv_blends.csv", index=False)

    with open(model_dir / "catboost_top60_hypercv_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cv": f"Repeated grouped {args.splits}x{args.repeats}",
                "top_n": TOP_N,
                "feature_cols": feature_cols,
                "cat_cols": cat_cols,
                "configs_run": configs,
                "summary": summary_df[summary_cols].to_dict(orient="records"),
                "blends": blend_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nResumen:")
    print(summary_df[summary_cols].round(5).to_string(index=False))
    print("\nArchivos:")
    print(feature_dir / "catboost_top60_hypercv_summary.csv")
    print(feature_dir / "catboost_top60_hypercv_folds.csv")
    print(feature_dir / "catboost_top60_hypercv_blends.csv")
    print(model_dir / "catboost_top60_hypercv_metadata.json")


if __name__ == "__main__":
    main()
