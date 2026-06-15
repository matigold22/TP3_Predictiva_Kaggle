import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


TARGET = "track_popularity"
ID_COL = "ID"
RANDOM_STATE = 2026


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


def load_numeric_feature_order(project_root, train):
    imp_path = project_root / "otros csv" / "catboost_best_feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        ordered = [c for c in imp["feature"].tolist() if c in train.columns]
    else:
        with open(project_root / "modelos" / "catboost_best_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        ordered = [c for c in meta["feature_cols"] if c in train.columns]
    return [c for c in ordered if pd.api.types.is_numeric_dtype(train[c])]


def candidate_configs():
    base = {
        "n_estimators": 700,
        "criterion": "squared_error",
        "bootstrap": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": 0,
    }
    return {
        "et_top45_leaf4": {
            "top_n": 45,
            "params": {
                **base,
                "max_features": 0.70,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "max_depth": None,
            },
        },
        "et_top70_leaf5": {
            "top_n": 70,
            "params": {
                **base,
                "max_features": 0.60,
                "min_samples_leaf": 5,
                "min_samples_split": 10,
                "max_depth": None,
            },
        },
        "et_top100_leaf8": {
            "top_n": 100,
            "params": {
                **base,
                "max_features": 0.55,
                "min_samples_leaf": 8,
                "min_samples_split": 16,
                "max_depth": None,
            },
        },
    }


def evaluate_candidate(name, config, train, test, lookup, feature_order, splits):
    feature_cols = [c for c in feature_order[: config["top_n"]] if c in test.columns]
    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    y = train[TARGET].copy()
    groups = train["track_id"].copy()

    params = config["params"].copy()
    cv = GroupKFold(n_splits=splits)
    oof = np.zeros(len(X))
    fold_rows = []
    start = time.time()

    print("\n" + "=" * 90)
    print(name, "features:", len(feature_cols))
    print("params:", {k: params[k] for k in ["n_estimators", "max_features", "min_samples_leaf", "min_samples_split"]})

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        model = ExtraTreesRegressor(**params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = np.clip(model.predict(X.iloc[val_idx]), 0, 100)
        oof[val_idx] = pred
        row = {
            "model": name,
            "fold": fold,
            "features_used": len(feature_cols),
            "rmse": rmse(y.iloc[val_idx], pred),
            "mae": mean_absolute_error(y.iloc[val_idx], pred),
            "r2": r2_score(y.iloc[val_idx], pred),
        }
        fold_rows.append(row)
        print(f"fold {fold}: RMSE={row['rmse']:.4f} MAE={row['mae']:.4f} R2={row['r2']:.4f}")

    model = ExtraTreesRegressor(**params)
    model.fit(X, y)
    raw_pred = np.clip(model.predict(X_test), 0, 100)
    lookup_pred, lookup_rows = apply_lookup(test, lookup, raw_pred)

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "model": name,
        "features_used": len(feature_cols),
        "top_n": config["top_n"],
        "cv_folds_total": splits,
        "rmse_mean": float(fold_df["rmse"].mean()),
        "rmse_std": float(fold_df["rmse"].std()),
        "mae_mean": float(fold_df["mae"].mean()),
        "r2_mean": float(fold_df["r2"].mean()),
        "oof_rmse": rmse(y, oof),
        "oof_mae": mean_absolute_error(y, oof),
        "oof_r2": r2_score(y, oof),
        "lookup_rows_applied": lookup_rows,
        "prediction_mean": float(lookup_pred.mean()),
        "prediction_min": float(lookup_pred.min()),
        "prediction_max": float(lookup_pred.max()),
        "elapsed_seconds": time.time() - start,
    }
    return fold_rows, summary, raw_pred, lookup_pred, feature_cols


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--splits", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    configs = candidate_configs()
    if args.list:
        for name, cfg in configs.items():
            print(name, cfg)
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
    feature_order = load_numeric_feature_order(project_root, train)

    print("ExtraTrees models")
    print("Train:", train.shape, "Test:", test.shape)
    print("Numeric ordered features:", len(feature_order))
    print("Candidatos:", ", ".join(configs.keys()))

    all_folds = []
    summaries = []
    metadata_candidates = {}
    for name, config in configs.items():
        fold_rows, summary, raw_pred, lookup_pred, feature_cols = evaluate_candidate(
            name=name,
            config=config,
            train=train,
            test=test,
            lookup=lookup,
            feature_order=feature_order,
            splits=args.splits,
        )
        all_folds.extend(fold_rows)
        summaries.append(summary)
        metadata_candidates[name] = {"config": config, "feature_cols": feature_cols}

        save_submit(submit_dir / f"{name}_submit.csv", test[ID_COL], lookup_pred)
        pd.DataFrame({ID_COL: test[ID_COL].astype(int), "track_popularity_raw": raw_pred}).to_csv(
            feature_dir / f"{name}_raw_predictions.csv", index=False
        )
        pd.DataFrame({"feature": feature_cols}).to_csv(feature_dir / f"{name}_features.csv", index=False)

    summary_df = pd.DataFrame(summaries).sort_values("rmse_mean").reset_index(drop=True)
    pd.DataFrame(all_folds).to_csv(feature_dir / "extratrees_cv_folds.csv", index=False)
    summary_df.to_csv(feature_dir / "extratrees_cv_summary.csv", index=False)
    with open(model_dir / "extratrees_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cv": f"GroupKFold_{args.splits}_track_id",
                "candidates": metadata_candidates,
                "summary": summary_df.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nResumen:")
    print(summary_df.round(5).to_string(index=False))
    print("\nArchivos:")
    print(feature_dir / "extratrees_cv_summary.csv")
    print(model_dir / "extratrees_metadata.json")


if __name__ == "__main__":
    main()
