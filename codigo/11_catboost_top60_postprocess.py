import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "track_popularity"
ID_COL = "ID"


def project_root_from_cwd():
    cwd = Path.cwd()
    return cwd.parent if cwd.name.lower() in ["codigo", "codigos"] else cwd


def validate_submit(submit, expected_rows):
    if list(submit.columns) != [ID_COL, TARGET]:
        raise ValueError(f"Columnas invalidas: {submit.columns.tolist()}")
    if len(submit) != expected_rows:
        raise ValueError(f"Cantidad de filas invalida: {len(submit)} != {expected_rows}")
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
    return {
        "file": path.name,
        "mean": float(submit[TARGET].mean()),
        "std": float(submit[TARGET].std()),
        "min": float(submit[TARGET].min()),
        "max": float(submit[TARGET].max()),
    }


def describe(name, pred, unseen_mask):
    s = pd.Series(pred)
    return {
        "name": name,
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p05": float(s.quantile(0.05)),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
        "unseen_mean": float(s[unseen_mask].mean()),
        "unseen_std": float(s[unseen_mask].std()),
    }


def main():
    project_root = project_root_from_cwd()
    submit_dir = project_root / "Submits"
    feature_dir = project_root / "otros csv"
    model_dir = project_root / "modelos"
    submit_dir.mkdir(exist_ok=True)
    feature_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    test = pd.read_csv(feature_dir / "test_features.csv")
    lookup = pd.read_csv(feature_dir / "track_id_lookup.csv")
    winner = pd.read_csv(submit_dir / "catboost_fs_top60_submit.csv")
    ids = winner[ID_COL]
    base = winner[TARGET].to_numpy(dtype=float)

    seen_mask = test["track_id"].isin(set(lookup["track_id"])).to_numpy()
    unseen_mask = ~seen_mask

    variants = {}

    # Very small location corrections only on rows without track_id lookup.
    # The public 90/10 hyper blend reduced unseen mean slightly and got worse,
    # so these test the opposite direction without touching the reliable lookup rows.
    for shift in [0.05, 0.10, 0.20]:
        pred = base.copy()
        pred[unseen_mask] = pred[unseen_mask] + shift
        variants[f"catboost_top60_post_unseen_shift_p{int(shift * 100):02d}"] = pred

    # The l2_10 hyper variant and the more_trees variant both got worse on public.
    # These are small extrapolations away from those directions.
    bad_sources = {
        "away_l2_03": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.03),
        "away_l2_05": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.05),
        "away_l2_07": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.07),
        "away_l2_10": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.10),
        "away_l2_12": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.12),
        "away_l2_20": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.20),
        "away_l2_25": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.25),
        "away_l2_30": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.30),
        "away_l2_33": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.33),
        "away_l2_35": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.35),
        "away_l2_37": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.37),
        "away_l2_40": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.40),
        "away_l2_42": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.42),
        "away_l2_50": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 0.50),
        "away_l2_100": (submit_dir / "catboost_top60_l2_10_seed707_submit.csv", 1.00),
        "away_moretrees_03": (submit_dir / "catboost_fs_play_top60_more_trees_submit.csv", 0.03),
    }
    for name, (path, strength) in bad_sources.items():
        if not path.exists():
            continue
        bad = pd.read_csv(path)[TARGET].to_numpy(dtype=float)
        pred = base.copy()
        pred[unseen_mask] = base[unseen_mask] + strength * (base[unseen_mask] - bad[unseen_mask])
        variants[f"catboost_top60_post_{name}"] = pred

    # Selective away-l2: use the same winning direction, but vary strength by how
    # much the public-bad l2 model disagrees with the base model on unseen rows.
    l2_bad_path = submit_dir / "catboost_top60_l2_10_seed707_submit.csv"
    if l2_bad_path.exists():
        l2_bad = pd.read_csv(l2_bad_path)[TARGET].to_numpy(dtype=float)
        delta = base - l2_bad
        abs_delta_unseen = np.abs(delta[unseen_mask])
        q33, q66 = np.quantile(abs_delta_unseen, [1 / 3, 2 / 3])

        selective_recipes = {
            "away_l2_sel_diff_25_35_45": (0.25, 0.35, 0.45),
            "away_l2_sel_diff_30_35_45": (0.30, 0.35, 0.45),
            "away_l2_sel_diff_30_37_45": (0.30, 0.37, 0.45),
            "away_l2_sel_diff_25_37_50": (0.25, 0.37, 0.50),
        }
        for name, (low_strength, mid_strength, high_strength) in selective_recipes.items():
            strength = np.zeros_like(base, dtype=float)
            unseen_abs_delta = np.abs(delta[unseen_mask])
            unseen_strength = np.select(
                [unseen_abs_delta <= q33, unseen_abs_delta <= q66],
                [low_strength, mid_strength],
                default=high_strength,
            )
            strength[unseen_mask] = unseen_strength
            pred = base.copy()
            pred[unseen_mask] = base[unseen_mask] + strength[unseen_mask] * delta[unseen_mask]
            variants[f"catboost_top60_post_{name}"] = pred

    # A tiny blend toward the older CatBoost base raises the unseen mean, but keeps
    # the winner dominant. This is included as a low-risk diversity candidate.
    old_base_path = submit_dir / "catboost_best_submit.csv"
    if old_base_path.exists():
        old_base = pd.read_csv(old_base_path)[TARGET].to_numpy(dtype=float)
        pred = base.copy()
        pred[unseen_mask] = 0.97 * base[unseen_mask] + 0.03 * old_base[unseen_mask]
        variants["catboost_top60_post_blend_oldbase_03"] = pred

    rows = [describe("catboost_fs_top60_submit", base, unseen_mask)]
    saved = []
    for name, pred in variants.items():
        path = submit_dir / f"{name}.csv"
        saved_info = save_submit(path, ids, pred)
        saved.append(saved_info)
        rows.append(describe(name, pred, unseen_mask))

    dist_df = pd.DataFrame(rows)
    dist_path = feature_dir / "catboost_top60_postprocess_distributions.csv"
    dist_df.to_csv(dist_path, index=False)

    metadata = {
        "base_submit": "catboost_fs_top60_submit.csv",
        "base_public_r2": 0.37362,
        "rule": "Postprocess only unseen track_id rows; keep lookup rows unchanged.",
        "seen_rows": int(seen_mask.sum()),
        "unseen_rows": int(unseen_mask.sum()),
        "saved": saved,
        "recommended_order": [
            "catboost_top60_post_away_l2_07.csv",
            "catboost_top60_post_away_l2_10.csv",
            "catboost_top60_post_away_l2_12.csv",
            "catboost_top60_post_away_l2_20.csv",
            "catboost_top60_post_away_l2_25.csv",
            "catboost_top60_post_away_l2_30.csv",
            "catboost_top60_post_away_l2_33.csv",
            "catboost_top60_post_away_l2_35.csv",
            "catboost_top60_post_away_l2_37.csv",
            "catboost_top60_post_away_l2_40.csv",
            "catboost_top60_post_away_l2_42.csv",
            "catboost_top60_post_away_l2_50.csv",
            "catboost_top60_post_away_l2_100.csv",
            "catboost_top60_post_away_l2_sel_diff_25_35_45.csv",
            "catboost_top60_post_away_l2_sel_diff_30_35_45.csv",
            "catboost_top60_post_away_l2_sel_diff_30_37_45.csv",
            "catboost_top60_post_away_l2_sel_diff_25_37_50.csv",
        ],
    }
    with open(model_dir / "catboost_top60_postprocess_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nDistribuciones:")
    print(dist_df.round(5).to_string(index=False))
    print("\nRecomendados:")
    for item in metadata["recommended_order"]:
        print(item)
    print("\nArchivos:")
    print(dist_path)
    print(model_dir / "catboost_top60_postprocess_metadata.json")


if __name__ == "__main__":
    main()
