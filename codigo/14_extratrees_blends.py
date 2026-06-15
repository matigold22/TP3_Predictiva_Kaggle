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


def align_unseen_distribution(source, target, unseen_mask):
    out = source.copy()
    src = source[unseen_mask]
    tgt = target[unseen_mask]
    src_std = src.std()
    if src_std == 0:
        out[unseen_mask] = tgt.mean()
    else:
        out[unseen_mask] = (src - src.mean()) / src_std * tgt.std() + tgt.mean()
    return np.clip(out, 0, 100)


def rank_blend_preserve_distribution(target, source, unseen_mask, weight):
    """Blend row ordering with source, then map back to target distribution."""
    out = target.copy()
    target_unseen = target[unseen_mask]
    source_unseen = source[unseen_mask]
    n = len(target_unseen)

    target_rank = pd.Series(target_unseen).rank(method="average").to_numpy()
    source_rank = pd.Series(source_unseen).rank(method="average").to_numpy()
    blended_rank = (1 - weight) * target_rank + weight * source_rank

    order = np.argsort(blended_rank, kind="mergesort")
    sorted_values = np.sort(target_unseen)
    reassigned = np.empty(n, dtype=float)
    reassigned[order] = sorted_values
    out[unseen_mask] = reassigned
    return np.clip(out, 0, 100)


def main():
    root = project_root_from_cwd()
    submit_dir = root / "Submits"
    feature_dir = root / "otros csv"

    test = pd.read_csv(feature_dir / "test_features.csv")
    lookup = pd.read_csv(feature_dir / "track_id_lookup.csv")
    unseen_mask = ~test["track_id"].isin(set(lookup["track_id"])).to_numpy()

    winner_path = submit_dir / "catboost_top60_post_away_l2_35.csv"
    et_path = submit_dir / "et_top70_leaf5_submit.csv"
    winner_df = pd.read_csv(winner_path)
    et_df = pd.read_csv(et_path)
    if not winner_df[ID_COL].equals(et_df[ID_COL]):
        raise ValueError("IDs no alineados entre winner y ExtraTrees.")

    ids = winner_df[ID_COL]
    winner = winner_df[TARGET].to_numpy(dtype=float)
    et = et_df[TARGET].to_numpy(dtype=float)
    et_aligned = align_unseen_distribution(et, winner, unseen_mask)

    et_sources = {
        "top45": submit_dir / "et_top45_leaf4_submit.csv",
        "top70": submit_dir / "et_top70_leaf5_submit.csv",
        "top100": submit_dir / "et_top100_leaf8_submit.csv",
    }
    aligned_sources = {}
    for source_name, source_path in et_sources.items():
        if not source_path.exists():
            continue
        source_df = pd.read_csv(source_path)
        if not winner_df[ID_COL].equals(source_df[ID_COL]):
            raise ValueError(f"IDs no alineados en {source_path.name}")
        source = source_df[TARGET].to_numpy(dtype=float)
        aligned_sources[source_name] = align_unseen_distribution(source, winner, unseen_mask)

    if aligned_sources:
        et_aligned_ensemble = np.mean(np.vstack(list(aligned_sources.values())), axis=0)
    else:
        et_aligned_ensemble = et_aligned

    variants = {}
    for weight in [0.02, 0.03, 0.05]:
        name = f"blend_winner95_et{int(weight * 100):02d}" if weight == 0.05 else f"blend_winner_et{int(weight * 100):02d}"
        variants[f"et_{name}"] = (1 - weight) * winner + weight * et

    for weight in [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]:
        variants[f"et_aligned_blend_{int(weight * 100):02d}"] = (1 - weight) * winner + weight * et_aligned

    for source_name, source in aligned_sources.items():
        if source_name == "top70":
            continue
        for weight in [0.05, 0.08]:
            variants[f"et_{source_name}_aligned_blend_{int(weight * 100):02d}"] = (
                (1 - weight) * winner + weight * source
            )

    for weight in [0.05, 0.08, 0.10]:
        variants[f"et_ensemble_aligned_blend_{int(weight * 100):02d}"] = (
            (1 - weight) * winner + weight * et_aligned_ensemble
        )

    for weight in [0.10, 0.20, 0.30, 0.32, 0.35, 0.40]:
        variants[f"et_rankblend_preserve_{int(weight * 100):02d}"] = rank_blend_preserve_distribution(
            winner, et, unseen_mask, weight
        )

    for weight in [0.10, 0.20, 0.30, 0.32, 0.35, 0.40]:
        variants[f"et_rankblend_ensemble_preserve_{int(weight * 100):02d}"] = rank_blend_preserve_distribution(
            winner, et_aligned_ensemble, unseen_mask, weight
        )

    rows = [
        describe("winner_away_l2_35", winner, unseen_mask),
        describe("et_top70_leaf5", et, unseen_mask),
        describe("et_top70_leaf5_aligned", et_aligned, unseen_mask),
        describe("et_aligned_ensemble", et_aligned_ensemble, unseen_mask),
    ]
    saved = []
    for name, pred in variants.items():
        path = submit_dir / f"{name}.csv"
        saved.append(save_submit(path, ids, pred))
        rows.append(describe(name, pred, unseen_mask))

    dist = pd.DataFrame(rows)
    dist_path = feature_dir / "extratrees_blends_distributions.csv"
    dist.to_csv(dist_path, index=False)
    print("\nDistribuciones:")
    print(dist.round(5).to_string(index=False))
    print("\nArchivo:", dist_path)


if __name__ == "__main__":
    main()
