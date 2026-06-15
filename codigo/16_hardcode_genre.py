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


def rank_preserve_from_score(base, score, unseen_mask):
    out = base.copy()
    base_unseen = base[unseen_mask]
    score_unseen = score[unseen_mask]
    order = np.argsort(score_unseen, kind="mergesort")
    reassigned = np.empty(len(base_unseen), dtype=float)
    reassigned[order] = np.sort(base_unseen)
    out[unseen_mask] = reassigned
    return np.clip(out, 0, 100)


def describe(name, pred, test, unseen_mask):
    s = pd.Series(pred)
    row = {
        "name": name,
        "mean": float(s.mean()),
        "std": float(s.std()),
        "unseen_mean": float(s[unseen_mask].mean()),
        "unseen_std": float(s[unseen_mask].std()),
    }
    for subgenre in ["progressive electro house", "electro house", "big room", "pop edm", "neo soul"]:
        mask = unseen_mask & test["playlist_subgenre"].eq(subgenre).to_numpy()
        row[f"mean_{subgenre.replace(' ', '_')}"] = float(s[mask].mean())
    return row


def build_shift(test, unseen_mask, recipe):
    shift = np.zeros(len(test), dtype=float)
    for col, value, amount in recipe:
        mask = unseen_mask & test[col].eq(value).to_numpy()
        shift[mask] += amount
    return shift


def main():
    root = project_root_from_cwd()
    submit_dir = root / "Submits"
    feature_dir = root / "otros csv"

    test = pd.read_csv(
        feature_dir / "test_features.csv",
        usecols=[ID_COL, "track_id", "playlist_genre", "playlist_subgenre", "genre_subgenre"],
    )
    lookup = pd.read_csv(feature_dir / "track_id_lookup.csv", usecols=["track_id"])
    unseen_mask = ~test["track_id"].isin(set(lookup["track_id"])).to_numpy()

    base_name = "et_rankblend_ensemble_preserve_30"
    base_df = pd.read_csv(submit_dir / f"{base_name}.csv")
    if not base_df[ID_COL].equals(test[ID_COL]):
        raise ValueError("IDs no alineados entre test_features y submit base.")

    ids = base_df[ID_COL]
    base = base_df[TARGET].to_numpy(dtype=float)

    recipes = {
        "popedm_up_08": [("playlist_subgenre", "pop edm", 0.8)],
        "progressive_up_08": [("playlist_subgenre", "progressive electro house", 0.8)],
        "electro_bigroom_up_06": [
            ("playlist_subgenre", "electro house", 0.6),
            ("playlist_subgenre", "big room", 0.6),
        ],
        "neo_down_04": [("playlist_subgenre", "neo soul", -0.4)],
        "neo_down_06": [("playlist_subgenre", "neo soul", -0.6)],
        "neo_down_08": [("playlist_subgenre", "neo soul", -0.8)],
        "neo_down_10": [("playlist_subgenre", "neo soul", -1.0)],
        "neo_down_12": [("playlist_subgenre", "neo soul", -1.2)],
        "neo_down_14": [("playlist_subgenre", "neo soul", -1.4)],
        "neo_down_16": [("playlist_subgenre", "neo soul", -1.6)],
        "neo_down_18": [("playlist_subgenre", "neo soul", -1.8)],
        "neo_down_20": [("playlist_subgenre", "neo soul", -2.0)],
        "neo_down_22": [("playlist_subgenre", "neo soul", -2.2)],
        "neo_down_24": [("playlist_subgenre", "neo soul", -2.4)],
        "neo_down_28": [("playlist_subgenre", "neo soul", -2.8)],
        "neo_down_32": [("playlist_subgenre", "neo soul", -3.2)],
        "neo_down_36": [("playlist_subgenre", "neo soul", -3.6)],
        "neo_down_40": [("playlist_subgenre", "neo soul", -4.0)],
        "neo_down_44": [("playlist_subgenre", "neo soul", -4.4)],
        "neo_down_48": [("playlist_subgenre", "neo soul", -4.8)],
        "neo_down_52": [("playlist_subgenre", "neo soul", -5.2)],
        "neo_down_56": [("playlist_subgenre", "neo soul", -5.6)],
        "neo_down_60": [("playlist_subgenre", "neo soul", -6.0)],
        "edm_up_neo_down_05": [
            ("playlist_genre", "edm", 0.5),
            ("playlist_subgenre", "neo soul", -0.5),
        ],
        "popedm_up_progressive_down_06": [
            ("playlist_subgenre", "pop edm", 0.6),
            ("playlist_subgenre", "progressive electro house", -0.6),
        ],
    }

    rows = [describe(base_name, base, test, unseen_mask)]
    for name, recipe in recipes.items():
        shift = build_shift(test, unseen_mask, recipe)

        direct = base.copy()
        direct[unseen_mask] = direct[unseen_mask] + shift[unseen_mask]
        direct_name = f"hardcode_genre_direct_{name}"
        save_submit(submit_dir / f"{direct_name}.csv", ids, direct)
        rows.append(describe(direct_name, direct, test, unseen_mask))

        score = base + shift
        rank = rank_preserve_from_score(base, score, unseen_mask)
        rank_name = f"hardcode_genre_rank_{name}"
        save_submit(submit_dir / f"{rank_name}.csv", ids, rank)
        rows.append(describe(rank_name, rank, test, unseen_mask))

    dist = pd.DataFrame(rows)
    dist_path = feature_dir / "hardcode_genre_distributions.csv"
    dist.to_csv(dist_path, index=False)

    print("\nDistribuciones hardcode:")
    print(dist.round(5).to_string(index=False))
    print("\nArchivo:", dist_path)


if __name__ == "__main__":
    main()
