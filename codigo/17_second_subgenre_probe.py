from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "track_popularity"
ID_COL = "ID"


def project_root_from_cwd():
    cwd = Path.cwd()
    return cwd.parent if cwd.name.lower() in ["codigo", "codigos"] else cwd


def rank_preserve_from_score(base, score, unseen_mask):
    out = base.copy()
    base_unseen = base[unseen_mask]
    score_unseen = score[unseen_mask]
    order = np.argsort(score_unseen, kind="mergesort")
    reassigned = np.empty(len(base_unseen), dtype=float)
    reassigned[order] = np.sort(base_unseen)
    out[unseen_mask] = reassigned
    return np.clip(out, 0, 100)


def validate_submit(submit, expected_ids):
    if list(submit.columns) != [ID_COL, TARGET]:
        raise ValueError(f"Columnas invalidas: {submit.columns.tolist()}")
    if not submit[ID_COL].equals(expected_ids):
        raise ValueError("IDs invalidos o desalineados.")
    if submit[ID_COL].duplicated().any():
        raise ValueError("IDs duplicados.")
    if submit[TARGET].isna().any():
        raise ValueError("Predicciones nulas.")
    if not submit[TARGET].between(0, 100).all():
        raise ValueError("Predicciones fuera de rango.")


def main():
    root = project_root_from_cwd()
    submit_dir = root / "Submits"
    feature_dir = root / "otros csv"

    test = pd.read_csv(
        feature_dir / "test_features.csv",
        usecols=[ID_COL, "track_id", "playlist_subgenre"],
    )
    lookup = pd.read_csv(feature_dir / "track_id_lookup.csv", usecols=["track_id"])
    unseen_mask = ~test["track_id"].isin(set(lookup["track_id"])).to_numpy()

    base_df = pd.read_csv(submit_dir / "hardcode_genre_rank_neo_down_60.csv")
    if not base_df[ID_COL].equals(test[ID_COL]):
        raise ValueError("IDs no alineados entre test_features y submit base.")

    base = base_df[TARGET].to_numpy(dtype=float)
    progressive_mask = (
        unseen_mask
        & test["playlist_subgenre"].eq("progressive electro house").to_numpy()
    )

    for label, amount in [
        ("04", 0.4),
        ("08", 0.8),
        ("12", 1.2),
        ("24", 2.4),
        ("36", 3.6),
        ("48", 4.8),
    ]:
        score = base.copy()
        score[progressive_mask] -= amount
        pred = rank_preserve_from_score(base, score, unseen_mask)
        name = f"hardcode_rank_neo60_progressive_down_{label}.csv"
        submit = pd.DataFrame({ID_COL: base_df[ID_COL], TARGET: pred})
        validate_submit(submit, base_df[ID_COL])
        submit.to_csv(submit_dir / name, index=False)

        diff = np.abs(pred - base)
        print(
            f"{name}: changed={(diff > 1e-12).sum()}, "
            f"mean_abs={diff.mean():.6f}, max_abs={diff.max():.6f}"
        )


if __name__ == "__main__":
    main()
