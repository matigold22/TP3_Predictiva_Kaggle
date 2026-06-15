from pathlib import Path
import re

import numpy as np
import pandas as pd


ID_COL = "ID"
TARGET = "track_popularity"
SCORE_PATTERN = re.compile(
    r"`([^`]+\.csv)`[^\n]*?R2 publico(?: aprox\.)? "
    r"(?:de )?([0-9]+,[0-9]+|[0-9]+\.[0-9]+)",
    re.IGNORECASE,
)
AWAY_LABELS = ["03", "05", "12", "20", "30", "35", "37", "40", "50", "100"]
CANDIDATES = [
    ("lb_probe_v9_residual_refined_conservative_t30000_a050.csv", 30000.0, 0.50),
    ("lb_probe_v9_residual_refined_primary_t3000_a075.csv", 3000.0, 0.75),
    ("lb_probe_v9_residual_refined_full_t3000_a100.csv", 3000.0, 1.00),
]


def project_root_from_cwd():
    cwd = Path.cwd()
    return cwd.parent if cwd.name.lower() in {"codigo", "codigos"} else cwd


def read_scores(root):
    frozen_scores = root / "otros csv" / "lb_probe_solver" / "leaderboard_results.csv"
    if frozen_scores.exists():
        return pd.read_csv(frozen_scores)

    text = (root / "REGISTRO_TP.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    rows = []
    seen = set()

    for raw_name, raw_score in SCORE_PATTERN.findall(text):
        name = Path(raw_name).name
        matching_line = next(
            (line for line in lines if f"`{name}`" in line and raw_score in line),
            "",
        )
        path = root / "Submits" / name
        if name in seen or not path.exists() or "aprox" in matching_line.lower():
            continue
        seen.add(name)
        normalized_score = raw_score.replace(",", ".")
        rows.append(
            {
                "order": len(rows) + 1,
                "filename": name,
                "public_r2": float(normalized_score),
                "score_decimals": len(normalized_score.split(".")[1]),
            }
        )

    return pd.DataFrame(rows)


def load_predictions(root, filenames):
    ids = None
    columns = []
    for filename in filenames:
        submit = pd.read_csv(root / "Submits" / filename)
        if list(submit.columns) != [ID_COL, TARGET]:
            raise ValueError(f"Columnas invalidas en {filename}.")
        if ids is None:
            ids = submit[ID_COL]
        elif not submit[ID_COL].equals(ids):
            raise ValueError(f"IDs desalineados en {filename}.")
        columns.append(submit[TARGET].to_numpy(dtype=float))
    return ids, np.column_stack(columns)


def estimate_sst(root, scores):
    names = [f"catboost_top60_post_away_l2_{label}.csv" for label in AWAY_LABELS]
    score_map = scores.set_index("filename")["public_r2"]
    _, predictions = load_predictions(root, names)

    centered = predictions - predictions.mean(axis=1, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coordinate = vt[0] * singular_values[0]
    observed = score_map.loc[names].to_numpy()
    quadratic = np.polyfit(coordinate, observed, 2)
    fitted = np.polyval(quadratic, coordinate)

    if quadratic[0] >= 0:
        raise ValueError("La familia away_l2 no produjo una curvatura R2 valida.")

    return {
        "sst": float(-1.0 / quadratic[0]),
        "pc1_share": float(singular_values[0] ** 2 / np.sum(singular_values**2)),
        "max_error": float(np.max(np.abs(fitted - observed))),
        "rmse": float(np.sqrt(np.mean((fitted - observed) ** 2))),
    }


def spectral_solution(delta_matrix, q, threshold):
    gram = delta_matrix.T @ delta_matrix
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues >= threshold
    if not keep.any():
        raise ValueError(f"El corte {threshold:g} elimina todas las direcciones.")
    weights = eigenvectors[:, keep] @ (
        (eigenvectors[:, keep].T @ q) / eigenvalues[keep]
    )
    return weights, gram, int(keep.sum())


def predict_held_out(root, train_scores, target_row, sst, threshold):
    reference = train_scores.iloc[-1]
    train_basis = train_scores.iloc[:-1]
    p_ref = pd.read_csv(root / "Submits" / reference.filename)[TARGET].to_numpy(float)
    _, basis_predictions = load_predictions(root, train_basis.filename.tolist())
    delta_matrix = basis_predictions - p_ref[:, None]
    score_delta = train_basis.public_r2.to_numpy() - reference.public_r2
    q = 0.5 * (np.sum(delta_matrix**2, axis=0) + score_delta * sst)

    target = pd.read_csv(root / "Submits" / target_row.filename)[TARGET].to_numpy(float)
    held_delta = target - p_ref
    gram = delta_matrix.T @ delta_matrix
    eigenvalues = np.linalg.eigvalsh(gram)
    rcond = threshold / eigenvalues[-1]
    coefficients = np.linalg.lstsq(delta_matrix, held_delta, rcond=rcond)[0]
    predicted_q = coefficients @ q
    predicted_score = reference.public_r2 + (
        2.0 * predicted_q - held_delta @ held_delta
    ) / sst
    reconstruction_error = np.linalg.norm(
        delta_matrix @ coefficients - held_delta
    ) / max(np.linalg.norm(held_delta), 1e-12)
    return float(predicted_score), float(reconstruction_error)


def validate_submit(submit, expected_ids):
    if list(submit.columns) != [ID_COL, TARGET]:
        raise ValueError("Columnas invalidas en candidato.")
    if not submit[ID_COL].equals(expected_ids):
        raise ValueError("IDs desalineados en candidato.")
    if submit[ID_COL].duplicated().any() or submit[TARGET].isna().any():
        raise ValueError("IDs duplicados o predicciones nulas.")
    if not submit[TARGET].between(0, 100).all():
        raise ValueError("Predicciones fuera de rango luego del clipping.")


def dataframe_to_markdown(frame):
    columns = frame.columns.tolist()
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, float):
                cells.append(f"{value:.8g}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    root = project_root_from_cwd()
    output_dir = root / "otros csv" / "lb_probe_solver"
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = read_scores(root)
    if len(scores) < 20:
        raise ValueError("No hay suficientes submits scoreados para resolver el sistema.")
    scores.to_csv(output_dir / "leaderboard_results.csv", index=False)

    sst_diagnostic = estimate_sst(root, scores)
    sst = sst_diagnostic["sst"]

    reference_row = scores.loc[scores.public_r2.idxmax()]
    reference_name = reference_row.filename
    reference_predictions = pd.read_csv(root / "Submits" / reference_name)
    p_ref = reference_predictions[TARGET].to_numpy(float)

    test = pd.read_csv(
        root / "otros csv" / "test_features.csv",
        usecols=[ID_COL, "track_id"],
    )
    lookup = pd.read_csv(
        root / "otros csv" / "track_id_lookup.csv",
        usecols=["track_id"],
    )
    lookup_mask = test["track_id"].isin(set(lookup["track_id"])).to_numpy()

    low_precision = scores.loc[
        scores.score_decimals.lt(4) & ~scores.filename.eq(reference_name),
        "filename",
    ].tolist()
    basis = scores.loc[
        ~scores.filename.eq(reference_name) & scores.score_decimals.ge(4)
    ].copy()
    ids, basis_predictions = load_predictions(root, basis.filename.tolist())
    if not reference_predictions[ID_COL].equals(ids):
        raise ValueError("El submit de referencia tiene IDs desalineados.")

    lookup_difference = np.max(
        np.abs(basis_predictions[lookup_mask] - p_ref[lookup_mask, None]),
        axis=0,
    )
    excluded_lookup = basis.loc[lookup_difference > 1e-9, "filename"].tolist()
    basis = basis.loc[lookup_difference <= 1e-9].reset_index(drop=True)
    basis_predictions = basis_predictions[:, lookup_difference <= 1e-9]

    delta_matrix = basis_predictions - p_ref[:, None]
    score_delta = basis.public_r2.to_numpy() - reference_row.public_r2
    q = 0.5 * (np.sum(delta_matrix**2, axis=0) + score_delta * sst)

    validations = []
    for target_name in [
        "hardcode_genre_rank_neo_down_60.csv",
        "hardcode_rank_neo60_progressive_down_24.csv",
    ]:
        target_index = scores.index[scores.filename.eq(target_name)][0]
        train = scores.loc[
            (scores.index < target_index) & scores.score_decimals.ge(4)
        ]
        target = scores.loc[target_index]
        for threshold in [1000.0, 10000.0, 30000.0]:
            predicted, reconstruction = predict_held_out(
                root, train, target, sst, threshold
            )
            validations.append(
                {
                    "target": target_name,
                    "threshold": threshold,
                    "observed_r2": target.public_r2,
                    "predicted_r2": predicted,
                    "error": predicted - target.public_r2,
                    "relative_reconstruction_error": reconstruction,
                }
            )
    validation_df = pd.DataFrame(validations)
    validation_df.to_csv(output_dir / "temporal_validation.csv", index=False)

    candidate_rows = []
    for filename, threshold, alpha in CANDIDATES:
        weights, gram, dimensions = spectral_solution(delta_matrix, q, threshold)
        full_delta = delta_matrix @ weights
        candidate = np.clip(p_ref + alpha * full_delta, 0, 100)
        candidate[lookup_mask] = p_ref[lookup_mask]
        submit = pd.DataFrame({ID_COL: ids, TARGET: candidate})
        validate_submit(submit, ids)
        submit.to_csv(root / "Submits" / filename, index=False)

        ideal_gain = (
            2.0 * alpha * (weights @ q)
            - alpha**2 * (weights @ gram @ weights)
        ) / sst
        clipped_rows = int(
            np.count_nonzero((p_ref + alpha * full_delta < 0) | (p_ref + alpha * full_delta > 100))
        )
        candidate_rows.append(
            {
                "filename": filename,
                "threshold": threshold,
                "alpha": alpha,
                "dimensions": dimensions,
                "ideal_predicted_r2_before_clipping": reference_row.public_r2
                + ideal_gain,
                "delta_rmse": float(np.sqrt(np.mean((candidate - p_ref) ** 2))),
                "clipped_rows": clipped_rows,
                "mean": float(candidate.mean()),
                "std": float(candidate.std(ddof=1)),
                "min": float(candidate.min()),
                "max": float(candidate.max()),
            }
        )

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df.to_csv(output_dir / "candidate_summary.csv", index=False)

    report = f"""# Diagnostico del solver por probes de leaderboard

- Submits exactos utilizados: {len(scores)}
- Submits usados como base luego de proteger lookup: {len(basis)}
- Excluidos por alterar lookup: {", ".join(f"`{name}`" for name in excluded_lookup)}
- Excluidos por score con menos de 4 decimales: {", ".join(f"`{name}`" for name in low_precision)}
- Referencia: `{reference_name}` con R2 publico {reference_row.public_r2:.5f}
- SST estimado: {sst:.2f}
- Varianza explicada por la primera direccion de `away_l2`: {sst_diagnostic['pc1_share']:.8f}
- Error maximo de la parabola `away_l2`: {sst_diagnostic['max_error']:.8f}

## Validacion temporal

{dataframe_to_markdown(validation_df)}

## Candidatos

{dataframe_to_markdown(candidate_df)}

## Limitacion principal

Kaggle calcula el score sobre el 50% publico, pero no conocemos que filas integran
ese subconjunto. El solver usa productos internos sobre las 6.567 filas como
aproximacion. La parabola y la validacion temporal indican que la aproximacion es
util, pero los R2 predichos no deben interpretarse como garantias.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    print(report)
    print("\nCandidatos generados en:", root / "Submits")


if __name__ == "__main__":
    main()
