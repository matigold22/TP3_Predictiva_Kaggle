import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


STEPS = [
    [sys.executable, "-m", "jupyter", "execute", "codigo/2_feature_engineering.ipynb"],
    [sys.executable, "codigo/3_catboost_best_submit.py"],
    [sys.executable, "codigo/5_catboost_feature_selection.py"],
    [sys.executable, "codigo/7_catboost_top60_hypercv.py"],
    [sys.executable, "codigo/11_catboost_top60_postprocess.py"],
    [sys.executable, "codigo/13_extratrees_models.py"],
    [sys.executable, "codigo/14_extratrees_blends.py"],
    [sys.executable, "codigo/16_hardcode_genre.py"],
    [sys.executable, "codigo/17_second_subgenre_probe.py"],
]


def run_step(command):
    print("\n" + "=" * 100)
    print("Ejecutando:", " ".join(command))
    print("=" * 100)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    train_path = ROOT / "bases de datos" / "base_train.csv"
    test_path = ROOT / "bases de datos" / "base_val.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Faltan bases de datos/base_train.csv y/o bases de datos/base_val.csv."
        )

    for command in STEPS:
        run_step(command)

    final_path = ROOT / "Submits" / "hardcode_rank_neo60_progressive_down_24.csv"
    print("\nModelo final reproducible generado:")
    print(final_path)


if __name__ == "__main__":
    main()
