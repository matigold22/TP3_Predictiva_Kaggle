import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINAL_NAME = "lb_probe_v9_residual_refined_primary_t3000_a075.csv"


def main():
    subprocess.run([sys.executable, "codigo/19_lb_probe_solver.py"], cwd=ROOT, check=True)
    final_path = ROOT / "Submits" / FINAL_NAME
    if not final_path.exists():
        raise FileNotFoundError(f"No se genero {final_path}")
    print("\nMejor resultado de competencia reconstruido:")
    print(final_path)


if __name__ == "__main__":
    main()
