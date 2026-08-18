"""
notebooks/run_validation_loop_all.py
=======================================
Lance la boucle de validation (Phase 3, approche 6) sur les 4 datasets, avec le
prompt "profile" (le plus performant jusqu'ici), jusqu'a MAX_ITERATIONS iterations
(5 par defaut) ou F1>=TARGET_F1 (0.90 par defaut).

ATTENTION : ceci fait jusqu'a MAX_ITERATIONS x plus d'appels API que
generate_all_workflows.py (un appel par iteration). Prevoir le cout/quota
Mistral API en consequence (5 iterations = jusqu'a 5x plus d'appels).

Usage:
    python notebooks/run_validation_loop_all.py
    python notebooks/run_validation_loop_all.py --dataset titanic
    python notebooks/run_validation_loop_all.py --max-iterations 8
    python notebooks/run_validation_loop_all.py --target-f1 0.95
    python notebooks/run_validation_loop_all.py --dataset titanic --max-iterations 10 --target-f1 0.95
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd

from app.services.validation_loop import run_validation_loop
from build_error_log_from_dirty import build_error_log_from_dirty

DATASETS = {
    "hotel-booking-demand": {
        "dir": Path("datasets/hotel-booking-demand"),
        "clean": "clean.csv", "noisy": "noisy_medium.csv", "error_log": "error_log_medium.csv",
    },
    "titanic": {
        "dir": Path("datasets/titanic"),
        "clean": "clean.csv", "noisy": "noisy_medium.csv", "error_log": "error_log_medium.csv",
    },
    "hospital": {
        "dir": Path("datasets/hospital"),
        "clean": "clean.csv", "noisy": "dirty.csv", "error_log": None,
    },
    "flights": {
        "dir": Path("datasets/flights"),
        "clean": "clean.csv", "noisy": "dirty.csv", "error_log": None,
    },
}

PROMPT_TYPE = "profile"   # le plus performant jusqu'ici
PROVIDER = "mistral_api"
MAX_ITERATIONS = 5   # valeur par defaut ; surchargeable avec --max-iterations
TARGET_F1 = 0.90      # valeur par defaut ; surchargeable avec --target-f1


def get_error_log(dataset_name, cfg):
    if cfg["error_log"] is not None:
        return pd.read_csv(cfg["dir"] / cfg["error_log"], low_memory=False)
    df_clean = pd.read_csv(cfg["dir"] / cfg["clean"], low_memory=False)
    df_dirty = pd.read_csv(cfg["dir"] / cfg["noisy"], low_memory=False)
    return build_error_log_from_dirty(df_clean, df_dirty, dataset_name)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Boucle de validation sur un ou plusieurs datasets")
    parser.add_argument("--dataset", default=None,
                         help="Nom d'un seul dataset a traiter (ex: titanic). "
                              "Omis = traite les 4 datasets.")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS,
                         help=f"Nombre maximum d'iterations de la boucle de validation "
                              f"(defaut : {MAX_ITERATIONS}). Chaque iteration supplementaire "
                              f"= un appel API en plus par dataset.")
    parser.add_argument("--target-f1", type=float, default=TARGET_F1,
                         help=f"F1 cible : la boucle s'arrete des que ce score est atteint "
                              f"(defaut : {TARGET_F1}).")
    args = parser.parse_args()

    max_iterations = args.max_iterations
    target_f1 = args.target_f1

    datasets_to_run = DATASETS
    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"Dataset inconnu : '{args.dataset}'. Choix possibles : {list(DATASETS.keys())}")
            return
        datasets_to_run = {args.dataset: DATASETS[args.dataset]}

    print(f"[config] max_iterations={max_iterations} | target_f1={target_f1}")

    all_results = {}

    for dataset_name, cfg in datasets_to_run.items():
        clean_path = cfg["dir"] / cfg["clean"]
        noisy_path = cfg["dir"] / cfg["noisy"]
        if not clean_path.exists() or not noisy_path.exists():
            print(f"SKIP {dataset_name}")
            continue

        print(f"\n=== {dataset_name} ===")
        df_clean = pd.read_csv(clean_path, low_memory=False)
        error_log = get_error_log(dataset_name, cfg)

        result = run_validation_loop(
            str(noisy_path), dataset_name, PROMPT_TYPE, df_clean, error_log,
            provider=PROVIDER, max_iterations=max_iterations, target_f1=target_f1,
        )
        all_results[dataset_name] = {
            "best_f1": result["best_f1"],
            "best_iteration": result["best_iteration"],
            "history": result["history"],
            "saved_path": result["saved_path"],
        }
        print(f"  MEILLEUR : F1={result['best_f1']} (iteration {result['best_iteration']})")

    out_path = Path("results/metrics_tables/validation_loop_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)

    print("\n=== RESUME FINAL ===")
    for name, r in all_results.items():
        status = "ATTEINT" if (r["best_f1"] or 0) >= target_f1 else "pas atteint"
        print(f"  {name:22s} | F1={r['best_f1']} | {status}")
    print(f"\nSauvegarde : {out_path}")


if __name__ == "__main__":
    main()