"""
notebooks/evaluate_all_workflows.py
======================================
Execute et evalue les workflows generes pour les 4 datasets (hotel, titanic,
hospital, flights), Mistral API uniquement, prompts schema + profile uniquement.
Produit un tableau de comparaison unique pour le rapport.

Usage:
    python notebooks/evaluate_all_workflows.py
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.services.safe_executor import execute_workflow
from app.services.evaluation.metrics import evaluate_workflow
from app.services.evaluation.traceability import generate_traceability_report
from build_error_log_from_dirty import build_error_log_from_dirty

# --- Les 4 datasets : (nom, dossier, fichier bruite, fichier error_log) ---
# Pour hotel/titanic : error_log deja genere par le notebook d'injection.
# Pour hospital/flights : error_log construit a la volee (clean.csv vs dirty.csv).
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
        "clean": "clean.csv", "noisy": "dirty.csv", "error_log": None,  # construit ci-dessous
    },
    "flights": {
        "dir": Path("datasets/flights"),
        "clean": "clean.csv", "noisy": "dirty.csv", "error_log": None,
    },
}

PROMPT_TYPES = ["schema", "profile"]
PROVIDERS = ["mistral_api"]

GENERATED_DIR = Path("workflows/generated")


def get_error_log(dataset_name: str, cfg: dict) -> pd.DataFrame:
    """Charge l'error_log existant, ou le construit a la volee pour hospital/flights."""
    if cfg["error_log"] is not None:
        return pd.read_csv(cfg["dir"] / cfg["error_log"], low_memory=False)

    df_clean = pd.read_csv(cfg["dir"] / cfg["clean"], low_memory=False)
    df_dirty = pd.read_csv(cfg["dir"] / cfg["noisy"], low_memory=False)
    return build_error_log_from_dirty(df_clean, df_dirty, dataset_name)


def main():
    summary_rows = []

    for dataset_name, cfg in DATASETS.items():
        clean_path = cfg["dir"] / cfg["clean"]
        noisy_path = cfg["dir"] / cfg["noisy"]

        if not clean_path.exists() or not noisy_path.exists():
            print(f"SKIP {dataset_name} : fichiers introuvables dans {cfg['dir']}")
            continue

        df_clean = pd.read_csv(clean_path, low_memory=False)
        error_log = get_error_log(dataset_name, cfg)
        print(f"[DEBUG] {dataset_name} | df_clean.shape={df_clean.shape} | "
              f"error_log rows={len(error_log)}")

        for provider in PROVIDERS:
            for prompt_type in PROMPT_TYPES:
                workflow_name = f"{dataset_name}_{prompt_type}_{provider}"
                script_path = GENERATED_DIR / f"workflow_{dataset_name}_{prompt_type}_{provider}.py"

                print(f"--- {workflow_name} ---")
                row = {"dataset": dataset_name, "provider": provider, "prompt_type": prompt_type,
                       "execution_success": False, "error": None,
                       "rows_lost_pct": None, "global_f1": None,
                       "global_precision": None, "global_recall": None,
                       "global_accuracy_all_cells": None, "warning_over_cleaning": None}

                if not script_path.exists():
                    row["error"] = f"Script introuvable : {script_path}"
                    print(f"  SKIP : {row['error']}")
                    summary_rows.append(row)
                    continue

                exec_result = execute_workflow(str(script_path), str(noisy_path),
                                                workflow_name, timeout=120,
                                                dataset_name=dataset_name)

                if not exec_result["success"]:
                    row["error"] = exec_result["error"]
                    print(f"  ECHEC EXECUTION : {row['error']}")
                    summary_rows.append(row)
                    continue

                row["execution_success"] = True
                df_cleaned = pd.read_csv(exec_result["output_path"], low_memory=False)
                print(f"  [DEBUG] output_path={exec_result['output_path']} | "
                      f"result_variable_used={exec_result['result_variable_used']} | "
                      f"df_cleaned.shape={df_cleaned.shape}")

                pipeline_time = round(exec_result["execution_time_seconds"], 2)
                report = evaluate_workflow(df_clean, None, df_cleaned, error_log,
                                            workflow_name=workflow_name,
                                            pipeline_time_seconds=pipeline_time)
                generate_traceability_report(df_clean, df_cleaned, error_log,
                                              workflow_name=workflow_name)

                # Sauvegarde le rapport complet (format attendu par l'encadrante), un
                # fichier JSON par (dataset, prompt_type, provider).
                results_dir = Path("results/benchmark_results")
                results_dir.mkdir(parents=True, exist_ok=True)
                benchmark_json_path = results_dir / f"benchmark_results_{workflow_name}.json"
                with open(benchmark_json_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"  [benchmark] Rapport JSON sauvegarde : {benchmark_json_path}")

                row.update({
                    "rows_lost_pct": report["rows_lost_pct"],
                    "global_f1": report["global_f1"],
                    "global_precision": report["global_precision"],
                    "global_recall": report["global_recall"],
                    "global_accuracy_all_cells": report["global_accuracy_all_cells"],
                    "warning_over_cleaning": report["warning_over_cleaning"],
                })
                print(f"  OK : F1={report['global_f1']} rows_lost={report['rows_lost_pct']}%")

                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    out_dir = Path("results/metrics_tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_summary_all_datasets.csv"
    summary_df.to_csv(out_path, index=False)

    print("\n=== TABLEAU DE COMPARAISON (4 datasets) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSauvegarde : {out_path}")

    # Verification rapide de l'objectif F1 >= 0.90 fixe par l'encadrante
    print("\n=== Objectif F1 >= 0.90 ===")
    for _, r in summary_df.iterrows():
        if r["global_f1"] is not None and pd.notna(r["global_f1"]):
            status = "ATTEINT" if r["global_f1"] >= 0.90 else "PAS ATTEINT"
            print(f"  {r['dataset']:22s} | {r['prompt_type']:8s} | "
                  f"F1={r['global_f1']:.4f} | {status}")


if __name__ == "__main__":
    main()