"""
app/services/evaluation/metrics.py
=====================================
Calcule TOUTES les métriques de qualité d'un workflow de nettoyage généré par LLM :
accuracy, precision, recall, F1-score — au niveau colonne ET global.

IMPORTANT — format de sortie aligné sur l'exemple fourni par l'encadrante
(benchmark_results_Hospital_pattern_openai-gpt-5_5_codestral-2508_round3.json) :
les clés top-level et par-colonne suivent EXACTEMENT les memes noms que cet exemple
(ex: "failed_repair" au singulier, pas "failed_repairs" ; "initial_error_rate" et
"final_error_rate" par colonne ; "regression_samples" en plus de "success_samples" et
"failed_samples" ; "pipeline_time_seconds" au top-level). Des champs supplementaires
utiles a notre propre diagnostic (rows_lost_pct, by_error_family, etc.) sont conserves
EN PLUS de ces champs standards, jamais a la place.

Comparaison à 4 sources :
  - df_clean   : version PROPRE de référence (vérité absolue)
  - df_noisy   : version BRUITÉE envoyée au LLM
  - df_cleaned : sortie du LLM (résultat du script généré, après exécution)
  - error_log  : journal des erreurs injectées (row_index, column, original_value,
                 injected_value, error_type, error_family)

Définitions (sur les cellules où une erreur a été injectée) :
  - correct_repair : cellule nettoyée == valeur originale propre (réparation réussie)
  - failed_repair  : cellule nettoyée != originale ET != valeur bruitée
                      (le LLM a modifié la cellule, mais vers une mauvaise valeur)
  - regression     : cellule nettoyée == valeur bruitée (rien n'a été corrigé)

  precision = correct_repairs / (correct_repairs + failed_repair)
  recall    = correct_repairs / total_errors
  f1_score  = 2 * precision * recall / (precision + recall)
  initial_error_rate (par colonne) = total_errors / n_rows
  final_error_rate   (par colonne) = (failed_repair + regressions) / n_rows
"""

import json
import time
import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def _values_equal(a, b):
    """Compare deux valeurs de façon tolérante (types, espaces, casse pour le texte)."""
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        pass
    return str(a).strip().lower() == str(b).strip().lower()


def _safe_str(v):
    return None if pd.isna(v) else str(v)


def evaluate_column(col, sub_log, df_clean, df_cleaned, n_rows, n_samples=3, clean_col=None):
    """
    col : nom de colonne cote 'dirty'/genere (utilise pour lire df_cleaned).
    clean_col : nom de colonne cote 'clean.csv' (utilise pour lire df_clean). Si None,
                on suppose que le meme nom est valide des deux cotes.
    n_rows : nombre total de lignes du dataset, utilise comme denominateur pour
             initial_error_rate / final_error_rate (format attendu par l'encadrante).
    """
    clean_col = clean_col or col

    correct_repairs = 0
    failed_repair = 0
    regressions = 0
    success_samples = []
    failed_samples = []
    regression_samples = []

    total_errors = len(sub_log)

    for _, row in sub_log.iterrows():
        idx = row["row_index"]
        injected_value = row["injected_value"]

        if idx not in df_cleaned.index or idx not in df_clean.index:
            continue
        if col not in df_cleaned.columns or clean_col not in df_clean.columns:
            continue

        original_value = df_clean.at[idx, clean_col]
        cleaned_value = df_cleaned.at[idx, col]

        if _values_equal(cleaned_value, original_value):
            correct_repairs += 1
            if len(success_samples) < n_samples:
                success_samples.append({
                    "id": str(idx), "dirty": _safe_str(injected_value),
                    "truth": _safe_str(original_value), "llm_output": _safe_str(cleaned_value),
                })
        elif _values_equal(cleaned_value, injected_value):
            regressions += 1
            if len(regression_samples) < n_samples:
                regression_samples.append({
                    "id": str(idx), "dirty": _safe_str(injected_value),
                    "truth": _safe_str(original_value), "llm_output": _safe_str(cleaned_value),
                })
        else:
            failed_repair += 1
            if len(failed_samples) < n_samples:
                failed_samples.append({
                    "id": str(idx), "dirty": _safe_str(injected_value),
                    "truth": _safe_str(original_value), "llm_output": _safe_str(cleaned_value),
                })

    precision = correct_repairs / (correct_repairs + failed_repair) if (correct_repairs + failed_repair) > 0 else 0.0
    recall = correct_repairs / total_errors if total_errors > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    initial_error_rate = round(total_errors / n_rows, 4) if n_rows > 0 else 0.0
    final_error_rate = round((failed_repair + regressions) / n_rows, 4) if n_rows > 0 else 0.0

    return {
        "initial_error_rate": initial_error_rate,
        "final_error_rate": final_error_rate,
        "total_errors": total_errors,
        "correct_repairs": correct_repairs,
        "failed_repair": failed_repair,
        "regressions": regressions,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "success_samples": success_samples,
        "failed_samples": failed_samples,
        "regression_samples": regression_samples,
    }


def compute_global_accuracy(df_clean, df_cleaned, col_name_map=None):
    """
    Accuracy calculée sur TOUTES les cellules du dataset (pas seulement les cellules où
    une erreur a été injectée) : vérifie que le LLM n'a pas modifié par erreur des
    cellules qui étaient déjà correctes. Champ supplémentaire (pas dans le format de
    référence de l'encadrante, mais utile pour notre propre diagnostic).
    """
    col_name_map = col_name_map or {}
    common_idx = df_clean.index.intersection(df_cleaned.index)

    pairs = []
    for cleaned_col in df_cleaned.columns:
        clean_col = col_name_map.get(cleaned_col, cleaned_col)
        if clean_col in df_clean.columns:
            pairs.append((clean_col, cleaned_col))

    if len(common_idx) == 0 or len(pairs) == 0:
        return None

    total = 0
    correct = 0
    for clean_col, cleaned_col in pairs:
        clean_series = df_clean.loc[common_idx, clean_col]
        cleaned_series = df_cleaned.loc[common_idx, cleaned_col]
        for idx in common_idx:
            total += 1
            if _values_equal(cleaned_series.at[idx], clean_series.at[idx]):
                correct += 1

    return round(correct / total, 4) if total > 0 else None


def evaluate_workflow(df_clean, df_noisy, df_cleaned, error_log, workflow_name="unnamed",
                       pipeline_time_seconds=None):
    """
    pipeline_time_seconds : temps total (génération + exécution), optionnel — champ
    attendu par l'encadrante dans son format de référence. Passer la somme de
    latency_seconds (workflow_generator) + execution_time_seconds (safe_executor) si
    disponible ; laissé à None sinon (n'apparaît alors pas dans le JSON, comme dans
    l'exemple de référence pipeline_time_seconds=222.0 correspondait à un run réel).
    """
    n_rows = df_clean.shape[0]
    total_cells = df_clean.shape[0] * df_clean.shape[1]
    total_errors_start = len(error_log)

    # --- Détection de la perte de lignes (sur-nettoyage via dropna()/drop trop agressif) ---
    rows_expected = df_clean.shape[0]
    rows_remaining = df_cleaned.shape[0]
    rows_lost = rows_expected - rows_remaining
    rows_lost_pct = round(100 * rows_lost / rows_expected, 2) if rows_expected > 0 else 0.0

    errors_on_missing_rows = int(
        (~error_log["row_index"].isin(df_cleaned.index)).sum()
    )

    col_name_map = {}
    if "column_clean_csv" in error_log.columns:
        pairs = error_log[["column", "column_clean_csv"]].drop_duplicates()
        col_name_map = dict(zip(pairs["column"], pairs["column_clean_csv"]))

    columns_report = {}
    total_correct = 0
    total_failed = 0
    total_regressions = 0

    family_stats = {}

    for col in error_log["column"].unique():
        if col not in df_cleaned.columns:
            continue
        sub_log = error_log[error_log["column"] == col]
        clean_col = col_name_map.get(col, col)
        col_result = evaluate_column(col, sub_log, df_clean, df_cleaned, n_rows, clean_col=clean_col)
        columns_report[col] = col_result
        total_correct += col_result["correct_repairs"]
        total_failed += col_result["failed_repair"]
        total_regressions += col_result["regressions"]

        for family in sub_log["error_family"].unique():
            fam_sub_log = sub_log[sub_log["error_family"] == family]
            fam_result = evaluate_column(col, fam_sub_log, df_clean, df_cleaned, n_rows, clean_col=clean_col)
            if family not in family_stats:
                family_stats[family] = {"correct": 0, "failed": 0, "regressions": 0, "total": 0}
            family_stats[family]["correct"] += fam_result["correct_repairs"]
            family_stats[family]["failed"] += fam_result["failed_repair"]
            family_stats[family]["regressions"] += fam_result["regressions"]
            family_stats[family]["total"] += fam_result["total_errors"]

    family_report = {}
    for family, s in family_stats.items():
        prec = s["correct"] / (s["correct"] + s["failed"]) if (s["correct"] + s["failed"]) > 0 else 0.0
        rec = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        family_report[family] = {
            "total_errors": s["total"], "correct_repairs": s["correct"],
            "failed_repair": s["failed"], "regressions": s["regressions"],
            "precision": round(prec, 4), "recall": round(rec, 4), "f1_score": round(f1, 4),
        }

    total_errors_end = total_failed + total_regressions
    global_precision = total_correct / (total_correct + total_failed) if (total_correct + total_failed) > 0 else 0.0
    global_recall = total_correct / total_errors_start if total_errors_start > 0 else 0.0
    global_f1 = (2 * global_precision * global_recall / (global_precision + global_recall)
                 if (global_precision + global_recall) > 0 else 0.0)
    global_accuracy_all_cells = compute_global_accuracy(df_clean, df_cleaned, col_name_map=col_name_map)

    report = {
        # --- Champs EXACTS du format de reference de l'encadrante (meme noms/ordre) ---
        "global_precision": round(global_precision, 4),
        "global_recall": round(global_recall, 4),
        "global_f1": round(global_f1, 4),
        "global_initial_error_rate": round(total_errors_start / total_cells, 4),
        "global_final_error_rate": round(total_errors_end / total_cells, 4),
        "total_cells": total_cells,
        "total_errors_start": total_errors_start,
        "total_errors_end": total_errors_end,
        "total_fixed": total_correct,
        "total_regressions": total_regressions,
        "columns": columns_report,
        "pipeline_time_seconds": pipeline_time_seconds,
        # --- Champs supplementaires (diagnostic interne, en plus, jamais a la place) ---
        "workflow_name": workflow_name,
        "rows_expected": int(rows_expected),
        "rows_remaining": int(rows_remaining),
        "rows_lost": int(rows_lost),
        "rows_lost_pct": rows_lost_pct,
        "errors_lost_with_deleted_rows": errors_on_missing_rows,
        "warning_over_cleaning": rows_lost_pct > 5.0,
        "global_accuracy_all_cells": global_accuracy_all_cells,
        "total_failed_repairs": total_failed,
        "by_error_family": family_report,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Évalue un workflow de nettoyage LLM")
    parser.add_argument("--clean", required=True)
    parser.add_argument("--noisy", required=True)
    parser.add_argument("--cleaned", required=True)
    parser.add_argument("--error_log", required=True)
    parser.add_argument("--name", default="unnamed")
    parser.add_argument("--output", default=None)
    parser.add_argument("--pipeline_time_seconds", type=float, default=None)
    args = parser.parse_args()

    df_clean = pd.read_csv(args.clean, low_memory=False)
    df_noisy = pd.read_csv(args.noisy, low_memory=False)
    df_cleaned = pd.read_csv(args.cleaned, low_memory=False)
    error_log = pd.read_csv(args.error_log, low_memory=False)

    report = evaluate_workflow(df_clean, df_noisy, df_cleaned, error_log, workflow_name=args.name,
                                pipeline_time_seconds=args.pipeline_time_seconds)

    output_path = args.output or f"results/metrics/benchmark_{args.name}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Rapport sauvegardé : {output_path}")
    print(f"precision={report['global_precision']} recall={report['global_recall']} "
          f"f1={report['global_f1']}")


if __name__ == "__main__":
    main()