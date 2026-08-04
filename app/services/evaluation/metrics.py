"""
app/services/evaluation/metrics.py
=====================================
Calcule TOUTES les métriques de qualité d'un workflow de nettoyage généré par LLM :
accuracy, precision, recall, F1-score — au niveau colonne ET global.

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

  precision = correct_repairs / (correct_repairs + failed_repairs)
              -> parmi les cellules modifiées par le LLM, combien sont correctes
  recall    = correct_repairs / total_errors
              -> parmi toutes les erreurs injectées, combien ont été corrigées
  f1_score  = 2 * precision * recall / (precision + recall)
  accuracy  = correct_repairs / total_errors
              -> équivalent au recall ICI car on ne mesure que sur les cellules
                 où une erreur était connue (pas de vrai négatif à ce niveau)

En complément, ce module calcule aussi une "accuracy globale" au niveau de TOUT le
dataset (cellules corrompues + cellules jamais touchées), pour vérifier que le LLM n'a
pas dégradé des cellules qui étaient déjà correctes ("sur-nettoyage").
"""

import json
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


def evaluate_column(col, sub_log, df_clean, df_cleaned, n_samples=3):
    correct_repairs = 0
    failed_repairs = 0
    regressions = 0
    success_samples = []
    failed_samples = []

    total_errors = len(sub_log)

    for _, row in sub_log.iterrows():
        idx = row["row_index"]
        injected_value = row["injected_value"]

        if idx not in df_cleaned.index or idx not in df_clean.index:
            continue

        original_value = df_clean.at[idx, col]
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
        else:
            failed_repairs += 1
            if len(failed_samples) < n_samples:
                failed_samples.append({
                    "id": str(idx), "dirty": _safe_str(injected_value),
                    "truth": _safe_str(original_value), "llm_output": _safe_str(cleaned_value),
                })

    precision = correct_repairs / (correct_repairs + failed_repairs) if (correct_repairs + failed_repairs) > 0 else 0.0
    recall = correct_repairs / total_errors if total_errors > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = correct_repairs / total_errors if total_errors > 0 else 0.0

    return {
        "total_errors": total_errors,
        "correct_repairs": correct_repairs,
        "failed_repairs": failed_repairs,
        "regressions": regressions,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "success_samples": success_samples,
        "failed_samples": failed_samples,
    }


def _safe_str(v):
    return None if pd.isna(v) else str(v)


def compute_global_accuracy(df_clean, df_cleaned):
    """
    Accuracy calculée sur TOUTES les cellules du dataset (pas seulement les cellules où
    une erreur a été injectée) : vérifie que le LLM n'a pas modifié par erreur des
    cellules qui étaient déjà correctes.
    """
    common_cols = [c for c in df_clean.columns if c in df_cleaned.columns]
    common_idx = df_clean.index.intersection(df_cleaned.index)
    if len(common_idx) == 0 or len(common_cols) == 0:
        return None

    total = 0
    correct = 0
    for col in common_cols:
        clean_series = df_clean.loc[common_idx, col]
        cleaned_series = df_cleaned.loc[common_idx, col]
        for idx in common_idx:
            total += 1
            if _values_equal(cleaned_series.at[idx], clean_series.at[idx]):
                correct += 1

    return round(correct / total, 4) if total > 0 else None


def evaluate_workflow(df_clean, df_noisy, df_cleaned, error_log, workflow_name="unnamed"):
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

    columns_report = {}
    total_correct = 0
    total_failed = 0
    total_regressions = 0

    for col in error_log["column"].unique():
        if col not in df_cleaned.columns:
            continue
        sub_log = error_log[error_log["column"] == col]
        col_result = evaluate_column(col, sub_log, df_clean, df_cleaned)
        columns_report[col] = col_result
        total_correct += col_result["correct_repairs"]
        total_failed += col_result["failed_repairs"]
        total_regressions += col_result["regressions"]

    total_errors_end = total_failed + total_regressions
    global_precision = total_correct / (total_correct + total_failed) if (total_correct + total_failed) > 0 else 0.0
    global_recall = total_correct / total_errors_start if total_errors_start > 0 else 0.0
    global_f1 = (2 * global_precision * global_recall / (global_precision + global_recall)
                 if (global_precision + global_recall) > 0 else 0.0)
    global_accuracy_on_errors = total_correct / total_errors_start if total_errors_start > 0 else 0.0
    global_accuracy_all_cells = compute_global_accuracy(df_clean, df_cleaned)

    report = {
        "workflow_name": workflow_name,
        "rows_expected": int(rows_expected),
        "rows_remaining": int(rows_remaining),
        "rows_lost": int(rows_lost),
        "rows_lost_pct": rows_lost_pct,
        "errors_lost_with_deleted_rows": errors_on_missing_rows,
        "warning_over_cleaning": rows_lost_pct > 5.0,
        "global_accuracy_on_error_cells": round(global_accuracy_on_errors, 4),
        "global_accuracy_all_cells": global_accuracy_all_cells,
        "global_precision": round(global_precision, 4),
        "global_recall": round(global_recall, 4),
        "global_f1": round(global_f1, 4),
        "global_initial_error_rate": round(total_errors_start / total_cells, 4),
        "global_final_error_rate": round(total_errors_end / total_cells, 4),
        "total_cells": total_cells,
        "total_errors_start": total_errors_start,
        "total_errors_end": total_errors_end,
        "total_fixed": total_correct,
        "total_failed_repairs": total_failed,
        "total_regressions": total_regressions,
        "columns": columns_report,
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
    args = parser.parse_args()

    df_clean = pd.read_csv(args.clean, low_memory=False)
    df_noisy = pd.read_csv(args.noisy, low_memory=False)
    df_cleaned = pd.read_csv(args.cleaned, low_memory=False)
    error_log = pd.read_csv(args.error_log, low_memory=False)

    report = evaluate_workflow(df_clean, df_noisy, df_cleaned, error_log, workflow_name=args.name)

    output_path = args.output or f"results/metrics/benchmark_{args.name}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Rapport sauvegardé : {output_path}")
    print(f"accuracy(errors)={report['global_accuracy_on_error_cells']} "
          f"accuracy(all)={report['global_accuracy_all_cells']} "
          f"precision={report['global_precision']} recall={report['global_recall']} "
          f"f1={report['global_f1']}")


if __name__ == "__main__":
    main()
