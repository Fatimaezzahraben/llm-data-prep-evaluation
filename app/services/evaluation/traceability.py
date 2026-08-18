"""
app/services/evaluation/traceability.py
==========================================
Génère un rapport de TRAÇABILITÉ lisible : pour chaque erreur injectée, montre la
valeur d'origine (vérité), la valeur bruitée (ce que voyait le LLM), la valeur produite
par le LLM après nettoyage, et le statut de la correction.

Produit 2 fichiers :
  - un CSV complet (une ligne par erreur) -> results/traceability/traceability_<name>.csv
  - un résumé Markdown (statistiques + exemples) -> results/traceability/traceability_<name>.md
"""

from pathlib import Path

import pandas as pd

from app.services.evaluation.metrics import _values_equal, _safe_str

TRACEABILITY_DIR = Path("results/traceability")


def build_traceability_table(df_clean: pd.DataFrame, df_cleaned: pd.DataFrame,
                              error_log: pd.DataFrame) -> pd.DataFrame:
    """
    Construit une table détaillée : une ligne par erreur injectée, avec le statut de
    la correction (Corrigé / Non corrigé / Mal corrigé).
    """
    has_dual_naming = "column_clean_csv" in error_log.columns

    rows = []
    for _, log_row in error_log.iterrows():
        idx = log_row["row_index"]
        col = log_row["column"]                 # nom cote df_cleaned
        clean_col = log_row["column_clean_csv"] if has_dual_naming else col  # nom cote df_clean

        if (col not in df_cleaned.columns or clean_col not in df_clean.columns
                or idx not in df_cleaned.index or idx not in df_clean.index):
            status = "Colonne/ligne absente après exécution"
            llm_output = None
            original_value = None
        else:
            original_value = df_clean.at[idx, clean_col]
            dirty_value = log_row["injected_value"]
            llm_output = df_cleaned.at[idx, col]

            if _values_equal(llm_output, original_value):
                status = "Corrigé"
            elif _values_equal(llm_output, dirty_value):
                status = "Non corrigé (regression)"
            else:
                status = "Mal corrigé (valeur différente mais fausse)"

        rows.append({
            "row_index": idx,
            "column": col,
            "error_family": log_row.get("error_family"),
            "error_type": log_row.get("error_type"),
            "valeur_originale": _safe_str(original_value),
            "valeur_bruitee_vue_par_llm": _safe_str(log_row["injected_value"]),
            "valeur_produite_par_llm": _safe_str(llm_output),
            "statut": status,
        })

    return pd.DataFrame(rows)


def generate_traceability_report(df_clean, df_cleaned, error_log, workflow_name="unnamed"):
    """
    Génère le rapport complet (CSV + résumé Markdown) et le sauvegarde dans
    results/traceability/.
    """
    table = build_traceability_table(df_clean, df_cleaned, error_log)

    TRACEABILITY_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TRACEABILITY_DIR / f"traceability_{workflow_name}.csv"
    table.to_csv(csv_path, index=False)

    status_counts = table["statut"].value_counts()
    total = len(table)

    md_lines = [
        f"# Rapport de traçabilité — {workflow_name}",
        "",
        f"**Total d'erreurs suivies :** {total}",
        "",
        "## Répartition des statuts",
        "",
        "| Statut | Nombre | % |",
        "|---|---|---|",
    ]
    for status, count in status_counts.items():
        pct = round(100 * count / total, 1) if total else 0
        md_lines.append(f"| {status} | {count} | {pct}% |")

    md_lines += ["", "## Répartition par famille d'erreur", "",
                 "| Famille | Corrigé | Non corrigé | Mal corrigé |", "|---|---|---|---|"]
    for family in table["error_family"].unique():
        sub = table[table["error_family"] == family]
        c = (sub["statut"] == "Corrigé").sum()
        nc = (sub["statut"] == "Non corrigé (regression)").sum()
        mc = (sub["statut"] == "Mal corrigé (valeur différente mais fausse)").sum()
        md_lines.append(f"| {family} | {c} | {nc} | {mc} |")

    md_lines += ["", "## Exemples de corrections réussies", ""]
    examples_ok = table[table["statut"] == "Corrigé"].head(5)
    for _, r in examples_ok.iterrows():
        md_lines.append(f"- **{r['column']}** (ligne {r['row_index']}) : "
                         f"`{r['valeur_bruitee_vue_par_llm']}` → `{r['valeur_produite_par_llm']}` "
                         f"(attendu : `{r['valeur_originale']}`)")

    md_lines += ["", "## Exemples de corrections manquées ou erronées", ""]
    examples_bad = table[table["statut"] != "Corrigé"].head(5)
    for _, r in examples_bad.iterrows():
        md_lines.append(f"- **{r['column']}** (ligne {r['row_index']}, {r['statut']}) : "
                         f"`{r['valeur_bruitee_vue_par_llm']}` → `{r['valeur_produite_par_llm']}` "
                         f"(attendu : `{r['valeur_originale']}`)")

    md_path = TRACEABILITY_DIR / f"traceability_{workflow_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return {"csv_path": str(csv_path), "md_path": str(md_path), "table": table}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Génère le rapport de traçabilité")
    parser.add_argument("--clean", required=True)
    parser.add_argument("--cleaned", required=True)
    parser.add_argument("--error_log", required=True)
    parser.add_argument("--name", default="unnamed")
    args = parser.parse_args()

    df_clean = pd.read_csv(args.clean, low_memory=False)
    df_cleaned = pd.read_csv(args.cleaned, low_memory=False)
    error_log = pd.read_csv(args.error_log, low_memory=False)

    result = generate_traceability_report(df_clean, df_cleaned, error_log, workflow_name=args.name)
    print(f"CSV : {result['csv_path']}")
    print(f"Markdown : {result['md_path']}")