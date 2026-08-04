"""
app/services/profiler.py
===========================
Construit un profil statistique d'un dataset : % de valeurs manquantes (réelles et
déguisées), min/max/mean/std pour les colonnes numériques, top valeurs pour les
colonnes catégorielles. Utilisé par prompt_builder.py pour construire le "prompt avec
profilage" (approche 4 de la taxonomie du cahier des charges).
"""

import pandas as pd

DISGUISED_MISSING = {"", "na", "n/a", "unknown", " ", "nan", "none", "null"}


def profile_dataset(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Retourne un DataFrame avec une ligne par colonne : dtype détecté, % manquants
    (réels + déguisés), nombre de valeurs uniques, et statistiques adaptées au type.
    """
    records = []
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        n_na_real = series.isna().sum()
        if not is_numeric:
            n_na_disguised = (
                series.astype(str).str.strip().str.lower().isin(DISGUISED_MISSING).sum()
            )
        else:
            n_na_disguised = 0

        pct_missing = round(100 * (n_na_real + n_na_disguised) / n_rows, 2)

        record = {
            "column": col,
            "dtype_detected": "numeric" if is_numeric else "categorical/text",
            "pct_missing": pct_missing,
            "n_unique": series.nunique(dropna=True),
        }

        if is_numeric:
            record["min"] = series.min()
            record["max"] = series.max()
            record["mean"] = round(series.mean(), 2)
            record["std"] = round(series.std(), 2)
        else:
            record["top_values"] = series.value_counts().head(top_n).to_dict()

        records.append(record)

    return pd.DataFrame(records)
