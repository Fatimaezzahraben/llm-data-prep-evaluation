"""
app/services/prompt_builder.py
=================================
Construit automatiquement les 3 prompts (simple / schema / profile) à partir d'un
dataset donné, sans avoir besoin de coller les statistiques à la main comme on le
faisait avant. Chaque fonction retourne le texte du prompt prêt à être envoyé au LLM
via app/services/llm.py.
"""

import pandas as pd
from app.services.profiler import profile_dataset


def build_prompt_simple(dataset_name: str, n_rows: int) -> str:
    """Prompt 1 : instruction générique, aucun contexte sur le dataset."""
    return f"""Clean this dataset ({dataset_name}, {n_rows} rows). Write a Python/Pandas \
script that:
- fixes any inconsistent date formats
- handles missing values appropriately
- removes duplicate rows if any
- corrects obvious typos and inconsistent categories
- makes sure numeric columns contain only numeric values (no letters mixed in)

Return only the final cleaned Python/Pandas script, with brief comments explaining each step."""


def build_prompt_schema(df: pd.DataFrame, dataset_name: str) -> str:
    """Prompt 2 : noms de colonnes, types détectés, quelques exemples de valeurs."""
    lines = []
    for col in df.columns:
        dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical/text"
        examples = df[col].dropna().unique()[:3]
        examples_str = ", ".join(str(e) for e in examples)
        lines.append(f"- {col} ({dtype}) — e.g. {examples_str}")
    schema_block = "\n".join(lines)

    return f"""You are a data cleaning assistant. Clean the following dataset described below.

Dataset: {dataset_name}
Number of rows: {df.shape[0]}
Number of columns: {df.shape[1]}

Columns and detected types:
{schema_block}

Write a Python/Pandas script that:
- standardizes inconsistent date formats
- handles missing values per column, using a method appropriate to its type
- removes duplicate rows if any
- fixes typos and inconsistent categories in text columns
- ensures numeric columns contain only numeric values, converting or flagging any
  non-numeric entries found
- detects and handles unrealistic outlier values in numeric columns

Return only the final cleaned Python/Pandas script, with brief comments explaining each step."""


def build_prompt_profile(df: pd.DataFrame, dataset_name: str, top_n: int = 5) -> str:
    """Prompt 3 : statistiques réelles (% manquant, min/max, top valeurs)."""
    profile = profile_dataset(df, top_n=top_n)
    profile_str = profile.to_string(index=False)

    return f"""You are a data cleaning assistant. Below is a statistical profile of a \
dataset. Use it to propose a targeted, column-by-column cleaning plan, then write the \
corresponding Python/Pandas script.

Dataset: {dataset_name}
Rows: {df.shape[0]} — Columns: {df.shape[1]}

Column profile (dtype detected, % missing [real + disguised as "NA"/"unknown"/empty/etc.],
number of unique values, and either min/max/mean/std for numeric columns or top-{top_n} most
frequent values for categorical columns):

{profile_str}

Based on this profile, write a Python/Pandas script that:
- treats both real (NaN) and disguised missing values ("NA", "N/A", "unknown", empty string,
  whitespace) as missing, and imputes or flags them appropriately per column
- standardizes inconsistent date formats
- corrects typos and harmonizes inconsistent categories, especially in columns with a high
  number of unique values relative to their expected cardinality
- ensures numeric columns contain only numeric values
- flags or corrects values that fall far outside the observed min/max range for each numeric
  column (potential outliers)

Return only the final cleaned Python/Pandas script, with brief comments explaining how each
step relates to the profile above."""


PROMPT_BUILDERS = {
    "simple": lambda df, name: build_prompt_simple(name, df.shape[0]),
    "schema": lambda df, name: build_prompt_schema(df, name),
    "profile": lambda df, name: build_prompt_profile(df, name),
}


def build_prompt(prompt_type: str, df: pd.DataFrame, dataset_name: str) -> str:
    """Point d'entrée unique : build_prompt("profile", df, "hotel_bookings")."""
    if prompt_type not in PROMPT_BUILDERS:
        raise ValueError(f"prompt_type doit être {list(PROMPT_BUILDERS)}, reçu : {prompt_type}")
    return PROMPT_BUILDERS[prompt_type](df, dataset_name)
