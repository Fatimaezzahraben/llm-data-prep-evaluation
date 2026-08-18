"""
notebooks/build_error_log_from_dirty.py (ou app/services/error_log_builder.py)
=================================================================================
Pour les datasets Raha (hospital, flights) qui fournissent directement dirty.csv +
clean.csv (erreurs REELLES, pas injectees), on construit un error_log au MEME FORMAT
que celui genere par inject_all() sur hotel_bookings/titanic, afin que metrics.py et
traceability.py fonctionnent de facon identique sur les 4 datasets.

Hypothese : dirty.csv et clean.csv ont le meme nombre de lignes, dans le meme ordre
(vrai pour les datasets Raha standards). Si ce n'est pas le cas pour un dataset donne,
il faut d'abord aligner les deux fichiers par une cle commune avant d'utiliser ce script.
"""

from pathlib import Path
import pandas as pd


def _normalize_col_name(name: str) -> str:
    """Normalise un nom de colonne pour la comparaison : minuscules, sans underscore
    ni espace. Permet de faire correspondre 'ProviderNumber' avec 'provider_number'."""
    return str(name).lower().replace("_", "").replace(" ", "")


def _match_columns(df_clean: pd.DataFrame, df_dirty: pd.DataFrame) -> list:
    """
    Retourne une liste de tuples (col_clean, col_dirty) pour les colonnes qui
    correspondent au meme champ, meme si leur nom differe en casse/underscores
    (ex: 'ZipCode' <-> 'zip_code' <-> 'zip'). Fait d'abord une correspondance exacte
    normalisee, puis une correspondance par inclusion (l'un contient l'autre) pour les
    cas type 'ZipCode' <-> 'zip' ou 'HospitalName' <-> 'name'.
    """
    clean_norm = {_normalize_col_name(c): c for c in df_clean.columns}
    dirty_norm = {_normalize_col_name(c): c for c in df_dirty.columns}

    pairs = []
    used_dirty = set()

    # 1) correspondance exacte normalisee
    for norm, clean_col in clean_norm.items():
        if norm in dirty_norm:
            pairs.append((clean_col, dirty_norm[norm]))
            used_dirty.add(dirty_norm[norm])

    # 2) correspondance par inclusion pour les noms restants (ex: 'zipcode' vs 'zip')
    remaining_clean = {n: c for n, c in clean_norm.items()
                        if c not in [p[0] for p in pairs]}
    remaining_dirty = {n: c for n, c in dirty_norm.items() if c not in used_dirty}

    for norm_c, clean_col in remaining_clean.items():
        best_match = None
        for norm_d, dirty_col in remaining_dirty.items():
            if dirty_col in used_dirty:
                continue
            if norm_c in norm_d or norm_d in norm_c:
                best_match = dirty_col
                break
        if best_match:
            pairs.append((clean_col, best_match))
            used_dirty.add(best_match)

    return pairs


def _values_equal(a, b) -> bool:
    """Compare deux valeurs de facon tolerante aux types (ex: int 10018 vs str '10018'),
    pour eviter de detecter de fausses erreurs quand clean.csv et dirty.csv n'ont pas
    ete lus avec le meme typage de colonnes par pandas."""
    a_na, b_na = pd.isna(a), pd.isna(b)
    if a_na and b_na:
        return True
    if a_na or b_na:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (ValueError, TypeError):
        pass
    return str(a).strip() == str(b).strip()


def build_error_log_from_dirty(df_clean: pd.DataFrame, df_dirty: pd.DataFrame,
                                 dataset_name: str) -> pd.DataFrame:
    """
    Compare clean et dirty cellule par cellule et construit un DataFrame au format :
    row_index, column, original_value, injected_value, error_type, error_family,
    noise_level.

    Fait correspondre les colonnes intelligemment meme si leurs noms different entre
    les deux fichiers (ex: 'ProviderNumber' dans clean.csv vs 'provider_number' dans
    dirty.csv -- cas reel observe sur le dataset hospital de Raha).
    """
    if df_clean.shape[0] != df_dirty.shape[0]:
        raise ValueError(
            f"clean.csv ({df_clean.shape[0]} lignes) et dirty.csv "
            f"({df_dirty.shape[0]} lignes) n'ont pas le meme nombre de lignes pour "
            f"'{dataset_name}' -- alignement manuel necessaire avant d'utiliser ce script."
        )

    column_pairs = _match_columns(df_clean, df_dirty)
    print(f"[{dataset_name}] {len(column_pairs)} colonnes appariees entre clean et dirty "
          f"(sur {df_clean.shape[1]} / {df_dirty.shape[1]})")
    unmatched_clean = set(df_clean.columns) - {p[0] for p in column_pairs}
    if unmatched_clean:
        print(f"[{dataset_name}] Colonnes de clean.csv sans correspondance : {unmatched_clean}")

    records = []

    for clean_col, dirty_col in column_pairs:
        clean_series = df_clean[clean_col].reset_index(drop=True)
        dirty_series = df_dirty[dirty_col].reset_index(drop=True)

        for idx in range(len(clean_series)):
            clean_val = clean_series.iloc[idx]
            dirty_val = dirty_series.iloc[idx]

            if _values_equal(clean_val, dirty_val):
                continue

            if pd.isna(dirty_val):
                error_type = "real_missing_value"
                error_family = "missing_values"
            else:
                error_type = "real_dirty_value"
                error_family = "real_world_error"

            records.append({
                "row_index": idx,
                "column": dirty_col,          # nom COTE DIRTY : utilise pour lire df_cleaned
                "column_clean_csv": clean_col,  # nom COTE CLEAN : utilise pour lire df_clean
                "original_value": clean_val,
                "injected_value": dirty_val,
                "error_type": error_type,
                "error_family": error_family,
                "noise_level": "real",
            })

    EXPECTED_COLUMNS = ["row_index", "column", "column_clean_csv", "original_value",
                        "injected_value", "error_type", "error_family", "noise_level"]
    log = pd.DataFrame(records, columns=EXPECTED_COLUMNS)
    print(f"[{dataset_name}] {len(log)} erreurs reelles detectees "
          f"({len(log) / (df_clean.shape[0] * max(len(column_pairs), 1)) * 100:.2f}% des cellules)")
    return log


def build_and_save(dataset_dir: str, dataset_name: str):
    """Lit clean.csv + dirty.csv dans dataset_dir, sauvegarde error_log.csv au meme endroit."""
    data_dir = Path(dataset_dir)
    df_clean = pd.read_csv(data_dir / "clean.csv", low_memory=False)
    df_dirty = pd.read_csv(data_dir / "dirty.csv", low_memory=False)

    log = build_error_log_from_dirty(df_clean, df_dirty, dataset_name)
    log.to_csv(data_dir / "error_log.csv", index=False)
    print(f"Sauvegarde : {data_dir / 'error_log.csv'}")
    return log


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    args = parser.parse_args()
    build_and_save(args.dataset_dir, args.dataset_name)