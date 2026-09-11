"""
app/services/agent_c.py
=====================================
Agent C -- l'agent evaluateur, COMPLETEMENT INDEPENDANT du pipeline de generation
(validation_loop.py). Se lance separement, APRES qu'un script de nettoyage a ete
execute, pour inspecter le resultat et calculer deux pourcentages de proprete.

CONCEPTION VOULUE PAR L'ENCADRANTE -- AUCUNE CONNAISSANCE D'UN DATASET PRECIS N'EST
CODEE EN DUR ICI : ni noms de colonnes, ni cles de dependances fonctionnelles, ni
listes de valeurs valides. Tout est DECOUVERT AUTOMATIQUEMENT a partir du DataFrame
fourni -- donner un dataset totalement nouveau a cet agent doit fonctionner sans
modifier ce fichier ni ecrire une seule ligne de configuration specifique a ce
dataset. C'est la difference cle avec la version precedente (qui utilisait un
fichier agent_c_config.py avec une entree par dataset) : cette configuration
externe n'existe plus, tout est infere directement depuis les donnees.

REGLE CENTRALE POUR LE CONTROLE SEMANTIQUE (consigne de l'encadrante) : le LLM
evaluateur doit distinguer une VRAIE erreur (faute de frappe, valeur cassee --
ex: "Pariis" au lieu de "Paris") d'une VARIANTE ACCEPTABLE (abreviation standard,
format alternatif valide -- ex: "US" pour "United States" n'est PAS une erreur).
"""
import json
import re

import pandas as pd

from app.services.llm import call_llm


# ---------------------------------------------------------------------------
# AUTO-DECOUVERTE DES TYPES DE COLONNES -- remplace toute configuration
# manuelle (expected_columns/numeric_columns/date_columns/semantic_columns).
# ---------------------------------------------------------------------------

def _normalize_col_name(name: str) -> str:
    """Normalise un nom de colonne pour la comparaison/le matching automatique
    (insensible a la casse et aux underscores) -- permet de faire correspondre
    'ProviderNumber' et 'provider_number' sans configuration explicite."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def _is_id_like_column(series: pd.Series, n_rows: int) -> bool:
    """Detecte une colonne de type identifiant (quasi-unique par ligne) --
    a exclure de la plupart des controles (FD, semantique) car elle ne porte
    pas de categorie/valeur repetee a verifier."""
    if n_rows == 0:
        return False
    n_unique = series.nunique(dropna=True)
    return (n_unique / n_rows) > 0.9


def auto_infer_column_types(df: pd.DataFrame) -> dict:
    """
    Classe automatiquement chaque colonne d'un DataFrame, sans aucune
    configuration prealable -- uniquement a partir de son dtype, son nom, et
    sa distribution de valeurs.

    Returns
    -------
    dict:
        {
            "numeric_columns": [...],      # dtype numerique OU >=90% des valeurs
                                            # se parsent comme un nombre
            "date_like_columns": {col: subtype, ...},  # subtype in
                                            # {"year","day","month_num","month_name","full_date"}
                                            # detecte par le NOM de la colonne
                                            # (heuristique : contient year/annee,
                                            # day/jour, month/mois, date) + un
                                            # sondage de plausibilite des valeurs
            "categorical_columns": [...],  # dtype objet/string, cardinalite
                                            # compatible avec une categorie
                                            # (ni quasi-constante, ni quasi-unique)
            "id_like_columns": [...],      # quasi-uniques, exclues des autres
                                            # categories (identifiants)
        }
    """
    n_rows = len(df)
    numeric_columns, date_like_columns = [], {}
    categorical_columns, id_like_columns = [], []

    for col in df.columns:
        series = df[col]
        col_norm = _normalize_col_name(col)

        if _is_id_like_column(series, n_rows):
            id_like_columns.append(col)
            continue

        is_numeric_dtype = pd.api.types.is_numeric_dtype(series)
        if is_numeric_dtype:
            numeric_columns.append(col)
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Une colonne texte peut en fait etre numerique-en-tant-que-chaine :
            # si >=90% des valeurs non nulles se parsent comme un nombre, la
            # traiter comme numerique plutot que categorielle. IMPORTANT : on ne
            # retire que les caracteres de FORMATAGE numerique courants ($, %,
            # virgules de milliers, espaces) -- jamais les LETTRES. Un bug reel
            # corrige ici : retirer aussi les lettres faisait passer une adresse
            # comme "1720 university blvd" pour "1720" (numerique), la
            # classant a tort comme colonne numerique au lieu de categorielle/texte.
            non_null = series.dropna()
            if len(non_null) > 0:
                cleaned_for_check = non_null.astype(str).str.strip().str.replace(
                    r'^[\$€£]|[,%\s]', '', regex=True
                )
                parsed = pd.to_numeric(cleaned_for_check, errors='coerce')
                if parsed.notna().mean() >= 0.9:
                    numeric_columns.append(col)
                    continue

            # Detection heuristique de colonnes de type date, par le NOM de la
            # colonne (pas de config externe -- juste des mots-cles courants).
            if re.search(r'year|annee|_yr$', col_norm):
                date_like_columns[col] = "year"
            elif re.search(r'day.?of.?month|jour', col_norm) and 'week' not in col_norm:
                date_like_columns[col] = "day"
            elif re.search(r'month.?num|mois.?num', col_norm):
                date_like_columns[col] = "month_num"
            elif re.search(r'month|mois', col_norm):
                date_like_columns[col] = "month_name"
            elif re.search(r'date', col_norm):
                date_like_columns[col] = "full_date"
            else:
                n_unique = series.nunique(dropna=True)
                if n_unique >= 2:
                    categorical_columns.append(col)

    return {
        "numeric_columns": numeric_columns,
        "date_like_columns": date_like_columns,
        "categorical_columns": categorical_columns,
        "id_like_columns": id_like_columns,
    }


def auto_discover_fd_pairs(df: pd.DataFrame, min_group_size: int = 2,
                            purity_threshold: float = 0.98,
                            max_target_columns: int = 30) -> list:
    """
    DECOUVERTE AUTOMATIQUE de dependances fonctionnelles EXACTES, sans aucune
    connaissance prealable du dataset -- meme principe que la technique
    d'auto-decouverte deja donnee au LLM generateur dans prompt_builder.py, mais
    ici executee directement en code pour l'agent evaluateur.

    Pour chaque colonne candidate a etre une CIBLE (target), essaie chaque autre
    colonne comme cle (source) et ne retient QUE les paires ou :
      - la cle determine la cible a (quasi-)100% de purete parmi les groupes qui
        ont plus d'une ligne (les groupes de taille 1 sont ignores car triviaux
        -- un groupe d'une seule ligne "determine" n'importe quoi par
        definition, sans aucune information reelle) ;
      - la taille MEDIANE des groupes (parmi les groupes de taille > 1) est
        >= min_group_size, pour eviter le piege des cles quasi-uniques qui
        semblent "exactes" par accident.

    Cette fonction ne teste que des cles a UNE SEULE colonne (pas de paires
    composites) pour rester rapide sur des datasets larges -- une cle composite
    connue peut toujours etre fournie explicitement si necessaire, mais n'est
    plus requise pour un usage generique.

    Returns
    -------
    list de tuples (source_col, target_col) pour chaque dependance EXACTE
    decouverte.
    """
    n_rows = len(df)
    if n_rows == 0:
        return []

    types = auto_infer_column_types(df)
    # Candidats CLE : colonnes categorielles OU numeriques (un code numerique
    # comme un zip ou un provider_number peut parfaitement etre une cle de
    # regroupement valide, meme s'il est classe "numerique" par
    # auto_infer_column_types -- la seule chose qui compte ici est qu'il ait
    # une repetition suffisante, verifiee juste apres via grouped_sizes).
    key_candidates = types["categorical_columns"] + types["numeric_columns"]
    # Candidats CIBLE : toute colonne qui n'est pas elle-meme la cle testee.
    target_candidates = (types["categorical_columns"] + types["numeric_columns"])[:max_target_columns]

    discovered = []
    for key_col in key_candidates:
        grouped_sizes = df.groupby(key_col, dropna=True).size()
        multi_row_groups = grouped_sizes[grouped_sizes >= 2]
        if len(multi_row_groups) == 0:
            continue  # cle sans aucun groupe >1 ligne -> aucune information a verifier
        if multi_row_groups.median() < min_group_size:
            continue  # trop peu de redondance pour etre une decouverte fiable

        for target_col in target_candidates:
            if target_col == key_col:
                continue
            nunique_per_group = df.groupby(key_col, dropna=True)[target_col].nunique(dropna=True)
            # Ne considere que les groupes de taille > 1 pour le calcul de purete
            relevant_groups = nunique_per_group[grouped_sizes[nunique_per_group.index] >= 2]
            if len(relevant_groups) == 0:
                continue
            purity = (relevant_groups <= 1).mean()
            if purity >= purity_threshold:
                discovered.append((key_col, target_col))

    return discovered


def auto_discover_composite_fd_pairs(df: pd.DataFrame, single_column_fd_pairs: list = None,
                                      min_group_size: int = 2, purity_threshold: float = 0.98,
                                      max_key_candidates: int = 15) -> list:
    """
    Extension de auto_discover_fd_pairs() aux cles COMPOSITES (2 colonnes) --
    necessaire pour des cas comme (ProviderNumber, MeasureCode) -> Score, ou
    aucune des deux colonnes seule ne determine la cible, mais la PAIRE si.

    Pour rester rapide meme sur un dataset large, ne teste PAS toutes les
    paires possibles : se limite aux `max_key_candidates` colonnes qui
    apparaissent DEJA comme cle ou cible dans au moins une dependance a une
    seule colonne (single_column_fd_pairs) -- ces colonnes ont deja montre un
    signal structurel, donc combiner deux d'entre elles est un candidat
    raisonnable, sans exploser combinatoirement sur toutes les colonnes.

    Returns
    -------
    list de tuples ((source_col1, source_col2), target_col).
    """
    n_rows = len(df)
    if n_rows == 0:
        return []

    types = auto_infer_column_types(df)
    all_candidates = types["categorical_columns"] + types["numeric_columns"]

    if single_column_fd_pairs:
        structural_cols = set()
        for source, target in single_column_fd_pairs:
            structural_cols.add(source)
            structural_cols.add(target)
        key_pool = [c for c in all_candidates if c in structural_cols][:max_key_candidates]
    else:
        key_pool = all_candidates[:max_key_candidates]

    target_candidates = all_candidates

    # Pour eviter de reporter des centaines de paires composites REDONDANTES
    # (ex: (HospitalName, City) -> State n'apporte rien de nouveau si
    # HospitalName -> State est deja une FD a une seule colonne connue) : on
    # ignore toute paire composite dont la cible est DEJA couverte par une FD
    # a une seule colonne impliquant l'UN des deux membres de la cle.
    already_covered = set()
    if single_column_fd_pairs:
        for source, target in single_column_fd_pairs:
            already_covered.add((source, target))

    discovered = []
    for i in range(len(key_pool)):
        for j in range(i + 1, len(key_pool)):
            key_cols = [key_pool[i], key_pool[j]]
            grouped_sizes = df.groupby(key_cols, dropna=True).size()
            multi_row_groups = grouped_sizes[grouped_sizes >= 2]
            if len(multi_row_groups) == 0 or multi_row_groups.median() < min_group_size:
                continue

            for target_col in target_candidates:
                if target_col in key_cols:
                    continue
                # Ignore si deja couvert par une FD a une seule colonne
                if (key_cols[0], target_col) in already_covered or (key_cols[1], target_col) in already_covered:
                    continue
                nunique_per_group = df.groupby(key_cols, dropna=True)[target_col].nunique(dropna=True)
                relevant_groups = nunique_per_group[grouped_sizes[nunique_per_group.index] >= 2]
                if len(relevant_groups) == 0:
                    continue
                purity = (relevant_groups <= 1).mean()
                if purity >= purity_threshold:
                    discovered.append((tuple(key_cols), target_col))

    return discovered


# ---------------------------------------------------------------------------
# RULE-BASED CHECKS (no LLM call — fast and deterministic)
# ---------------------------------------------------------------------------

def check_data_loss(df_original: pd.DataFrame, df_cleaned: pd.DataFrame,
                     threshold: float = 0.30) -> list:
    """Detecte une perte de lignes excessive (ex: un dropna() trop agressif)."""
    issues = []
    initial = len(df_original)
    final = len(df_cleaned)
    if initial == 0:
        return issues
    loss_pct = (initial - final) / initial
    if loss_pct > threshold:
        issues.append(
            f"CRITICAL_DATA_LOSS: the script dropped {loss_pct:.1%} of rows "
            f"({initial} -> {final}). Threshold is {threshold:.0%}. The final row "
            f"count must equal the original row count — check for any dropna() or "
            f"filtering without an explicit subset=[...] of only the columns being "
            f"cleaned."
        )
    return issues


def check_column_preservation(df_original: pd.DataFrame, df_cleaned: pd.DataFrame,
                               expected_columns: list = None) -> list:
    """Detecte une colonne supprimee par erreur."""
    issues = []
    expected = expected_columns if expected_columns is not None else list(df_original.columns)
    for col in expected:
        if col not in df_cleaned.columns:
            issues.append(f"CRITICAL: column '{col}' was dropped from the output — "
                           f"every original column must still be present.")
    return issues


def check_empty_columns(df_cleaned: pd.DataFrame, columns_to_check: list = None) -> list:
    """Detecte une colonne totalement videe (toutes les valeurs sont NaN ou une
    chaine "vide" deguisee) -- signe d'un bug de nettoyage trop agressif."""
    issues = []
    invalid_null_strings = {"nan", "null", "none", "<na>", "", " "}
    columns = columns_to_check if columns_to_check is not None else list(df_cleaned.columns)
    for col in columns:
        if col not in df_cleaned.columns:
            continue
        is_string_null = df_cleaned[col].astype(str).str.lower().str.strip().isin(invalid_null_strings)
        is_real_null = df_cleaned[col].isna()
        if (is_real_null | is_string_null).all():
            issues.append(f"CRITICAL: column '{col}' is completely empty after "
                           f"cleaning (every value is missing) — this is almost "
                           f"certainly a bug, not a correct result.")
    return issues


def check_date_validity(df_cleaned: pd.DataFrame, date_columns: dict) -> list:
    """
    date_columns : dict {nom_colonne: sous-type}, sous-type in
        {"year", "day", "month_name", "month_num", "full_date"}.
    Verifie que les valeurs sont dans une plage plausible / un format valide.
    """
    issues = []
    current_year = pd.Timestamp.now().year

    for col, subtype in date_columns.items():
        if col not in df_cleaned.columns:
            continue
        series = df_cleaned[col]

        if subtype == "year":
            numeric = pd.to_numeric(series, errors="coerce")
            bad_mask = ((numeric < 1900) | (numeric > current_year + 2)) & numeric.notna()
            if bad_mask.any():
                bad_values = numeric[bad_mask].unique()[:5].tolist()
                issues.append(f"Column '{col}' contains out-of-range year values: "
                               f"{bad_values}. Plausible range is 1900-{current_year + 2}.")

        elif subtype == "day":
            numeric = pd.to_numeric(series, errors="coerce")
            bad_mask = ((numeric < 1) | (numeric > 31)) & numeric.notna()
            if bad_mask.any():
                issues.append(f"Column '{col}' contains invalid day-of-month values "
                               f"(must be 1-31): {numeric[bad_mask].unique()[:5].tolist()}")

        elif subtype == "month_num":
            numeric = pd.to_numeric(series, errors="coerce")
            bad_mask = ((numeric < 1) | (numeric > 12)) & numeric.notna()
            if bad_mask.any():
                issues.append(f"Column '{col}' contains invalid month numbers "
                               f"(must be 1-12): {numeric[bad_mask].unique()[:5].tolist()}")

        elif subtype == "month_name":
            import calendar
            valid_months = ([m.lower() for m in calendar.month_name if m] +
                             [m.lower() for m in calendar.month_abbr if m])
            clean_series = series.astype(str).str.strip().str.lower()
            clean_series = clean_series.replace(
                {"": None, "null": None, "nan": None, "none": None}
            )
            invalid_mask = ~clean_series.isin(valid_months) & clean_series.notna()
            if invalid_mask.any():
                issues.append(f"Column '{col}' contains invalid month names: "
                               f"{clean_series[invalid_mask].unique()[:5].tolist()}")

        elif subtype == "full_date":
            if not pd.api.types.is_datetime64_any_dtype(series):
                coerced = pd.to_datetime(series, errors="coerce")
                bad_mask = coerced.isna() & series.notna()
                if bad_mask.any():
                    issues.append(f"Column '{col}' contains unparseable date values: "
                                   f"{series[bad_mask].unique()[:5].tolist()}")
    return issues


def check_numeric_validity(df_cleaned: pd.DataFrame, numeric_columns: list) -> list:
    """Verifie que les colonnes censees etre numeriques le sont reellement."""
    issues = []
    for col in numeric_columns:
        if col not in df_cleaned.columns:
            continue
        numeric = pd.to_numeric(df_cleaned[col], errors="coerce")
        is_fully_numeric = numeric.notna().eq(df_cleaned[col].notna()).all()
        if not is_fully_numeric:
            bad_mask = numeric.isna() & df_cleaned[col].notna()
            bad_values = df_cleaned[col][bad_mask].unique()[:5].tolist()
            issues.append(f"Column '{col}' should be numeric but contains "
                           f"non-numeric values: {bad_values}")
    return issues


def _count_numeric_issues(df_cleaned: pd.DataFrame, numeric_columns: list) -> tuple:
    """
    Meme verification que check_numeric_validity(), mais au niveau de chaque
    VALEUR individuelle (pas juste "la colonne a au moins un probleme") --
    necessaire pour pouvoir fusionner ce compte avec semantic_correctness dans
    le MEME ratio numerateur/denominateur (meme granularite), comme demande par
    l'encadrante pour l'evaluation du fichier brut seul.

    Returns
    -------
    (total_checked, total_invalid) : nombre total de valeurs non-nulles
    examinees, et combien d'entre elles ne sont PAS numeriques.
    """
    total_checked, total_invalid = 0, 0
    for col in numeric_columns:
        if col not in df_cleaned.columns:
            continue
        non_null = df_cleaned[col].dropna()
        total_checked += len(non_null)
        numeric = pd.to_numeric(non_null, errors="coerce")
        total_invalid += int(numeric.isna().sum())
    return total_checked, total_invalid


def _value_matches_any_pattern(value: str, patterns: list) -> bool:
    """Une valeur est valide si elle correspond a AU MOINS UN des patterns donnes
    (plusieurs formats alternatifs peuvent etre legitimes pour une meme colonne,
    ex: 'H:MM a.m.' OU 'H:MM' pour une heure).

    SENSIBLE A LA CASSE PAR DEFAUT (pas de re.IGNORECASE global) -- un vrai cas
    observe montre pourquoi : sur hotel-booking-demand, une casse incoherente
    EST elle-meme une erreur reelle a detecter (ex: 'RESORT HOTEL' ou 'City
    hotel' sont des corruptions du typo-injector, pas des variantes valides de
    'Resort Hotel'/'City Hotel') -- les accepter silencieusement en etant
    insensible a la casse manquerait exactement le type d'erreur que ce
    controle doit attraper. A l'inverse, quand la casse n'a AUCUNE importance
    pour une colonne donnee (ex: 'state' du dataset hospital, stocke en
    minuscules par convention mais dont la forme correcte est juste "2
    lettres"), c'est au PATTERN lui-meme de l'exprimer explicitement via une
    classe de caracteres couvrant les deux casses (ex: '^[A-Za-z]{2}$'), plutot
    qu'un flag global qui affaiblirait aussi les colonnes ou la casse compte."""
    return any(re.match(pattern, value) for pattern in patterns)


def check_format_patterns(df_cleaned: pd.DataFrame, format_patterns_map: dict) -> list:
    """
    Verifie que chaque valeur d'une colonne respecte AU MOINS UN des formats
    (regex) attendus pour cette colonne -- complementaire aux controles
    numeric/date deja existants : ceux-la verifient un TYPE ("est-ce parsable
    comme un nombre/une date"), celui-ci verifie une FORME precise ("un code
    postal a exactement 5 chiffres", "un code mesure a la forme 'xxx-1'")
    qu'aucun des deux autres controles ne peut exprimer.

    format_patterns_map : dict optionnel {nom_colonne: [pattern_regex, ...]}
        -- fourni par l'appelant (ex: agent_c_dataset_hints.py), PAS auto-
        decouvert : contrairement aux types numeric/date/FD, la FORME exacte
        attendue d'une colonne (ex: "10 chiffres" pour un telephone) n'est pas
        quelque chose qu'on peut deviner de facon fiable a partir des donnees
        seules -- une colonne sans entree dans ce dict n'est simplement pas
        verifiee ici (comportement inchange, comme pour semantic_hints).

    Ne signale QUE les valeurs qui ne correspondent a AUCUN des patterns donnes
    pour leur colonne -- une colonne avec plusieurs formats alternatifs valides
    (ex: sched_dep_time accepte 'H:MM a.m.' ET 'H:MM') n'est en erreur que si
    une valeur ne correspond a AUCUN des deux.
    """
    issues = []
    for col, patterns in format_patterns_map.items():
        if col not in df_cleaned.columns or not patterns:
            continue
        non_null = df_cleaned[col].dropna().astype(str)
        if len(non_null) == 0:
            continue
        bad_mask = ~non_null.apply(lambda v: _value_matches_any_pattern(v, patterns))
        if bad_mask.any():
            bad_values = non_null[bad_mask].unique()[:5].tolist()
            issues.append(f"Column '{col}' contains values that do not match the "
                           f"expected format {patterns}: {bad_values}")
    return issues


def _count_format_pattern_issues(df_cleaned: pd.DataFrame, format_patterns_map: dict) -> tuple:
    """Meme verification que check_format_patterns(), mais au niveau de chaque
    VALEUR individuelle -- meme raison que _count_numeric_issues() : permet de
    fusionner ce compte dans le MEME ratio numerateur/denominateur que
    semantic_correctness pour l'evaluation du fichier brut seul.

    Returns
    -------
    (total_checked, total_invalid) : nombre total de valeurs non-nulles
    examinees (parmi les colonnes ayant des patterns fournis), et combien
    d'entre elles ne correspondent a AUCUN pattern attendu.
    """
    total_checked, total_invalid = 0, 0
    for col, patterns in format_patterns_map.items():
        if col not in df_cleaned.columns or not patterns:
            continue
        non_null = df_cleaned[col].dropna().astype(str)
        total_checked += len(non_null)
        bad_mask = ~non_null.apply(lambda v: _value_matches_any_pattern(v, patterns))
        total_invalid += int(bad_mask.sum())
    return total_checked, total_invalid


def check_functional_dependencies(df_cleaned: pd.DataFrame, fd_pairs: list) -> list:
    """
    fd_pairs : liste de (source, target) ou (source_tuple, target) -- verifie que
    chaque valeur de la cle source correspond a UNE SEULE valeur de la colonne
    cible (comme nos propres FUNCTIONAL_DEPENDENCIES "EXACT" dans prompt_builder.py
    -- ceci VERIFIE APRES COUP que le script genere a effectivement respecte la
    dependance, plutot que de simplement l'esperer).
    """
    issues = []
    for source, target in fd_pairs:
        source_cols = [source] if isinstance(source, str) else list(source)
        if any(c not in df_cleaned.columns for c in source_cols) or target not in df_cleaned.columns:
            continue
        violations = df_cleaned.groupby(source_cols)[target].nunique(dropna=True)
        violation_count = (violations > 1).sum()
        if violation_count > 0:
            source_repr = source if isinstance(source, str) else tuple(source)
            issues.append(
                f"Functional dependency violation: {source_repr} -> '{target}' "
                f"should map to exactly one value per group, but {violation_count} "
                f"group(s) have multiple different '{target}' values. This means "
                f"the whole-column-overwrite technique either wasn't applied here, "
                f"or a different key was used than declared."
            )
    return issues


# ---------------------------------------------------------------------------
# LLM-BASED SEMANTIC CHECK (the genuinely judgment-requiring part)
# ---------------------------------------------------------------------------

SEMANTIC_VALIDATOR_SYSTEM_PROMPT = """You are a strict but FAIR data-quality auditor. You are given a small sample \
of UNIQUE values actually present in one column of a cleaned dataset, after an automated cleaning script has \
already run. Your ONLY job is to flag values that are GENUINELY WRONG for that column — not to enforce a single \
preferred style.

A value is a REAL ERROR only if it is:
- A misspelling or garbled/corrupted text of a real value (e.g. "Pariis" instead of "Paris", "Londn" instead \
of "London", "Cana da" instead of "Canada").
- A value that is not a real instance of the expected category at all (e.g. a phone number appearing in a \
city column, a random string that matches nothing real).
- A clearly impossible or out-of-range value for the stated semantic meaning (e.g. a "day of month" of 45, \
a year of 3051).

A value is NOT an error — do not flag it — if it is:
- A standard abbreviation or alternate valid form of a correct value (e.g. "US" or "USA" for "United States", \
"NY" for "New York", "St." for "Saint", "Corp." for "Corporation", "UK" for "United Kingdom").
- A different but equally valid formatting/capitalization choice (e.g. "PARIS" vs "Paris" vs "paris" — same \
place, just different case; capitalization differences alone are NEVER an error to flag here).
- A legitimate variation, synonym, or regional spelling of something real (e.g. "Center" vs "Centre").
- A value you are simply unfamiliar with but that could plausibly be real (place names, rare-but-real names, \
uncommon-but-valid codes) — when in doubt, do NOT flag it. Only flag values you are CONFIDENT are actually wrong.

Respond with ONLY a JSON object, no other text, no markdown fences, in this exact shape:
{"errors": [{"value": "<the exact wrong value from the list>", "reason": "<short reason it's wrong>", \
"suggested_fix": "<the corrected value, or null if unclear>", "is_missing_marker": <true or false>}]}

Set "is_missing_marker" to true when the wrong value looks like a corrupted/garbled placeholder for "missing" \
or "unknown" (e.g. "NNA", "N/A" variants, "/A", "unnown", "nul", stray punctuation) rather than a typo of one \
specific real value — this tells the downstream system to treat it as a missing value (to be imputed normally) \
instead of hunting for a literal replacement string. In that case, leave "suggested_fix" as null. Only set \
"suggested_fix" to a concrete value when you are identifying a typo/corruption of ONE specific real value \
(e.g. "Pariis" -> "Paris", or matching one of the valid values given to you if a fixed list was provided).

If no genuine errors are found, respond with exactly: {"errors": []}
Never include an entry for a value you are not confident is a real error."""


def _build_user_prompt(column_name: str, semantic_hint: str, values: list,
                        valid_values: list = None) -> str:
    hint_line = f"Expected semantic meaning of this column: {semantic_hint}\n" if semantic_hint else ""
    valid_line = ""
    if valid_values:
        valid_line = (
            f"IMPORTANT: this column only has {len(valid_values)} legitimate values: "
            f"{valid_values!r}. Any suggested_fix you give MUST be one of these exact "
            f"values, or null if the value looks like a disguised MISSING marker "
            f"(e.g. a corrupted 'N/A', 'unknown', empty) rather than a typo of one of "
            f"the valid values — never invent or guess a value that is not in this "
            f"list, and never suggest a single leftover character from the dirty "
            f"string as if it were a real value.\n"
        )
    values_block = "\n".join(f"- {v!r}" for v in values)
    return (
        f"Column name: {column_name}\n"
        f"{hint_line}"
        f"{valid_line}"
        f"Here are {len(values)} unique values actually present in this column after cleaning:\n"
        f"{values_block}\n\n"
        f"List ONLY the values that are genuinely wrong, per the rules you were given."
    )


def _parse_llm_json(text: str) -> dict:
    """Extrait le JSON de la reponse du LLM, meme si elle contient des fences
    markdown (```json ... ```) ou du texte parasite autour -- un LLM plus petit
    respecte moins fidelement 'reponds avec UNIQUEMENT du JSON' que les gros modeles.

    IMPORTANT : text peut etre None -- certains modeles (surtout via un routeur
    qui choisit un modele ALEATOIRE, ex: OpenRouter "openrouter/free") renvoient
    parfois un message.content vide/None plutot qu'une chaine, sans lever
    d'erreur cote SDK. Un crash reel a ete observe ici (AttributeError sur
    None.strip()) -- on traite maintenant ce cas comme une reponse vide (aucune
    erreur detectee), exactement comme une reponse non-parsable, plutot que de
    faire planter tout le run pour une seule colonne.
    """
    if text is None:
        return {"errors": [], "_parse_failed": True, "_raw_response": None}
    text = text.strip()
    # Retire les fences markdown eventuelles
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Filet de securite : cherche le premier bloc { ... } dans le texte
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"errors": [], "_parse_failed": True, "_raw_response": text[:500]}


def check_column_semantics(df: pd.DataFrame, column: str, semantic_hint: str = "",
                            valid_values: list = None, sample_size: int = 25,
                            provider: str = None, model: str = None) -> dict:
    """
    Interroge le LLM validateur sur UNE colonne. Retourne :
    {
        "column": str,
        "checked_values": int,
        "errors": [{"value": ..., "reason": ..., "suggested_fix": ...}, ...],
        "has_errors": bool,
    }

    valid_values : liste optionnelle des seules valeurs legitimes pour cette colonne
        (utile pour une colonne a faible cardinalite avec un ensemble de valeurs
        fixe et connu, ex: ['S','C','Q'] pour un code de port). Quand fournie, le
        LLM est explicitement contraint a ne suggerer un suggested_fix QUE parmi ces
        valeurs (ou null) -- evite un vrai probleme observe en pratique : sans
        cette contrainte, le LLM a suggere 'A' comme correction de '/A' pour un
        code de port a 3 valeurs (S/C/Q) -- 'A' n'etait meme pas une valeur valide,
        juste "le caractere le plus proche" dans la chaine corrompue. Un garde-fou
        en aval (pas seulement l'instruction du prompt) neutralise aussi tout
        suggested_fix qui ne serait pas dans cette liste.
    """
    if column not in df.columns:
        return {"column": column, "checked_values": 0, "errors": [], "has_errors": False,
                "skipped_reason": "column not found"}

    unique_vals = df[column].dropna().astype(str).unique().tolist()
    if not unique_vals:
        return {"column": column, "checked_values": 0, "errors": [], "has_errors": False,
                "skipped_reason": "no non-null values"}

    # Echantillonnage deterministe (pas de hasard, pour rester reproductible) : les
    # valeurs les moins frequentes sont plus susceptibles d'etre des erreurs qu'un
    # cas courant repete -- on les priorise dans l'echantillon envoye au LLM.
    value_counts = df[column].astype(str).value_counts()
    sorted_vals = sorted(unique_vals, key=lambda v: value_counts.get(v, 0))
    sample = sorted_vals[:sample_size]

    user_prompt = _build_user_prompt(column, semantic_hint, sample, valid_values=valid_values)
    result = call_llm(user_prompt, provider=provider, model=model,
                       system_prompt=SEMANTIC_VALIDATOR_SYSTEM_PROMPT)
    parsed = _parse_llm_json(result["text"])
    errors = parsed.get("errors", [])

    # Garde-fou 1 : ne garde que les erreurs dont la valeur signalee etait REELLEMENT
    # dans l'echantillon envoye -- une "erreur" sur une valeur halluciner par le LLM
    # (qui n'existe pas dans les donnees) doit etre ignoree silencieusement plutot que
    # de generer un feedback trompeur pour l'iteration suivante.
    sample_set = set(sample)
    errors = [e for e in errors if isinstance(e, dict) and e.get("value") in sample_set]

    # Garde-fou 2 : si une liste de valeurs valides est fournie, neutralise (met a
    # null) tout suggested_fix qui n'en fait pas partie -- l'erreur elle-meme reste
    # signalee (la valeur EST bien fausse), mais on n'expose pas une correction
    # inventee/implausible en aval.
    if valid_values:
        valid_set = set(valid_values)
        for e in errors:
            fix = e.get("suggested_fix")
            if fix is not None and fix not in valid_set:
                e["suggested_fix"] = None
                e["_fix_rejected_not_in_valid_values"] = True

    return {
        "column": column,
        "checked_values": len(sample),
        "checked_rows": int(sum(value_counts.get(v, 0) for v in sample)),
        "error_rows": int(sum(value_counts.get(e["value"], 0) for e in errors)),
        "errors": errors,
        "has_errors": len(errors) > 0,
    }


def validate_semantics(df_cleaned: pd.DataFrame, columns: list = None,
                        semantic_hints: dict = None, valid_values_map: dict = None,
                        sample_size: int = 25, provider: str = None,
                        model: str = None) -> dict:
    """
    Point d'entree principal de l'Agent C (Validateur semantique).

    Parameters
    ----------
    df_cleaned : le DataFrame nettoye a inspecter (typiquement la sortie de la
        meilleure iteration en cours du validation_loop, ou un echantillon de celle-ci
        pour limiter le cout).
    columns : liste explicite de colonnes a verifier. Si None, verifie automatiquement
        toutes les colonnes categorielles (dtype object) dont la cardinalite suggere
        une categorie semantique verifiable (ni un identifiant unique par ligne, ni
        une colonne quasi-constante) -- typiquement entre 2 et 200 valeurs distinctes.
    semantic_hints : dict optionnel {nom_colonne: description} pour guider le LLM
        (ex: {"city": "a US city name", "state": "a US state, name or abbreviation"}).
        Ameliore nettement la precision du jugement quand fourni.
    valid_values_map : dict optionnel {nom_colonne: [valeurs legitimes]} pour les
        colonnes a ensemble de valeurs FIXE et connu (ex: {"Embarked": ["S","C","Q"]}).
        Contraint le suggested_fix du LLM a rester dans cette liste (ou null) -- voir
        check_column_semantics pour le bug concret que cela evite.
    sample_size : nombre de valeurs uniques envoyees au LLM par colonne (limite le
        cout/tokens ; les valeurs les moins frequentes sont priorisees, voir
        check_column_semantics).

    Returns
    -------
    dict:
        {
            "columns_checked": [...],
            "results": {col: check_column_semantics(...) for col in columns},
            "has_errors": bool,          # vrai si AU MOINS UNE colonne a de vraies erreurs
            "feedback_text": str | None, # texte pret a inserer dans le prompt de
                                          # l'iteration suivante ; None si aucune erreur
        }
    """
    semantic_hints = semantic_hints or {}
    valid_values_map = valid_values_map or {}

    if columns is None:
        columns = []
        n_rows = len(df_cleaned)
        for col in df_cleaned.columns:
            # is_object_dtype OR is_string_dtype (pas juste dtype == "object") :
            # pandas peut inferer une colonne texte comme le nouveau dtype "string"
            # (string[python]/string[pyarrow]) plutot que "object" selon la version
            # et la config -- une simple comparaison dtype != "object" les rate
            # silencieusement (meme categorie de piege que le bug .median() sur
            # dtype string rencontre ailleurs dans ce projet).
            if not (pd.api.types.is_object_dtype(df_cleaned[col])
                    or pd.api.types.is_string_dtype(df_cleaned[col])):
                continue
            n_unique = df_cleaned[col].nunique(dropna=True)
            if n_unique < 2:
                continue  # constant column, nothing to check
            # Toujours inclure si <=50 valeurs distinctes (clairement une categorie,
            # quelle que soit la taille du dataset -- l'ancienne regle basee sur
            # n_rows//2 excluait a tort des colonnes categorielles valides sur de
            # petits dataframes, ex: un df de test a 5 lignes). Au-dela de 50,
            # n'inclure que si la majorite des valeurs se repetent (ratio <= 50%),
            # pour ecarter les colonnes quasi-uniques type identifiant.
            unique_ratio = n_unique / n_rows if n_rows > 0 else 1.0
            if n_unique <= 50 or (n_unique <= 200 and unique_ratio <= 0.5):
                columns.append(col)

    results = {}
    for col in columns:
        results[col] = check_column_semantics(
            df_cleaned, col, semantic_hint=semantic_hints.get(col, ""),
            valid_values=valid_values_map.get(col), sample_size=sample_size,
            provider=provider, model=model,
        )

    any_errors = any(r["has_errors"] for r in results.values())

    feedback_text = None
    if any_errors:
        lines = ["SEMANTIC REVIEW FEEDBACK (from a second LLM inspecting your cleaned "
                 "output for genuinely wrong values, not just formatting differences):"]
        for col, r in results.items():
            if not r["has_errors"]:
                continue
            lines.append(f"\nColumn '{col}':")
            for e in r["errors"]:
                fix = e.get("suggested_fix")
                if fix:
                    fix_str = f" -> likely correct value: {fix!r}"
                elif e.get("is_missing_marker"):
                    fix_str = (
                        " -> this looks like a corrupted/disguised MISSING value "
                        "marker, not a typo of one specific value — set this EXACT "
                        "value to NaN directly (e.g. `df[col] = df[col].replace("
                        f"{e.get('value')!r}, np.nan)`), then let your normal "
                        "missing-value imputation handle it. Do NOT try to 'salvage' "
                        "a partial value by stripping out the bad character(s) with "
                        "a regex or .replace('x','') — a real, severe regression was "
                        "caused by this exact mistake before: stripping a non-numeric "
                        "character out of a corrupted key value (e.g. turning "
                        "'x0005' into '0005') silently changes it into a DIFFERENT, "
                        "WRONG value instead of NaN, which is especially damaging if "
                        "this column is used as a grouping key elsewhere in your "
                        "script — it breaks that grouping for this row across every "
                        "column that depends on it. Only two outcomes are acceptable "
                        "here: NaN (preferred), or the literal correct value if you "
                        "are genuinely confident what it originally was — never a "
                        "mechanically-stripped fragment of the corrupted string."
                    )
                else:
                    fix_str = ""
                lines.append(f"  - {e.get('value')!r} is wrong: {e.get('reason', '')}{fix_str}")
        lines.append(
            "\nThese are confirmed real errors (not stylistic variations, not "
            "abbreviations) — fix these specific values in your next version. Do "
            "NOT change any value not listed above; other values in these columns "
            "were reviewed and found acceptable, including abbreviations and "
            "alternate valid formats."
        )
        feedback_text = "\n".join(lines)

    return {
        "columns_checked": columns,
        "results": results,
        "has_errors": any_errors,
        "feedback_text": feedback_text,
    }


# ---------------------------------------------------------------------------
# MASTER ORCHESTRATOR — Agent C's single entry point, combining every role
# from the reference validator.py: rule-based checks (fast, free) run first,
# the LLM semantic check runs last (only place a real LLM call happens here).
# ---------------------------------------------------------------------------

def validate(df_original: pd.DataFrame, df_cleaned: pd.DataFrame,
             expected_columns: list = None, date_columns: dict = None,
             numeric_columns: list = None, fd_pairs: list = None,
             semantic_columns: list = None, semantic_hints: dict = None,
             valid_values_map: dict = None, format_patterns_map: dict = None,
             data_loss_threshold: float = 0.30, sample_size: int = 25,
             provider: str = None, model: str = None,
             skip_semantic_check: bool = False,
             auto_discover: bool = True,
             include_composite_fds: bool = True,
             skip_before_after_checks: bool = False) -> dict:
    """
    Point d'entree unique de l'Agent C. Lance TOUS les controles (rapides d'abord,
    LLM en dernier) et retourne un resultat unifie.

    AUTO-DECOUVERTE PAR DEFAUT (auto_discover=True) -- AUCUNE CONFIGURATION PAR
    DATASET N'EST NECESSAIRE : si expected_columns/numeric_columns/date_columns/
    fd_pairs ne sont pas fournis explicitement, ils sont INFERES automatiquement
    depuis df_original via auto_infer_column_types() et auto_discover_fd_pairs().
    Donner un nouveau dataset a cette fonction fonctionne sans ecrire une seule
    ligne de configuration specifique a ce dataset. Passer un parametre
    explicitement le remplace toujours (utile pour forcer/completer une
    decouverte automatique imparfaite), mais ce n'est plus jamais REQUIS.

    format_patterns_map ({col: [regex, ...]}) est la SEULE exception a l'auto-
    decouverte : la forme exacte attendue d'une colonne (ex: "un code postal a
    5 chiffres") n'est pas quelque chose qu'on peut deviner de facon fiable a
    partir des donnees seules -- si non fourni, ce controle est simplement
    desactive (comportement inchange), voir check_format_patterns().

    IMPORTANT — LA VERIFICATION LLM S'EXECUTE PAR DEFAUT SUR TOUTES LES COLONNES
    CATEGORIELLES PERTINENTES, PAS SEULEMENT SUR CE QUI EST EXPLICITEMENT LISTE.
    L'exemple "Paris/Pariis" donne par l'encadrante n'etait qu'UNE illustration du
    type d'erreur a detecter, pas une limitation de la portee du controle a la
    seule categorie "lieu" -- le controle semantique doit tout verifier.

    Returns
    -------
    dict:
        {
            "valid": bool,               # True seulement si AUCUN probleme (regles
                                          # ET semantique) n'a ete trouve
            "rule_issues": [...],        # issues des controles rapides (str)
            "semantic_result": {...},    # sortie complete de validate_semantics()
            "feedback_text": str | None, # texte unifie ; None si tout est OK
            "auto_discovered": {...},    # ce qui a ete infere automatiquement,
                                          # pour transparence/debug
        }
    """
    auto_discovered = {}

    if auto_discover:
        if expected_columns is None:
            expected_columns = list(df_original.columns)
            auto_discovered["expected_columns"] = "from df_original.columns"

        inferred_types = auto_infer_column_types(df_cleaned)

        if numeric_columns is None:
            numeric_columns = inferred_types["numeric_columns"]
            auto_discovered["numeric_columns"] = numeric_columns

        if date_columns is None:
            date_columns = inferred_types["date_like_columns"]
            auto_discovered["date_columns"] = date_columns

        if fd_pairs is None:
            single_fds = auto_discover_fd_pairs(df_cleaned)
            fd_pairs = list(single_fds)
            if include_composite_fds:
                composite_fds = auto_discover_composite_fd_pairs(df_cleaned, single_column_fd_pairs=single_fds)
                fd_pairs = fd_pairs + composite_fds
            auto_discovered["fd_pairs"] = fd_pairs

    rule_issues = []
    # IMPORTANT (remarque de l'encadrante) : data_loss et column_preservation
    # comparent df_original a df_cleaned -- si les DEUX sont EN FAIT le meme
    # fichier (cas de l'evaluation "DIRTY seul", ou on evalue le fichier brut
    # contre lui-meme faute d'un "avant" reel), ces deux controles sont VIDES DE
    # SENS et retournent TOUJOURS 100% par construction mathematique (0 difference
    # entre un fichier et lui-meme), pas parce que les donnees sont bonnes. Ce
    # 100% artificiel gonflerait a tort le Pourcentage 1 du fichier brut.
    # skip_before_after_checks=True les exclut dans ce cas precis.
    if not skip_before_after_checks:
        rule_issues += check_data_loss(df_original, df_cleaned, threshold=data_loss_threshold)
        rule_issues += check_column_preservation(df_original, df_cleaned, expected_columns)
    rule_issues += check_empty_columns(df_cleaned, expected_columns)
    if date_columns:
        rule_issues += check_date_validity(df_cleaned, date_columns)
    if numeric_columns:
        rule_issues += check_numeric_validity(df_cleaned, numeric_columns)
    if fd_pairs:
        rule_issues += check_functional_dependencies(df_cleaned, fd_pairs)
    if format_patterns_map:
        rule_issues += check_format_patterns(df_cleaned, format_patterns_map)

    semantic_result = None
    if not skip_semantic_check:
        semantic_result = validate_semantics(
            df_cleaned, columns=semantic_columns, semantic_hints=semantic_hints,
            valid_values_map=valid_values_map, sample_size=sample_size,
            provider=provider, model=model,
        )

    has_semantic_errors = bool(semantic_result and semantic_result["has_errors"])
    is_valid = (len(rule_issues) == 0) and not has_semantic_errors

    feedback_parts = []
    if rule_issues:
        feedback_parts.append(
            "DATA QUALITY REVIEW — rule-based checks found issues:\n" +
            "\n".join(f"- {issue}" for issue in rule_issues)
        )
    if has_semantic_errors:
        feedback_parts.append(semantic_result["feedback_text"])

    feedback_text = "\n\n".join(feedback_parts) if feedback_parts else None

    return {
        "valid": is_valid,
        "rule_issues": rule_issues,
        "semantic_result": semantic_result,
        "feedback_text": feedback_text,
        "auto_discovered": auto_discovered,
        "resolved_params": {
            "expected_columns": expected_columns,
            "numeric_columns": numeric_columns,
            "date_columns": date_columns,
            "fd_pairs": fd_pairs,
            "format_patterns_map": format_patterns_map,
        },
    }


# ---------------------------------------------------------------------------
# DOUBLE POURCENTAGE DE PROPRETE — demande de l'encadrante :
#   1) pourcentage de respect des regles definies (data loss, colonnes,
#      dependances fonctionnelles, controle semantique LLM)
#   2) pourcentage base sur la comparaison DIRECTE avec les donnees propres de
#      reference (clean.csv) -- reutilise compute_global_accuracy() de
#      metrics.py plutot que de redupliquer cette logique.
# Les deux pourcentages sont volontairement gardes SEPARES (jamais moyennes
# ensemble) : ils mesurent deux choses differentes -- le respect de regles
# GENERALES applicables a n'importe quel dataset (pourcentage 1) versus l'exactitude
# reelle mesuree contre UNE reference precise (pourcentage 2, seulement disponible
# quand on connait la verite terrain, ce qui n'est pas toujours le cas en
# production reelle).
# ---------------------------------------------------------------------------

def compute_rule_based_cleanliness(df_original: pd.DataFrame, df_cleaned: pd.DataFrame,
                                    expected_columns: list = None, date_columns: dict = None,
                                    numeric_columns: list = None, fd_pairs: list = None,
                                    semantic_columns: list = None, semantic_hints: dict = None,
                                    valid_values_map: dict = None, format_patterns_map: dict = None,
                                    data_loss_threshold: float = 0.30, sample_size: int = 25,
                                    provider: str = None, model: str = None,
                                    skip_semantic_check: bool = False,
                                    skip_before_after_checks: bool = False) -> dict:
    """
    Pourcentage 1 : a quel point les donnees nettoyees respectent les regles
    definies (structurelles + semantiques), independamment de toute reference
    "verite terrain". Chaque categorie de regle produit un sous-score 0-100 ;
    le score global est la moyenne des sous-scores APPLICABLES (une categorie
    sans parametre fourni, ex: pas de date_columns, n'est simplement pas comptee).

    Returns
    -------
    dict:
        {
            "overall_percentage": float,      # 0-100, moyenne des sous-scores
            "breakdown": {nom_categorie: float, ...},  # 0-100 chacun
            "details": dict,                  # sortie complete de validate()
        }
    """
    result = validate(
        df_original, df_cleaned,
        expected_columns=expected_columns, date_columns=date_columns,
        numeric_columns=numeric_columns, fd_pairs=fd_pairs,
        semantic_columns=semantic_columns, semantic_hints=semantic_hints,
        valid_values_map=valid_values_map, format_patterns_map=format_patterns_map,
        data_loss_threshold=data_loss_threshold,
        sample_size=sample_size, provider=provider, model=model,
        skip_semantic_check=skip_semantic_check,
        skip_before_after_checks=skip_before_after_checks,
    )

    # IMPORTANT : reutilise les valeurs RESOLUES par validate() (auto-decouvertes
    # si non fournies ici) pour le calcul du breakdown ci-dessous -- sinon, si
    # l'appelant n'a rien fourni explicitement, ce breakdown utiliserait les
    # variables locales encore a None et sauterait silencieusement toutes les
    # sous-categories au lieu de refleter ce qui a reellement ete verifie.
    resolved = result["resolved_params"]
    expected_columns = resolved["expected_columns"]
    numeric_columns = resolved["numeric_columns"]
    date_columns = resolved["date_columns"]
    fd_pairs = resolved["fd_pairs"]
    format_patterns_map = resolved["format_patterns_map"]

    breakdown = {}

    # --- Data loss : 100% si sous le seuil, degrade lineairement au-dela ---
    # (exclu si skip_before_after_checks=True -- voir validate() pour l'explication :
    # comparer un fichier a lui-meme rend ce controle vide de sens.)
    if not skip_before_after_checks:
        initial = len(df_original)
        if initial > 0:
            loss_pct = max(0.0, (initial - len(df_cleaned)) / initial)
            breakdown["data_loss"] = round(max(0.0, 100.0 * (1 - loss_pct / max(data_loss_threshold, 1e-9))), 2) \
                if loss_pct > 0 else 100.0

        # --- Preservation des colonnes ---
        if expected_columns:
            present = sum(1 for c in expected_columns if c in df_cleaned.columns)
            breakdown["column_preservation"] = round(100.0 * present / len(expected_columns), 2)

        # --- Colonnes vides (parmi les colonnes attendues verifiees) ---
        # EXCLU pour l'evaluation "DIRTY seul" (skip_before_after_checks=True), sur
        # demande de l'encadrante : cette categorie est presque toujours ~100%
        # (une colonne entierement vide est rare), donc elle dilue artificiellement
        # le score global vers le haut sans refleter la vraie proportion d'erreurs
        # -- meme logique que pour data_loss/column_preservation ci-dessus.
        if expected_columns:
            empty_issues = check_empty_columns(df_cleaned, expected_columns)
            breakdown["non_empty_columns"] = round(
                100.0 * (1 - len(empty_issues) / len(expected_columns)), 2
            )

    # --- Dates : % de valeurs dans une plage plausible ---
    if date_columns:
        date_issue_count = len(check_date_validity(df_cleaned, date_columns))
        breakdown["date_validity"] = round(
            100.0 * (1 - date_issue_count / max(len(date_columns), 1)), 2
        )

    # --- Numerique : % de colonnes reellement numeriques ---
    # Pour l'evaluation CLEANED normale, reste une categorie separee (colonne
    # ayant au moins un probleme / total colonnes), comme avant. Pour
    # l'evaluation DIRTY (skip_before_after_checks=True), fusionnee plus bas
    # dans le meme ratio pool que semantic_correctness (au niveau VALEUR, pas
    # colonne), pour la meme raison que functional_dependencies : une categorie
    # separee ici ne refleterait pas la vraie proportion d'erreurs au niveau
    # cellule, et diluerait le score global de facon incoherente avec le
    # Pourcentage 2 (comparaison directe, qui EST au niveau cellule).
    if numeric_columns and not skip_before_after_checks:
        numeric_issue_count = len(check_numeric_validity(df_cleaned, numeric_columns))
        breakdown["numeric_validity"] = round(
            100.0 * (1 - numeric_issue_count / max(len(numeric_columns), 1)), 2
        )

    # --- Format/pattern (regex) : meme traitement que numeric_validity juste
    # au-dessus -- categorie separee (au niveau colonne) pour l'evaluation
    # CLEANED normale, fusionnee au niveau VALEUR dans le pool semantique plus
    # bas pour l'evaluation DIRTY seul (meme raison : eviter qu'une categorie
    # quasi-100% dilue le score de facon incoherente avec le Pourcentage 2).
    if format_patterns_map and not skip_before_after_checks:
        format_issue_count = len(check_format_patterns(df_cleaned, format_patterns_map))
        checked_cols = [c for c in format_patterns_map if c in df_cleaned.columns]
        if checked_cols:
            breakdown["format_compliance"] = round(
                100.0 * (1 - format_issue_count / len(checked_cols)), 2
            )

    # --- Semantique (LLM) + dependances fonctionnelles ---
    # IMPORTANT (consigne de l'encadrante) : pour l'evaluation "DIRTY seul"
    # (skip_before_after_checks=True), les violations de dependances
    # fonctionnelles sont FUSIONNEES dans le MEME ratio que les erreurs
    # semantiques (meme numerateur/denominateur "erreurs trouvees / elements
    # verifies"), PLUTOT que comptees comme une categorie separee. Raison :
    # functional_dependencies, calculee separement et moyennee a poids egal
    # avec les autres categories, atteint quasi-systematiquement ~99% (tres peu
    # de groupes violent une FD en pratique) -- ce quasi-100% tirait le score
    # global vers le haut de facon disproportionnee, faisant que le Pourcentage
    # 1 (regles) ne ressemblait plus au Pourcentage 2 (comparaison directe avec
    # la reference), qui lui capture la vraie proportion d'erreurs. En fusionnant
    # les deux dans un seul ratio pool, le Pourcentage 1 redevient un signal
    # comparable au Pourcentage 2.
    #
    # Pour l'evaluation normale (CLEANED, skip_before_after_checks=False),
    # functional_dependencies reste une categorie SEPAREE comme avant -- ce
    # comportement n'est pas modifie, seule l'evaluation du fichier brut seul
    # est concernee par ce changement.
    semantic_result = result.get("semantic_result")
    total_checked = 0
    total_errors = 0
    if semantic_result and semantic_result.get("results"):
        # IMPORTANT (bug corrige) : utilise checked_rows/error_rows (pondere par
        # le nombre REEL de lignes que chaque valeur representait), PAS
        # checked_values/len(errors) (comptage de VALEURS DISTINCTES). Sans ce
        # pondere, une colonne ou "male"/"female" apparaissent des centaines de
        # fois (corrects) mais chaque faute de frappe n'apparait qu'une ou deux
        # fois, faisait paraitre le taux d'erreur artificiellement BAS en
        # comptage de valeurs uniques -- incoherent avec le Pourcentage 2, qui
        # lui compte au niveau CELLULE/LIGNE. Ce pondere aligne les deux
        # granularites, rendant le Pourcentage 1 comparable au Pourcentage 2.
        total_checked += sum(r.get("checked_rows", r.get("checked_values", 0))
                              for r in semantic_result["results"].values())
        total_errors += sum(r.get("error_rows", len(r.get("errors", [])))
                             for r in semantic_result["results"].values())

    if numeric_columns and skip_before_after_checks:
        numeric_checked, numeric_invalid = _count_numeric_issues(df_cleaned, numeric_columns)
        total_checked += numeric_checked
        total_errors += numeric_invalid

    if format_patterns_map and skip_before_after_checks:
        format_checked, format_invalid = _count_format_pattern_issues(df_cleaned, format_patterns_map)
        total_checked += format_checked
        total_errors += format_invalid

    if fd_pairs and skip_before_after_checks:
        fd_total_groups, fd_violated_groups = 0, 0
        for source, target in fd_pairs:
            source_cols = [source] if isinstance(source, str) else list(source)
            if any(c not in df_cleaned.columns for c in source_cols) or target not in df_cleaned.columns:
                continue
            nunique_per_group = df_cleaned.groupby(source_cols)[target].nunique(dropna=True)
            fd_total_groups += len(nunique_per_group)
            fd_violated_groups += int((nunique_per_group > 1).sum())
        total_checked += fd_total_groups
        total_errors += fd_violated_groups
    elif fd_pairs:
        # Comportement inchange pour l'evaluation CLEANED normale : FD reste une
        # categorie separee dans le breakdown.
        total_groups, violated_groups = 0, 0
        for source, target in fd_pairs:
            source_cols = [source] if isinstance(source, str) else list(source)
            if any(c not in df_cleaned.columns for c in source_cols) or target not in df_cleaned.columns:
                continue
            nunique_per_group = df_cleaned.groupby(source_cols)[target].nunique(dropna=True)
            total_groups += len(nunique_per_group)
            violated_groups += int((nunique_per_group > 1).sum())
        if total_groups > 0:
            breakdown["functional_dependencies"] = round(
                100.0 * (1 - violated_groups / total_groups), 2
            )

    if total_checked > 0:
        breakdown["semantic_correctness"] = round(
            100.0 * (1 - total_errors / total_checked), 2
        )

    overall = round(sum(breakdown.values()) / len(breakdown), 2) if breakdown else 100.0

    return {
        "overall_percentage": overall,
        "breakdown": breakdown,
        "details": result,
    }


def _values_equal_generic(a, b) -> bool:
    """Compare deux valeurs de facon tolerante (numerique avec tolerance,
    texte insensible a la casse/espaces) -- copie volontairement autonome
    (pas d'import de metrics.py) pour qu'agent_c.py reste 100% independant."""
    a_is_na = a is None or pd.isna(a)
    b_is_na = b is None or pd.isna(b)
    if a_is_na and b_is_na:
        return True
    if a_is_na or b_is_na:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        pass
    return str(a).strip().lower() == str(b).strip().lower()


def compute_reference_comparison_cleanliness(df_reference_clean: pd.DataFrame,
                                              df_cleaned: pd.DataFrame) -> dict | None:
    """
    Pourcentage 2 : comparaison DIRECTE, cellule par cellule, entre les donnees
    nettoyees et les donnees propres de reference (clean.csv) -- necessite de
    connaitre la verite terrain (uniquement possible en evaluation/benchmark,
    pas en usage reel sur des donnees totalement inconnues).

    AUCUNE CONFIGURATION DE NOMS DE COLONNES N'EST NECESSAIRE : les colonnes des
    deux DataFrames sont automatiquement mises en correspondance par nom
    NORMALISE (insensible a la casse/underscores, via _normalize_col_name) --
    fonctionne meme si un cote utilise 'ProviderNumber' et l'autre
    'provider_number', sans avoir a le preciser.

    Fonction 100% autonome (ne depend pas de metrics.py) pour qu'Agent C reste
    completement independant du pipeline de generation.

    Returns
    -------
    dict ou None si aucune colonne/ligne en commun :
        {
            "overall_percentage": float,           # 0-100
            "per_column_percentage": {col: float, ...},  # 0-100 par colonne,
                                                    # nom de colonne cote df_cleaned
        }
    """
    common_idx = df_reference_clean.index.intersection(df_cleaned.index)
    if len(common_idx) == 0:
        return None

    ref_cols_by_norm = {_normalize_col_name(c): c for c in df_reference_clean.columns}

    per_column_percentage = {}
    total_cells, total_matches = 0, 0

    for cleaned_col in df_cleaned.columns:
        ref_col = ref_cols_by_norm.get(_normalize_col_name(cleaned_col))
        if ref_col is None:
            continue
        ref_series = df_reference_clean.loc[common_idx, ref_col]
        cleaned_series = df_cleaned.loc[common_idx, cleaned_col]
        matches = sum(_values_equal_generic(a, b) for a, b in zip(cleaned_series, ref_series))
        n = len(common_idx)
        per_column_percentage[cleaned_col] = round(100.0 * matches / n, 2) if n > 0 else None
        total_cells += n
        total_matches += matches

    if total_cells == 0:
        return None

    return {
        "overall_percentage": round(100.0 * total_matches / total_cells, 2),
        "per_column_percentage": per_column_percentage,
    }


def compute_cleanliness_percentages(df_original: pd.DataFrame, df_cleaned: pd.DataFrame,
                                     df_reference_clean: pd.DataFrame = None,
                                     **rule_kwargs) -> dict:
    """
    Point d'entree unique demande par l'encadrante : calcule les DEUX
    pourcentages de propreté, gardes separes (jamais moyennes ensemble, car ils
    mesurent des choses differentes).

    Parameters
    ----------
    df_original : les donnees AVANT nettoyage (pour le calcul du pourcentage 1 :
        perte de lignes, etc.)
    df_cleaned : les donnees APRES nettoyage, a evaluer.
    df_reference_clean : optionnel. Les donnees propres de reference (clean.csv).
        Si fourni, le pourcentage 2 est calcule (correspondance des colonnes
        AUTOMATIQUE par nom normalise, aucune config requise). Si None, seul le
        pourcentage 1 (base sur les regles, ne necessite pas de verite terrain)
        est retourne -- c'est le cas d'usage REEL en production, ou clean.csv
        n'existe pas.
    **rule_kwargs : tous les parametres de compute_rule_based_cleanliness()
        (expected_columns, date_columns, numeric_columns, fd_pairs,
        semantic_columns, semantic_hints, valid_values_map,
        format_patterns_map, provider, model, auto_discover,
        include_composite_fds...).

    Returns
    -------
    dict:
        {
            "rule_based_percentage": float,       # 0-100, pourcentage 1
            "rule_based_breakdown": {...},
            "reference_comparison_percentage": float | None,   # 0-100, pourcentage 2 global
            "reference_comparison_per_column": {...} | None,   # 0-100 PAR COLONNE (style describe())
        }
    """
    rule_result = compute_rule_based_cleanliness(df_original, df_cleaned, **rule_kwargs)

    reference_percentage = None
    reference_per_column = None
    if df_reference_clean is not None:
        ref_result = compute_reference_comparison_cleanliness(df_reference_clean, df_cleaned)
        if ref_result is not None:
            reference_percentage = ref_result["overall_percentage"]
            reference_per_column = ref_result["per_column_percentage"]

    return {
        "rule_based_percentage": rule_result["overall_percentage"],
        "rule_based_breakdown": rule_result["breakdown"],
        "reference_comparison_percentage": reference_percentage,
        "reference_comparison_per_column": reference_per_column,
    }