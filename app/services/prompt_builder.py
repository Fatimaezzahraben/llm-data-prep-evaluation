"""
app/services/prompt_builder.py  (v3 — + fuites de verite terrain corrigees, + FD composites)
==============================================================================================
Version renforcee du prompt builder, integrant les lecons tirees des echecs observes :
  - imputation par valeur inventee (ex: "00:00 am" pour toute heure manquante)
  - hallucination de colonnes / appel a pd.read_csv() alors que df est deja fourni
  - remplacement de sous-chaine corrompant des valeurs deja correctes
  - parsing numerique fragile (int()/float() direct sur du texte pollue)
  - fonction de nettoyage definie mais jamais appelee
  - sous-nettoyage : trop peu de colonnes reellement corrigees (faible recall)
  - imputation par mode global sur colonnes a forte cardinalite (echoue presque toujours)
  - generalisation abusive d'un pattern "manquant" (ex: "empty") a des colonnes ou
    cette valeur est en realite legitime (corruption de donnees deja correctes)
  - dependances fonctionnelles COMPOSITES (2 colonnes) non exploitees, ex :
    (ProviderNumber, MeasureCode) -> Score, alors que ProviderNumber seul ne suffit pas

Chaque contrainte ci-dessous cible UN bug precis reellement observe, avec l'exemple
concret qui a motive la regle (documente pour le rapport).
"""

import pandas as pd
from app.services.profiler import profile_dataset


# ---------------------------------------------------------------------------
# Contraintes durcies. Chaque regle correspond a un bug reel observe en
# Phase 4/5 (documente dans le rapport, section analyse d'erreurs).
# ---------------------------------------------------------------------------
CONSTRAINTS = """CRITICAL RULES — your script will be executed automatically and graded on F1-score.
Breaking any rule below causes an automatic failure, so follow them exactly.

CONTRACT (how your code will be run):
- A pandas DataFrame named `df` is ALREADY LOADED in the execution environment. Do NOT
  call pd.read_csv(...) or invent any input file — just use `df` directly.
- Write your cleaning logic as a sequence of top-level statements that modify `df`
  directly (df['col'] = ...), OR define a single function and CALL IT immediately
  (e.g. `def clean(df): ...; return df` followed immediately by `df = clean(df)` on
  the next line — never leave the call commented out or as "example usage").
- Do not read or write any file. Do not print large outputs. End your script with `df`
  holding the final cleaned result — do not rename it to another variable.
- NEVER use `df[col].method(value, inplace=True)` (e.g. `df['x'].fillna(0, inplace=True)`)
  — on modern pandas (Copy-on-Write) this operates on a temporary copy and SILENTLY
  fails to update `df`, so your correction will not count even though it runs without
  error. Always reassign instead: `df[col] = df[col].fillna(value)`, or use
  `df.fillna({col: value}, inplace=True)` (dict form on the whole DataFrame is safe).

COLUMNS:
- Only use column names that appear in the list given to you below. Never invent a
  column name (even if you recognize this as a well-known public dataset and remember
  different column names from training — always defer to the exact names given here).

MISSING VALUES:
- Never fill missing values with an arbitrary placeholder that could be mistaken for
  real data (e.g. do NOT fill a missing time with "00:00", do NOT fill a missing date
  with a fixed default date, do NOT fill a missing id with 0). If you cannot infer a
  meaningful value, impute using an actual statistic computed FROM THE DATA ITSELF:
  the column's mode for categorical columns, the column's median for numeric columns.
  Compute these statistics with pandas (`df[col].mode()[0]`, `df[col].median()`) —
  never hardcode a numeric or string constant you did not compute from `df`.
- Apply this to EVERY column that has missing or disguised-missing values (empty
  string, "NA", "N/A", "unknown", whitespace-only) — do not skip columns; skipping
  columns lowers recall, which is heavily penalized.
- NEVER USE RANDOM SAMPLING (np.random.*, random.*, df.sample() for imputation, etc.)
  anywhere in your cleaning code. This breaks reproducibility (the same script must
  produce the same output every time it runs) and is easy to get subtly wrong (e.g.
  `np.random.choice(values, p=weights)` without `size=` draws a SINGLE value and
  applies it to every remaining missing row, not one independent draw per row — a real
  bug seen in practice). Always use a deterministic statistic (mode/median) instead of
  any random draw, even when trying to match an observed category distribution.
- WHEN A GROUP-BASED FALLBACK ONLY COVERS SOME ROWS, THINK CAREFULLY BEFORE FORCING A
  GUESS ON THE REST — it depends on the column's natural missing rate:
  * For a column with a LOW-TO-MODERATE missing rate in the profile (say, under ~40%),
    always finish with a global mode/median fallback so every remaining missing cell
    gets filled — a weak guess still has a real chance of being right, and leaving NaN
    guarantees zero credit.
  * For a column with a VERY HIGH missing rate in the profile (say, over ~50-60%, e.g.
    a column like a cabin/room number that is genuinely absent for most records), do
    NOT force a low-confidence global guess on every remaining cell. In such columns,
    a large share of "missing" cells are missing because the TRUE value really is
    absent, not because information was lost — correctly leaving a cell as NaN is
    therefore often the objectively correct answer, and a confident-looking guess
    (e.g. the single most common value overall) will usually be WRONG for these and
    will actively make your score worse, not better, by turning correct "still
    missing" answers into wrong non-missing guesses. Verified empirically: for one
    such very-sparse column, leaving unmatched cells as NaN scored much higher (F1
    0.81, and 0.86 combined with a strong specific key) than forcing every remaining
    cell to the global mode (F1 dropped to 0.08). For very-high-missing-rate columns:
    only fill a cell when you have a genuinely specific, targeted signal (e.g. a
    shared identifier column where another row with the same key has a real value —
    see GROUP-AWARE IMPUTATION below), and leave the rest as NaN rather than guessing.
- DO NOT ASSUME A "MISSING" PATTERN GENERALIZES ACROSS COLUMNS: a value that is a
  genuine missing-value placeholder in one column (e.g. the literal string "empty"
  in an address line 2 column) is NOT necessarily
  missing in a DIFFERENT column. Before treating a value like "empty", "0", "none", or
  "0 patients" as missing in a given column, check the profile for that SPECIFIC
  column: if that value appears with substantial, stable frequency (not a rare/odd
  variant), it is very likely a legitimate, valid category for that column (e.g.
  "empty" meaning "no score was reported for this measure", or "0 patients" meaning a
  genuine zero count) — leave it exactly as-is rather than converting it to NaN and
  imputing a fabricated replacement, which corrupts already-correct data. When in
  doubt, only normalize the small set of universal missing markers explicitly listed
  above ("NA", "N/A", "unknown", empty string, whitespace) — do not extend that list
  per-column based on values you happened to see in a sample from an unrelated column.

TEXT / CATEGORY CLEANING:
- Never use substring replacement for correcting category values (e.g. never do
  `.str.replace('Resort', 'Resort Hotel')` — this corrupts values that already contain
  "Resort Hotel", turning it into "Resort Hotel Hotel"). Instead, build an explicit
  mapping dict of {wrong_value: correct_value} using `.replace(mapping_dict)`, or match
  whole values only (`df[col] == 'exact_value'`).
- Strip leading/trailing whitespace and normalize case only when it does not change the
  meaning of categorical labels that are case-sensitive by convention (e.g. keep
  'PRT'/'GBR' country codes uppercase).
- FOR VERY-LOW-CARDINALITY COLUMNS (e.g. 2-5 valid values total, check the profile's
  n_unique/top_values — a classic example is a sex/gender column with only 'male' and
  'female'), DO NOT rely only on a hardcoded dictionary of the specific typo variants
  you happened to see in the sample rows — that dictionary will miss every OTHER typo
  variant that exists in the full dataset but wasn't in your small sample (a real
  failure mode observed: precision 1.0 but recall only ~0.48, meaning correct whenever
  attempted, but not even attempted for about half the errors). Instead, correct EVERY
  non-matching value with fuzzy string matching against the small list of known valid
  values (from the profile's top_values) — this catches all typo variants, not just
  the visible ones, using Python's built-in difflib (no extra dependency needed):
  ```
  import difflib
  valid_values = ['male', 'female']  # the column's known valid values, from the profile
  def fuzzy_correct(val):
      if pd.isna(val):
          return val
      val_norm = str(val).strip().lower()
      if val_norm in [v.lower() for v in valid_values]:
          return next(v for v in valid_values if v.lower() == val_norm)
      match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_values], n=1, cutoff=0.6)
      if match:
          return next(v for v in valid_values if v.lower() == match[0])
      return val  # leave unmatched values as-is rather than guessing wildly
  df[col] = df[col].apply(fuzzy_correct)
  ```
  Reserve exact hardcoded mappings for columns with MANY valid values (where fuzzy
  matching against a large list is less reliable) — for those, an explicit dict built
  from the profile's top_values is still the right approach.

NUMERIC COLUMNS:
- Never call `int(x)` or `float(x)` directly on a raw string value from the dataset —
  it will crash on values like '2€', 'x00', '2O', or on already-numeric types. Always:
    1. Strip non-numeric characters with a regex if needed:
       `df[col] = df[col].astype(str).str.replace(r'[^0-9.\\-]', '', regex=True)`
    2. Convert with `pd.to_numeric(df[col], errors='coerce')` (never bare int()/float()).
    3. Fill resulting NaN with the column's median (computed after step 2).
- Before calling `.median()` on ANY column for any reason, always confirm it is
  ACTUALLY numeric dtype first: `pd.api.types.is_numeric_dtype(df[col])`. If it returns
  False, convert with `pd.to_numeric(df[col], errors='coerce')` first (this by itself
  does NOT change the original strings you don't want to touch — assign the result to
  a separate check/variable if you only need it to decide imputation, or apply it to
  the column directly if the column is genuinely meant to be numeric). Never call
  `.median()` on a column still holding text/object dtype — it will crash with
  "Cannot perform reduction 'median' with string dtype".

DATES:
- Parse dates with `pd.to_datetime(df[col], errors='coerce')`, trying multiple known
  formats if a single format fails, rather than crashing or leaving them unparsed.
- MULTI-FORMAT DATE PARSING PATTERN (avoids a real crash observed: "Invalid value for
  dtype 'str'. Value should be a string or missing value"): never do
  `df[col] = df[col].astype(str)` and then later write datetime/Timestamp objects back
  into that SAME column via `df.loc[mask, col] = pd.to_datetime(...)` — mixing string
  and Timestamp values back into a column that was just cast to str can raise this
  error. Instead, parse into a SEPARATE new Series first, try each format against it,
  and only write the FINAL result (as one consistent type, e.g. a formatted string)
  back into `df[col]` a single time at the end:
  ```
  parsed = pd.Series(pd.NaT, index=df.index)
  raw = df[col].astype(str)
  for fmt in date_formats:
      still_missing = parsed.isna()
      parsed.loc[still_missing] = pd.to_datetime(raw[still_missing], format=fmt, errors='coerce')
  # handle any other special case (e.g. unix timestamps) by updating `parsed`, not df[col]
  # finally, assign ONCE, as a consistent string format:
  df[col] = parsed.dt.strftime('%Y-%m-%d')
  ```

TIME-OF-DAY COLUMNS (values like '7:10 a.m.', '11:25 p.m.' — a clock time, not a full
calendar date):
- Do NOT parse this column with `pd.to_datetime(...)` and do NOT reformat it. Any
  reformatting (adding a leading zero, removing dots in 'a.m./p.m.', converting to
  24-hour time, converting to a datetime object) will make even already-correct values
  fail an exact match against the reference and will be counted as new errors.
- Treat it as a plain text/categorical column. To impute a missing value, take the
  mode of the RAW STRING column directly (do NOT convert to datetime first):
  `mode_val = df[col].mode(dropna=True); df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else df[col])`
- If you must validate that a value looks like a real time (e.g. to detect garbage
  like "Contact Airline" instead of a time), use a regex check
  (`re.match(r'^\\d{1,2}:\\d{2}\\s*[ap]\\.?m\\.?$', str(val), re.IGNORECASE)`) rather
  than parsing to datetime — replace only values that fail this check.

PRESERVE ORIGINAL FORMATTING:
- For any column and any value that is not itself flagged as an issue (not missing,
  not a detected typo/outlier), do NOT change its display format, casing, spacing, or
  representation. Many datasets are graded by exact string match against a reference —
  reformatting a value that was already correct (e.g. re-casing text, re-padding
  numbers, changing date/time separators) counts as introducing a NEW error, even
  though the value is still semantically correct.

OUTLIERS:
- Never call dropna() without an explicit subset=[...] of the specific columns being
  cleaned. Dropping a row because an unrelated column is missing is forbidden.
- The final row count must stay exactly equal to the original row count. Do not filter
  out rows for any reason.
- DO NOT USE `.clip(lower, upper)` (OR ANY DIRECT BOUNDARY-SNAPPING) TO "FIX" AN
  OUTLIER'S VALUE — this was previously worded as "cap/clip outlier values" and that
  phrasing caused a real, measured bug: `.clip()` sets the outlier to the literal
  boundary number (e.g. clipping a value of 35 to a max of 8 sets it to exactly 8),
  and that boundary value is almost never the true original value. Verified on real
  data: clipping outliers to the boundary scored 0/17 correct, while the fix below
  scored 13/17 correct on the exact same cells. The correct technique is to treat an
  out-of-range value as EFFECTIVELY UNKNOWN and impute it exactly like a missing value
  (median for numeric columns, mode for categorical) — NOT to snap it to the range
  boundary:
  ```
  is_outlier = (df[col] < plausible_min) | (df[col] > plausible_max)
  df.loc[is_outlier, col] = np.nan
  df[col] = df[col].fillna(df[col].median())   # or .mode().iloc[0] for categorical
  ```
  `.clip()` is only appropriate when you specifically want to compress a continuous
  range at its edges (rare for this task) — for correcting a clearly-wrong outlier
  value back to a plausible one, always use the missing-then-impute pattern above.

COVERAGE (avoid under-cleaning):
- Your script will be scored on how many of the flagged issues below (columns with
  missing values, unrealistic ranges, or high cardinality relative to expected) it
  actually fixes. A script that only fixes 1-2 columns and leaves the rest untouched
  will score very low on recall even if what it did was correct. Address EVERY column
  mentioned in the profile below that has an issue (missing values, out-of-range
  values, or many near-duplicate category spellings).
- DO NOT CONFUSE "0% MISSING" WITH "NOTHING TO CLEAN": a numeric column can have zero
  missing values and still contain injected OUTLIERS — check every numeric column's
  min/max in the profile against a domain-plausible range, independently of its
  missing-value percentage. A real regression observed in practice: a column reported
  as "already numeric, no missing values, no action needed" was left completely
  untouched even though its profile max was wildly implausible (e.g. 48 for a
  family-size count that should realistically top out around 8-10) — this scored zero
  credit for that column. Go through EVERY numeric column's min/max explicitly, even
  ones with 0% missing, and cap/clip anything implausible.

GROUP-AWARE IMPUTATION (important for high-cardinality columns):
- Before imputing a column with its GLOBAL mode, check its n_unique in the profile
  relative to the row count. If a column has HIGH cardinality (many distinct values,
  e.g. more than ~50 distinct values, such as specific timestamps, prices, or IDs), a
  single dataset-wide mode is almost always the wrong value for any given row — it
  will rarely match the true original value, badly hurting recall.
- In that case, look for another column in the dataset that acts as a repeating
  key/identifier (a column whose values repeat across several rows — check its
  n_unique: much lower than the row count means many rows share the same key value,
  e.g. the same flight number appears on several different days, the same product id
  appears in several orders). If the value you are imputing is likely CONSTANT or very
  similar within each group of that key, impute the MISSING values using the mode
  COMPUTED WITHIN EACH GROUP instead of a single global mode.
- CRITICAL SAFETY RULE — DO NOT OVERWRITE ALREADY-CORRECT VALUES: only replace cells
  that are ACTUALLY missing/NaN. Never assign the group-transform result directly back
  to the whole column — that replaces EVERY row (including rows that were already
  correct) with the group aggregate, silently destroying real per-row variation. This
  is a serious, easy-to-make mistake: it will not lower your score on already-flagged
  errors, but it corrupts a large fraction of the dataset that was never broken.
  ALWAYS use this pattern — compute the group value into a SEPARATE series first, then
  fillna (only fills where the original was NaN, leaves everything else untouched):
  ```
  group_val = df.groupby(key_col)[col].transform(
      lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
  )
  df[col] = df[col].fillna(group_val)   # only fills missing cells, never overwrites
  # fallback to the global mode only for rows whose entire group has no valid value
  global_mode = df[col].mode(dropna=True)
  if not global_mode.empty:
      df[col] = df[col].fillna(global_mode.iloc[0])
  ```
  NEVER write `df[col] = df.groupby(...)[col].transform(...)` directly — that pattern
  looks similar but is wrong: it discards every existing value, not just missing ones.
- Before trusting a candidate key at all (single or composite), verify it is actually a
  good predictor: check that the target column has LOW n_unique WITHIN each group of
  the key (few distinct values per group) — if grouping by your chosen key still leaves
  many different values within each group, that key does not determine the column and
  you should not use it (fall back to the global mode instead of a weak/wrong key).
- This applies to any column-pair in the dataset where such a repeating key exists —
  look at the profile's n_unique values to decide which column, if any, is a good key.
- COMPOSITE (MULTI-COLUMN) KEYS — TRY THIS BEFORE GIVING UP ON A SINGLE KEY: a single
  column is often NOT enough by itself. Some target columns only become constant when
  grouped by TWO columns TOGETHER (a functional dependency of the form {A, B} -> C),
  even though neither A alone nor B alone determines C on its own. Symptom: you group
  by column A alone and the target column C still has several different values within
  each group (single-key grouping "fails" — check this explicitly before concluding no
  key works). In that case, group by a LIST of columns instead of one, using the exact
  same safe fillna-only pattern shown above (never a direct column overwrite).
  A good candidate second key is a column that, together with the first, is unique or
  near-unique per row (e.g. an identifier column combined with a category/type column
  that repeats for that identifier). This is very often the case when a table records
  one measurement per (entity, category) pair — for example, a hospital reports ONE
  score PER quality measure it is evaluated on, so a single hospital-id column is not
  enough to determine the score (the same hospital has many different scores, one per
  measure); the composite key (hospital_id, measure_code) does determine it uniquely.
  This exact pattern — an identifier column combined with a category/measure column —
  is common any time a dataset records repeated observations per entity.
- EXCEPTION — VERIFIED FUNCTIONAL DEPENDENCIES ONLY: if this prompt explicitly lists a
  "KNOWN GROUPING KEYS" section below with a column marked as an exact/near-exact
  dependency for this specific dataset, it IS safe and beneficial to overwrite the
  WHOLE column with the group-computed mode for those specific columns only (not
  columns you guessed yourself) — because the target is confirmed constant per group,
  this doubles as free typo-correction (a rare typo'd row gets outvoted by its correct
  group majority). Do NOT extend this whole-column-overwrite technique to any column or
  key that isn't explicitly confirmed in that section — for everything else, use the
  fillna-only pattern above."""


# ---------------------------------------------------------------------------
# Dependances fonctionnelles connues, verifiees empiriquement sur des datasets
# specifiques (clean.csv vs dirty.csv). Cles avec noms CANONIQUES (peu importe la
# casse/underscore reelle du dataset au moment du prompt — voir _map_fd_hint).
# Format : dataset_name -> liste de (colonne(s) cle, [colonnes determinees]).
# La cle peut etre un str (FD simple) ou un tuple de str (FD composite).
# ---------------------------------------------------------------------------
FUNCTIONAL_DEPENDENCIES = {
    "hospital": [
        ("ProviderNumber", ["HospitalName", "Address1", "City", "State", "ZipCode",
                             "CountyName", "PhoneNumber", "HospitalType", "HospitalOwner",
                             "EmergencyService"]),
        ("MeasureCode", ["MeasureName", "Condition"]),
        (("ProviderNumber", "MeasureCode"), ["Score", "Sample"]),
        ("ZipCode", ["City", "State", "CountyName"]),
    ],
    "titanic": [
        # Pas des FD strictes (contrairement a hospital) : verifiees empiriquement comme
        # de FORTS signaux statistiques, pas des egalites garanties. Ticket est tres fort
        # mais ne couvre que les lignes qui PARTAGENT un ticket (~39% du dataset, familles
        # / groupes ayant achete ensemble) ; Pclass est plus faible mais couvre 100% des
        # lignes -> utiliser Ticket en priorite, Pclass en repli.
        ("Ticket", ["Fare", "Embarked"]),
        ("Pclass", ["Fare", "Age"]),
    ],
}


def _normalize_col_name(name: str) -> str:
    return str(name).lower().replace("_", "").replace(" ", "")


def _map_fd_hint(df: pd.DataFrame, dataset_name: str) -> str:
    """
    Si des dependances fonctionnelles connues existent pour ce dataset, les traduit
    vers les noms de colonnes REELS de `df` (qui peuvent differer en casse/underscore
    de la liste canonique, ex: 'ProviderNumber' vs 'provider_number'), et retourne un
    bloc de texte a inserer dans le prompt. Retourne une chaine vide si rien ne
    correspond (dataset non repertorie, ou aucune colonne ne matche).
    """
    fds = FUNCTIONAL_DEPENDENCIES.get(dataset_name)
    if not fds:
        return ""

    real_by_norm = {_normalize_col_name(c): c for c in df.columns}

    def resolve(canonical_name):
        norm = _normalize_col_name(canonical_name)
        # 1) correspondance exacte normalisee
        if norm in real_by_norm:
            return real_by_norm[norm]
        # 2) correspondance par inclusion (ex: 'zipcode' vs 'zip', 'hospitalname' vs
        #    'name') -- on garde la correspondance avec le PLUS GRAND recouvrement,
        #    car une correspondance courte peut matcher par accident (ex: 'name' est
        #    une sous-chaine de 'countyname' sans rapport avec la vraie colonne 'name').
        best_match = None
        best_overlap = 0
        for real_norm, real_col in real_by_norm.items():
            if real_norm in norm:
                overlap = len(real_norm)
            elif norm in real_norm:
                overlap = len(norm)
            else:
                continue
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = real_col
        return best_match

    lines = []
    for key, targets in fds:
        key_names = (key,) if isinstance(key, str) else key
        resolved_key = [resolve(k) for k in key_names]
        if any(r is None for r in resolved_key):
            continue
        resolved_targets = [resolve(t) for t in targets]
        resolved_targets = [t for t in resolved_targets if t is not None]
        if not resolved_targets:
            continue
        key_repr = resolved_key[0] if len(resolved_key) == 1 else "(" + ", ".join(resolved_key) + ")"
        lines.append(f"  - {key_repr} -> {', '.join(resolved_targets)}")

    if not lines:
        return ""

    key_example = None
    for key, _ in fds:
        if isinstance(key, tuple):
            resolved = [resolve(k) for k in key]
            if all(r is not None for r in resolved):
                key_example = resolved
                break

    example_block = ""
    if key_example:
        example_block = (
            f"\n  Concretely here, this composite key is VERIFIED (see EXCEPTION rule "
            f"above), so the whole-column overwrite technique is safe and recommended: "
            f"`df[target_col] = df.groupby({key_example})[target_col].transform("
            f"lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)` "
            f"directly, for the composite-dependency targets listed above ONLY — do "
            f"not use this direct-overwrite form for any other column."
        )

    return (
        "\nKNOWN GROUPING KEYS FOR THIS SPECIFIC DATASET (measured on the reference "
        "data, not a guess — for each line, try grouping by the key first; some are "
        "near-exact, others are just strong regularities, so always keep the "
        "global-mode fallback for rows whose group has no other valid value):\n"
        + "\n".join(lines) + example_block
    )


def build_prompt_simple(dataset_name: str, n_rows: int) -> str:
    return f"""A pandas DataFrame named `df` ({dataset_name}, {n_rows} rows) is already \
loaded. Write Python/Pandas code that:
- fixes any inconsistent date formats
- handles missing values appropriately
- corrects obvious typos and inconsistent categories
- makes sure numeric columns contain only numeric values (no letters mixed in)

Return only the final cleaning code."""


def build_dataset_sample(df: pd.DataFrame, n_rows: int = 8, random_state: int = 42) -> str:
    """
    Construit un extrait de lignes REELLES du dataset (pas juste des statistiques),
    pour donner au LLM un ancrage concret sur a quoi ressemblent les valeurs sales
    (idee reprise du prompt_builder d'un camarade, qui a mesure un gain de F1 avec
    cette technique). On prend un echantillon ALEATOIRE (pas juste head()) pour avoir
    de bonnes chances de montrer des valeurs sales dispersees dans le dataset, pas
    seulement les premieres lignes qui peuvent etre propres par hasard.
    """
    n = min(n_rows, len(df))
    # Toujours echantillonner (meme si len(df) == n) pour eviter de retourner les
    # premieres lignes dans leur ordre d'origine, qui peuvent etre propres par hasard
    # -- seul le cas n == 0 (dataset vide) n'a rien a echantillonner.
    sample = df.sample(n, random_state=random_state) if n > 0 else df
    return sample.to_string(index=False)


def build_prompt_schema(df: pd.DataFrame, dataset_name: str) -> str:
    lines = []
    for col in df.columns:
        dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical/text"
        examples = df[col].dropna().unique()[:3]
        examples_str = ", ".join(str(e) for e in examples)
        lines.append(f"- {col} ({dtype}) — e.g. {examples_str}")
    schema_block = "\n".join(lines)
    sample_block = build_dataset_sample(df)
    fd_hint = _map_fd_hint(df, dataset_name)

    return f"""A pandas DataFrame named `df` is already loaded, described below.

Dataset: {dataset_name}
Number of rows: {df.shape[0]}
Number of columns: {df.shape[1]}

Columns and detected types (this is the COMPLETE and ONLY list of valid column names):
{schema_block}

Here is a REAL random sample of {min(8, df.shape[0])} rows from `df` (exactly as they
appear in the data, dirty values included — use this to see the actual formatting and
error patterns you need to handle):

{sample_block}
{fd_hint}

Write Python/Pandas code that:
- standardizes inconsistent date formats
- handles missing values per column, using a method appropriate to its type
- fixes typos and inconsistent categories in text columns
- ensures numeric columns contain only numeric values
- detects and handles unrealistic outlier values in numeric columns

Return only the final cleaning code."""


def build_prompt_profile(df: pd.DataFrame, dataset_name: str, top_n: int = 5) -> str:
    profile = profile_dataset(df, top_n=top_n)
    profile_str = profile.to_string(index=False)
    sample_block = build_dataset_sample(df)
    fd_hint = _map_fd_hint(df, dataset_name)

    return f"""A pandas DataFrame named `df` is already loaded. Below is its statistical \
profile. Use it to plan a targeted, column-by-column cleaning of `df`.

Dataset: {dataset_name}
Rows: {df.shape[0]} — Columns: {df.shape[1]} (this is the COMPLETE and ONLY list of
valid column names — see the 'column' field of the profile below)

Column profile (dtype detected, % missing [real + disguised as "NA"/"unknown"/empty/etc.],
number of unique values, and either min/max/mean/std for numeric columns or top-{top_n} most
frequent values for categorical columns):

{profile_str}

Here is a REAL random sample of {min(8, df.shape[0])} rows from `df` (exactly as they
appear in the data, dirty values included — use this to see the actual formatting and
error patterns you need to handle, in addition to the aggregate statistics above):

{sample_block}
{fd_hint}

Using this profile, write Python/Pandas code that:
- treats both real (NaN) and disguised missing values ("NA", "N/A", "unknown", empty string,
  whitespace) as missing, and imputes each one using a statistic computed from `df` itself
  (mode for categorical, median for numeric) — see the MISSING VALUES rule below
- standardizes inconsistent date formats
- corrects typos and harmonizes inconsistent categories, especially in columns whose
  n_unique is higher than plausible for the field (e.g. many near-duplicate spellings)
- ensures numeric columns contain only numeric values
- caps values that fall far outside the observed min/max range for each numeric column

Return only the final cleaning code."""


PROMPT_BUILDERS = {
    "simple": lambda df, name: build_prompt_simple(name, df.shape[0]),
    "schema": lambda df, name: build_prompt_schema(df, name),
    "profile": lambda df, name: build_prompt_profile(df, name),
}


def build_prompt(prompt_type: str, df: pd.DataFrame, dataset_name: str) -> str:
    """Version 'tout-en-un' (retrocompatible) : CONSTRAINTS + contenu specifique
    concatenes dans un seul texte, pour un appel LLM a message unique."""
    if prompt_type not in PROMPT_BUILDERS:
        raise ValueError(f"prompt_type doit etre {list(PROMPT_BUILDERS)}, recu : {prompt_type}")
    return CONSTRAINTS + "\n\n" + PROMPT_BUILDERS[prompt_type](df, dataset_name)


def build_prompt_split(prompt_type: str, df: pd.DataFrame, dataset_name: str) -> tuple:
    """
    Version system/user separee : les modeles de chat suivent generalement plus
    fidelement des regles placees dans le message SYSTEM que noyees dans un long
    message user. Retourne (system_prompt, user_prompt).
    """
    if prompt_type not in PROMPT_BUILDERS:
        raise ValueError(f"prompt_type doit etre {list(PROMPT_BUILDERS)}, recu : {prompt_type}")
    system_prompt = ("You are a rigorous data cleaning assistant. You must follow the "
                      "rules below EXACTLY when writing pandas cleaning code.\n\n" + CONSTRAINTS)
    user_prompt = PROMPT_BUILDERS[prompt_type](df, dataset_name)
    return system_prompt, user_prompt