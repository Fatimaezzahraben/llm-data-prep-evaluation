import pandas as pd
import numpy as np
import difflib

# --- 1. Handle disguised missing values (empty string, whitespace, "NA", "N/A", "unknown") ---
disguised_missing = ['NA', 'N/A', 'unknown', '', ' ']
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].replace(disguised_missing, np.nan)
        df[col] = df[col].str.strip()  # strip whitespace from remaining values

# --- 2. EXACT grouping keys (provider_number -> targets) ---
exact_provider_targets = ['name', 'address_1', 'city', 'state', 'zip', 'county', 'phone', 'type', 'owner', 'emergency_service']
for target in exact_provider_targets:
    if target in df.columns:
        df[target] = df.groupby('provider_number')[target].transform(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        )

# --- 3. EXACT grouping keys (measure_code -> targets) ---
exact_measure_targets = ['measure_name', 'condition']
for target in exact_measure_targets:
    if target in df.columns:
        df[target] = df.groupby('measure_code')[target].transform(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        )

# --- 4. EXACT grouping keys (zip -> targets) ---
exact_zip_targets = ['city', 'state', 'county']
for target in exact_zip_targets:
    if target in df.columns:
        df[target] = df.groupby('zip')[target].transform(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        )

# --- 5. Clean 'state' column (low cardinality, fuzzy match) ---
valid_states = ['al', 'ak']  # from profile: 'al': 954, 'ak': 20, 'xl': 16, 'ax': 10
def fuzzy_correct_state(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_states]:
        return next(v for v in valid_states if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_states], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_states if v.lower() == match[0])
    return val
df['state'] = df['state'].apply(fuzzy_correct_state)

# --- 6. Clean 'type' column (low cardinality, fuzzy match) ---
valid_types = ['acute care hospitals']  # from profile: 'acute care hospitals': 968
def fuzzy_correct_type(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_types]:
        return next(v for v in valid_types if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_types], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_types if v.lower() == match[0])
    return val
df['type'] = df['type'].apply(fuzzy_correct_type)

# --- 7. Clean 'emergency_service' column (low cardinality, fuzzy match) ---
valid_emergency = ['yes', 'no']  # from profile: 'yes': 830, 'no': 143
def fuzzy_correct_emergency(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_emergency]:
        return next(v for v in valid_emergency if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_emergency], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_emergency if v.lower() == match[0])
    return val
df['emergency_service'] = df['emergency_service'].apply(fuzzy_correct_emergency)

# --- 8. Clean 'score' column (pattern-based parsing) ---
def clean_score(val):
    if pd.isna(val) or val == 'empty':
        return np.nan
    val_str = str(val).strip()
    # Extract all digits and '%' (handle cases like '1xx%' -> '100%')
    digits = ''.join([c for c in val_str if c.isdigit()])
    if not digits:
        return np.nan
    # Reconstruct percentage (assume 2-3 digits)
    if len(digits) == 1:
        score = f"{digits}0%"
    elif len(digits) == 2:
        score = f"{digits}%"
    elif len(digits) >= 3:
        score = f"{digits[:2]}%"
    else:
        score = np.nan
    return score
df['score'] = df['score'].apply(clean_score)

# --- 9. Clean 'sample' column (pattern-based parsing) ---
def clean_sample(val):
    if pd.isna(val) or val == 'empty':
        return np.nan
    val_str = str(val).strip()
    # Extract all digits (handle cases like 'x patients' -> '0 patients')
    digits = ''.join([c for c in val_str if c.isdigit()])
    if not digits:
        return '0 patients'
    return f"{digits} patients"
df['sample'] = df['sample'].apply(clean_sample)

# --- 10. Impute remaining missing values (mode for categorical, median for numeric) ---
for col in df.columns:
    if col == 'index':
        continue  # already numeric, no missing
    if pd.api.types.is_numeric_dtype(df[col]):
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    else:
        _mode = df[col].mode(dropna=True)
        fill_value = _mode.iloc[0] if not _mode.empty else np.nan
        df[col] = df[col].fillna(fill_value)

# --- 11. Ensure numeric columns are properly typed ---
numeric_cols = ['index']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# --- 12. Final cleanup: strip whitespace from all string columns ---
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

df