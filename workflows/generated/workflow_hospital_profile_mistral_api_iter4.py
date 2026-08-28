import pandas as pd
import numpy as np
import difflib
import re

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
valid_states = ['al', 'ak']  # from profile
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
valid_types = ['acute care hospitals']  # from profile
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
valid_emergency = ['yes', 'no']  # from profile
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

# --- 8. Clean 'measure_code' column (pattern-based correction + fuzzy matching) ---
# Create mapping for common measure code patterns
measure_code_mapping = {
    r'^ami-': 'ami-',
    r'^hf-': 'hf-',
    r'^pn-': 'pn-',
    r'^scip-': 'scip-',
    r'^scix-': 'scip-',
    r'^scx-': 'scip-',
    r'^axi-': 'ami-',
    r'^amix': 'ami-',
    r'^xf-': 'hf-',
    r'^px-': 'pn-'
}

def clean_measure_code(val):
    if pd.isna(val):
        return val
    val_str = str(val).strip().lower()

    # Apply pattern-based corrections
    for pattern, replacement in measure_code_mapping.items():
        if re.search(pattern, val_str):
            # Extract the numeric part
            num_part = re.sub(r'[^0-9-]', '', val_str)
            if num_part:
                return f"{replacement}{num_part}"
            return replacement

    # Handle specific known cases
    if 'inf' in val_str:
        return re.sub(r'inf', 'inf', val_str)
    if 'vte' in val_str:
        return re.sub(r'vte', 'vte', val_str)
    if 'card' in val_str:
        return re.sub(r'card', 'card', val_str)

    return val

df['measure_code'] = df['measure_code'].apply(clean_measure_code)

# Post-processing: fuzzy matching against known valid codes
valid_measure_codes = ['hf-3', 'scip-card-2', 'hf-4', 'pn-2', 'pn-3b', 'ami-2', 'ami-3', 'ami-4', 'ami-7a',
                      'scip-inf-2', 'scip-inf-3', 'scip-inf-4', 'scip-inf-6', 'scip-vte-2', 'pn-4']
def fuzzy_correct_measure_code(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    # Check for exact match first
    if val_norm in [v.lower() for v in valid_measure_codes]:
        return next(v for v in valid_measure_codes if v.lower() == val_norm)
    # Fuzzy match
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_measure_codes], n=1, cutoff=0.7)
    if match:
        return next(v for v in valid_measure_codes if v.lower() == match[0])
    return val

df['measure_code'] = df['measure_code'].apply(fuzzy_correct_measure_code)

# --- 9. Clean 'provider_number' column (pattern-based correction) ---
def clean_provider_number(val):
    if pd.isna(val):
        return val
    val_str = str(val).strip().lower()

    # Extract all digits
    digits = re.sub(r'[^0-9]', '', val_str)
    if not digits:
        return val

    # Handle common patterns
    if len(digits) == 4 and 'x' in val_str:
        # Pattern like '1xx19' -> '10019'
        if val_str.startswith('1xx'):
            return f"100{digits[-2:]}"
        elif val_str.startswith('x00'):
            return f"{digits[0]}00{digits[-2:]}"
        elif val_str.endswith('xx'):
            return f"{digits[:-2]}00"
    elif len(digits) == 3 and 'x' in val_str:
        # Pattern like 'x00x5' -> '10015'
        if val_str.startswith('x00x'):
            return f"100{digits[-1]}"
        elif val_str.startswith('1xx'):
            return f"100{digits[-1]}"

    return val

df['provider_number'] = df['provider_number'].apply(clean_provider_number)

# Post-processing: fuzzy matching against known valid provider numbers
valid_provider_numbers = ['10046', '10038', '10018', '10001', '10007', '10023', '10035', '10032', '10108', '10033', '10029', '10022']
def fuzzy_correct_provider_number(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    # Check for exact match first
    if val_norm in [v.lower() for v in valid_provider_numbers]:
        return next(v for v in valid_provider_numbers if v.lower() == val_norm)
    # Extract digits for comparison
    digits = re.sub(r'[^0-9]', '', val_norm)
    if digits:
        # Find closest valid provider number
        current_num = int(digits) if digits else 0
        closest = min(valid_provider_numbers, key=lambda x: abs(int(x) - current_num))
        return closest
    return val

df['provider_number'] = df['provider_number'].apply(fuzzy_correct_provider_number)

# --- 10. Clean 'score' column (improved pattern-based parsing) ---
def clean_score(val):
    if pd.isna(val) or val == 'empty':
        return np.nan

    val_str = str(val).strip()

    # Extract all digits
    digits = re.sub(r'[^0-9]', '', val_str)
    if not digits:
        return np.nan

    # Handle common patterns
    if '%' in val_str:
        # For patterns like '9x%', 'x5%', '1xx%'
        if len(digits) == 1:
            # Single digit with x - likely 9x% or x5%
            if val_str.startswith('x'):
                return f"9{digits}%"
            else:
                return f"{digits}8%"
        elif len(digits) == 2:
            # Two digits with one x - likely 9x% or x5%
            if 'x' in val_str:
                if val_str.startswith('x'):
                    return f"9{digits[1]}%"
                else:
                    return f"{digits[0]}8%"
            return f"{digits}%"
        elif len(digits) >= 3:
            # Three or more digits - likely 1xx%
            return "100%"
        elif 'x' in val_str:
            # Handle cases like 'x7%' -> '97%'
            if val_str.startswith('x'):
                return f"9{digits}%"
    else:
        # For non-percentage patterns, return as is if it looks like a number
        if digits.isdigit():
            return f"{digits}%"

    return val_str

df['score'] = df['score'].apply(clean_score)

# Post-processing: fuzzy matching against known valid scores
valid_scores = ['100%', '97%', '98%', '99%', '95%', '92%', '90%', '85%', '80%']
def fuzzy_correct_score(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    # Check for exact match first
    if val_norm in [v.lower() for v in valid_scores]:
        return next(v for v in valid_scores if v.lower() == val_norm)
    # Extract digits for comparison
    digits = re.sub(r'[^0-9]', '', val_norm)
    if digits:
        # Find closest valid score
        current_num = int(digits) if digits else 0
        closest = min(valid_scores, key=lambda x: abs(int(re.sub(r'[^0-9]', '', x)) - current_num))
        return closest
    return val

df['score'] = df['score'].apply(fuzzy_correct_score)

# --- 11. Clean 'sample' column (pattern-based parsing) ---
def clean_sample(val):
    if pd.isna(val) or val == 'empty':
        return np.nan
    val_str = str(val).strip()
    # Extract all digits
    digits = re.sub(r'[^0-9]', '', val_str)
    if not digits:
        return '0 patients'
    return f"{digits} patients"
df['sample'] = df['sample'].apply(clean_sample)

# --- 12. Impute remaining missing values (mode for categorical, median for numeric) ---
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

# --- 13. Ensure numeric columns are properly typed ---
numeric_cols = ['index']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# --- 14. Final cleanup: strip whitespace from all string columns ---
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

df