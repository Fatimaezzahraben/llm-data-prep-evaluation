import pandas as pd
import numpy as np
import difflib
import re

# --- Helper functions ---
def extract_numeric_with_suffix(val, suffix):
    """Extract numeric part and reattach suffix if present in original value"""
    if pd.isna(val) or val == 'empty':
        return np.nan
    val_str = str(val)
    # Extract numeric part
    num_match = re.search(r'(\d+)', val_str)
    if not num_match:
        return np.nan
    num = num_match.group(1)
    # Check if suffix exists in original value
    if suffix.lower() in val_str.lower():
        return f"{num} {suffix}"
    return num

def correct_typo_with_mapping(val, mapping):
    """Correct typos using exact mapping, preserving case"""
    if pd.isna(val):
        return val
    val_str = str(val).strip()
    return mapping.get(val_str, val)

def fuzzy_correct_low_cardinality(val, valid_values):
    """Fuzzy correct low-cardinality categorical values"""
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_values]:
        return next(v for v in valid_values if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_values], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_values if v.lower() == match[0])
    return val

def correct_measure_code(val):
    """Correct measure_code with pattern-based rules"""
    if pd.isna(val):
        return val
    val_str = str(val).strip().lower()

    # Common patterns
    patterns = {
        r'^amix(\d+)': r'ami-\1',
        r'^ami(\d+)a$': r'ami-\1',
        r'^hx-(\d+)': r'hf-\1',
        r'^sxip-': 'scip-',
        r'^axi-(\d+)': r'ami-\1',
        r'^alxpn-': 'pn-',
        r'^pn-x': 'pn-2',
        r'^scip-inf-(\d+)': r'scip-inf-\1',
        r'^scip-vte-(\d+)': r'scip-vte-\1',
        r'^scip-card-(\d+)': r'scip-card-\1'
    }

    for pattern, replacement in patterns.items():
        if re.match(pattern, val_str):
            return re.sub(pattern, replacement, val_str)

    return val

def correct_score(val):
    """Correct score values with pattern-based rules"""
    if pd.isna(val) or val == 'empty':
        return val

    val_str = str(val).strip().lower()

    # Handle common score patterns
    if re.match(r'^x\d+%$', val_str):
        num = val_str[1:-1]
        return f"{100 - int(num)}%"
    elif re.match(r'^\d+x%$', val_str):
        num = val_str[:-2]
        return f"{num}%"
    elif re.match(r'^x\d+$', val_str):
        num = val_str[1:]
        return f"{100 - int(num)}%"
    elif re.match(r'^\d+x$', val_str):
        num = val_str[:-1]
        return f"{num}%"
    elif val_str in ['6%', '6', '6.0']:
        return '96%'
    elif val_str == '89x':
        return '89%'
    elif val_str == '1xx%':
        return '100%'

    return val

def correct_provider_number(val):
    """Correct provider_number with pattern-based rules"""
    if pd.isna(val):
        return val
    val_str = str(val).strip().lower()

    # Handle common patterns
    if re.match(r'^x\d{4}$', val_str):
        return val_str[1:]
    elif re.match(r'^1xx(\d{2})$', val_str):
        return f"100{val_str[-2:]}"
    elif re.match(r'^10x(\d{2})$', val_str):
        return f"100{val_str[-2:]}"
    elif re.match(r'^1010x$', val_str):
        return '10108'
    elif re.match(r'^100x(\d)$', val_str):
        return f"100{val_str[-1]}"

    return val

def correct_state(val):
    """Correct state values with pattern-based rules"""
    if pd.isna(val):
        return val
    val_str = str(val).strip().lower()

    # Common state typos
    if val_str in ['xl', 'ax', 'alx']:
        return 'al'
    elif val_str == 'ak':
        return 'ak'  # Keep as valid value

    return val

# --- address_2 & address_3: all 'empty' -> NaN, then impute with mode ---
for col in ['address_2', 'address_3']:
    df[col] = df[col].replace({'empty': np.nan})
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val.iloc[0])

# --- state: fuzzy-correct low-cardinality column with enhanced correction ---
valid_states = ['al', 'ak']
df['state'] = df['state'].apply(correct_state)
mode_state = df['state'].mode()
if not mode_state.empty:
    df['state'] = df['state'].fillna(mode_state.iloc[0])

# --- type: fuzzy-correct low-cardinality column ---
valid_types = ['acute care hospitals']
df['type'] = df['type'].apply(lambda x: fuzzy_correct_low_cardinality(x, valid_types))
mode_type = df['type'].mode()
if not mode_type.empty:
    df['type'] = df['type'].fillna(mode_type.iloc[0])

# --- emergency_service: fuzzy-correct low-cardinality column ---
valid_emergency = ['yes', 'no']
df['emergency_service'] = df['emergency_service'].apply(lambda x: fuzzy_correct_low_cardinality(x, valid_emergency))
mode_emergency = df['emergency_service'].mode()
if not mode_emergency.empty:
    df['emergency_service'] = df['emergency_service'].fillna(mode_emergency.iloc[0])

# --- score: clean with suffix preservation and enhanced correction ---
df['score'] = df['score'].replace({'empty': np.nan})
df['score'] = df['score'].apply(correct_score)
# Extract numeric part and preserve % suffix
df['score'] = df['score'].apply(lambda x: extract_numeric_with_suffix(x, '%') if pd.notna(x) else x)
# Group-aware imputation (composite key)
df['score'] = df.groupby(['provider_number', 'measure_code'])['score'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
# Global fallback
global_mode_score = df['score'].mode()
if not global_mode_score.empty:
    df['score'] = df['score'].fillna(global_mode_score.iloc[0])

# --- sample: clean with suffix preservation and typo correction ---
df['sample'] = df['sample'].replace({'empty': np.nan})
df['sample'] = df['sample'].apply(lambda x: extract_numeric_with_suffix(x, 'patients') if pd.notna(x) else x)
# Handle specific patterns
sample_mapping = {
    '0': '0 patients',
    '1': '1 patients',
    '2': '2 patients',
    '4': '4 patients',
    '6xpatients': '6 patients',
    '193 paxienxs': '193 patients',
    '44 paxienxs': '44 patients'
}
df['sample'] = df['sample'].apply(lambda x: correct_typo_with_mapping(x, sample_mapping))
# Group-aware imputation (composite key)
df['sample'] = df.groupby(['provider_number', 'measure_code'])['sample'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
# Global fallback
global_mode_sample = df['sample'].mode()
if not global_mode_sample.empty:
    df['sample'] = df['sample'].fillna(global_mode_sample.iloc[0])

# --- measure_code: enhanced correction with pattern-based rules ---
df['measure_code'] = df['measure_code'].apply(correct_measure_code)
# Group-aware imputation (single key)
df['measure_code'] = df.groupby('measure_name')['measure_code'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
# Global fallback
global_mode_measure_code = df['measure_code'].mode()
if not global_mode_measure_code.empty:
    df['measure_code'] = df['measure_code'].fillna(global_mode_measure_code.iloc[0])

# --- provider_number: enhanced correction with pattern-based rules ---
df['provider_number'] = df['provider_number'].apply(correct_provider_number)
# Group-aware imputation (single key)
df['provider_number'] = df.groupby('name')['provider_number'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
# Global fallback
global_mode_provider = df['provider_number'].mode()
if not global_mode_provider.empty:
    df['provider_number'] = df['provider_number'].fillna(global_mode_provider.iloc[0])

# --- county: typo correction ---
county_mapping = {
    'chxrokxx': 'cherokee'
}
df['county'] = df['county'].apply(lambda x: correct_typo_with_mapping(x, county_mapping))

# --- measure_name and condition: impute per measure_code ---
for col in ['measure_name', 'condition']:
    df[col] = df.groupby('measure_code')[col].transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    )
    global_mode = df[col].mode()
    if not global_mode.empty:
        df[col] = df[col].fillna(global_mode.iloc[0])

# --- name, address_1, city, zip, county, phone, owner, type, emergency_service: impute per provider_number ---
for col in ['name', 'address_1', 'city', 'zip', 'county', 'phone', 'owner', 'type', 'emergency_service']:
    df[col] = df.groupby('provider_number')[col].transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    )
    global_mode = df[col].mode()
    if not global_mode.empty:
        df[col] = df[col].fillna(global_mode.iloc[0])

# --- index: ensure numeric, no missing, no outliers ---
df['index'] = pd.to_numeric(df['index'], errors='coerce')
df.loc[df['index'] < 1, 'index'] = np.nan
df.loc[df['index'] > 1000, 'index'] = np.nan
df['index'] = df['index'].fillna(df['index'].median())

# --- zip: ensure text, no missing ---
df['zip'] = df['zip'].astype(str).replace({'nan': np.nan})
mode_zip = df['zip'].mode()
if not mode_zip.empty:
    df['zip'] = df['zip'].fillna(mode_zip.iloc[0])

# --- phone: ensure text, no missing ---
df['phone'] = df['phone'].astype(str).replace({'nan': np.nan})
mode_phone = df['phone'].mode()
if not mode_phone.empty:
    df['phone'] = df['phone'].fillna(mode_phone.iloc[0])

# --- state_average: ensure text, no missing ---
df['state_average'] = df['state_average'].astype(str).replace({'nan': np.nan})
mode_state_avg = df['state_average'].mode()
if not mode_state_avg.empty:
    df['state_average'] = df['state_average'].fillna(mode_state_avg.iloc[0])

df