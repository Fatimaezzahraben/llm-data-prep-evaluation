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

# --- address_2 & address_3: all 'empty' -> NaN, then impute with mode ---
for col in ['address_2', 'address_3']:
    df[col] = df[col].replace({'empty': np.nan})
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val.iloc[0])

# --- state: fuzzy-correct low-cardinality column ---
valid_states = ['al', 'ak']  # from profile
df['state'] = df['state'].apply(lambda x: fuzzy_correct_low_cardinality(x, valid_states))
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

# --- score: clean with suffix preservation and typo correction ---
# First handle 'empty' and extract numeric part
df['score'] = df['score'].replace({'empty': np.nan})
# Extract numeric part and preserve % suffix
df['score'] = df['score'].apply(lambda x: extract_numeric_with_suffix(x, '%'))
# Handle specific typo patterns (x6% -> 96%, etc.)
score_mapping = {
    '6%': '96%',
    '6': '96%',
    '6.0': '96%',
    '89x': '89%',
    '1xx%': '100%'
}
df['score'] = df['score'].apply(lambda x: correct_typo_with_mapping(x, score_mapping))
# Group-aware imputation (composite key)
group_score = df.groupby(['provider_number', 'measure_code'])['score'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['score'] = df['score'].fillna(group_score)
# Global fallback
global_mode_score = df['score'].mode()
if not global_mode_score.empty:
    df['score'] = df['score'].fillna(global_mode_score.iloc[0])

# --- sample: clean with suffix preservation and typo correction ---
df['sample'] = df['sample'].replace({'empty': np.nan})
# Extract numeric part and preserve 'patients' suffix
df['sample'] = df['sample'].apply(lambda x: extract_numeric_with_suffix(x, 'patients'))
# Handle specific typo patterns
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
group_sample = df.groupby(['provider_number', 'measure_code'])['sample'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['sample'] = df['sample'].fillna(group_sample)
# Global fallback
global_mode_sample = df['sample'].mode()
if not global_mode_sample.empty:
    df['sample'] = df['sample'].fillna(global_mode_sample.iloc[0])

# --- measure_code: typo correction ---
measure_code_mapping = {
    'pn-x': 'pn-2',
    'ami-7a': 'ami-7',
    'alxpn-3b': 'pn-3b'
}
df['measure_code'] = df['measure_code'].apply(lambda x: correct_typo_with_mapping(x, measure_code_mapping))
# Group-aware imputation (single key)
group_measure_code = df.groupby('measure_name')['measure_code'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['measure_code'] = df['measure_code'].fillna(group_measure_code)
# Global fallback
global_mode_measure_code = df['measure_code'].mode()
if not global_mode_measure_code.empty:
    df['measure_code'] = df['measure_code'].fillna(global_mode_measure_code.iloc[0])

# --- provider_number: typo correction ---
provider_mapping = {
    '100x8': '10038',
    '1xx45': '10045',
    '1010x': '10108',
    '10x08': '10008'
}
df['provider_number'] = df['provider_number'].apply(lambda x: correct_typo_with_mapping(x, provider_mapping))
# Group-aware imputation (single key)
group_provider = df.groupby('name')['provider_number'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['provider_number'] = df['provider_number'].fillna(group_provider)
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