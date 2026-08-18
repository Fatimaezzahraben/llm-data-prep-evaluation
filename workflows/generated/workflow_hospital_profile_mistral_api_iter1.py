import pandas as pd
import numpy as np
import difflib

# --- address_2 & address_3: all 'empty' -> NaN, then impute with mode (global fallback) ---
for col in ['address_2', 'address_3']:
    df[col] = df[col].replace({'empty': np.nan})
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val.iloc[0])

# --- state: fuzzy-correct low-cardinality column ---
valid_states = ['al', 'ak']  # from profile: only 4 values, but 'xl' and 'ax' are typos
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
# Impute any remaining missing with mode
mode_state = df['state'].mode()
if not mode_state.empty:
    df['state'] = df['state'].fillna(mode_state.iloc[0])

# --- type: fuzzy-correct low-cardinality column ---
valid_types = ['acute care hospitals']  # only one valid value, others are typos
def fuzzy_correct_type(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm == 'acute care hospitals'.lower():
        return 'acute care hospitals'
    match = difflib.get_close_matches(val_norm, ['acute care hospitals'.lower()], n=1, cutoff=0.6)
    if match:
        return 'acute care hospitals'
    return val
df['type'] = df['type'].apply(fuzzy_correct_type)
# Impute any remaining missing with mode
mode_type = df['type'].mode()
if not mode_type.empty:
    df['type'] = df['type'].fillna(mode_type.iloc[0])

# --- emergency_service: fuzzy-correct low-cardinality column ---
valid_emergency = ['yes', 'no']
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
# Impute any remaining missing with mode
mode_emergency = df['emergency_service'].mode()
if not mode_emergency.empty:
    df['emergency_service'] = df['emergency_service'].fillna(mode_emergency.iloc[0])

# --- score: clean numeric, treat 'empty' as missing, then impute per (provider_number, measure_code) ---
df['score'] = df['score'].replace({'empty': np.nan})
# Extract numeric part from strings like '1xx%' or '97%'
df['score'] = df['score'].astype(str).str.replace(r'[^0-9]', '', regex=True)
df['score'] = pd.to_numeric(df['score'], errors='coerce')
# Group-aware imputation (composite key)
group_score = df.groupby(['provider_number', 'measure_code'])['score'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['score'] = df['score'].fillna(group_score)
# Global fallback
global_mode_score = df['score'].mode()
if not global_mode_score.empty:
    df['score'] = df['score'].fillna(global_mode_score.iloc[0])

# --- sample: clean numeric, treat 'empty' as missing, then impute per (provider_number, measure_code) ---
df['sample'] = df['sample'].replace({'empty': np.nan})
# Extract numeric part from strings like '0 patients' or '1 patients'
df['sample'] = df['sample'].astype(str).str.replace(r'[^0-9]', '', regex=True)
df['sample'] = pd.to_numeric(df['sample'], errors='coerce')
# Group-aware imputation (composite key)
group_sample = df.groupby(['provider_number', 'measure_code'])['sample'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['sample'] = df['sample'].fillna(group_sample)
# Global fallback
global_median_sample = df['sample'].median()
df['sample'] = df['sample'].fillna(global_median_sample)

# --- measure_name: impute per measure_code (known grouping key) ---
df['measure_name'] = df.groupby('measure_code')['measure_name'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
# Global fallback
global_mode_measure_name = df['measure_name'].mode()
if not global_mode_measure_name.empty:
    df['measure_name'] = df['measure_name'].fillna(global_mode_measure_name.iloc[0])

# --- condition: impute per measure_code (known grouping key) ---
df['condition'] = df.groupby('measure_code')['condition'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
# Global fallback
global_mode_condition = df['condition'].mode()
if not global_mode_condition.empty:
    df['condition'] = df['condition'].fillna(global_mode_condition.iloc[0])

# --- name, address_1, city, zip, county, phone, owner, type, emergency_service: impute per provider_number (known grouping key) ---
for col in ['name', 'address_1', 'city', 'zip', 'county', 'phone', 'owner', 'type', 'emergency_service']:
    df[col] = df.groupby('provider_number')[col].transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    )
    # Global fallback
    global_mode = df[col].mode()
    if not global_mode.empty:
        df[col] = df[col].fillna(global_mode.iloc[0])

# --- index: ensure numeric, no missing, no outliers ---
df['index'] = pd.to_numeric(df['index'], errors='coerce')
# Cap outliers (min=1, max=1000 from profile)
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

# --- provider_number: ensure text, no missing ---
df['provider_number'] = df['provider_number'].astype(str).replace({'nan': np.nan})
mode_provider = df['provider_number'].mode()
if not mode_provider.empty:
    df['provider_number'] = df['provider_number'].fillna(mode_provider.iloc[0])

# --- measure_code: ensure text, no missing ---
df['measure_code'] = df['measure_code'].astype(str).replace({'nan': np.nan})
mode_measure_code = df['measure_code'].mode()
if not mode_measure_code.empty:
    df['measure_code'] = df['measure_code'].fillna(mode_measure_code.iloc[0])

df