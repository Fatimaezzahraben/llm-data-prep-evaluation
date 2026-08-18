import pandas as pd
import numpy as np
import re

# Standardize text columns by fixing common typos and inconsistencies
text_columns = ['provider_number', 'name', 'address_1', 'city', 'state', 'zip',
                'county', 'phone', 'type', 'owner', 'emergency_service',
                'condition', 'measure_code', 'measure_name']

# Common typo mappings for each column
typo_mappings = {
    'state': {
        'xl': 'al', 'ax': 'al', 'alabama': 'al', 'a1': 'al'
    },
    'type': {
        'acuxe care hospixals': 'acute care hospitals',
        'acxte care hospitals': 'acute care hospitals',
        'acute care hospitax': 'acute care hospitals',
        'acute care hospixals': 'acute care hospitals'
    },
    'owner': {
        'government - hospxtal dxstrxct or authorxty': 'government - hospital district or authority',
        'government - hospital dxstrxct or authority': 'government - hospital district or authority',
        'voluntary non-profit - prxvate': 'voluntary non-profit - private'
    },
    'emergency_service': {
        'yxs': 'yes', 'yex': 'yes', 'y es': 'yes', 'ye s': 'yes'
    },
    'measure_code': {
        'scip-card-2': 'scip-card-2',  # No typos found in example, but keeping for consistency
        'scip-inf-1': 'scip-inf-1',
        'scip-inf-2': 'scip-inf-2'
    }
}

# Apply typo corrections
for col in text_columns:
    if col in typo_mappings:
        df[col] = df[col].replace(typo_mappings[col])
    # Standardize case and whitespace
    if col not in ['provider_number', 'zip', 'phone', 'measure_code']:  # Don't lowercase these
        df[col] = df[col].str.strip()
        df[col] = df[col].str.lower()
        df[col] = df[col].str.title() if col in ['name', 'city', 'county'] else df[col]

# Handle missing values in text columns
for col in text_columns:
    # Identify missing values (empty strings, whitespace, 'empty', etc.)
    missing_mask = df[col].isna() | (df[col].astype(str).str.strip() == '') | (df[col].astype(str).str.lower() == 'empty')
    if missing_mask.any():
        # For high cardinality columns, try group-aware imputation first
        if col in ['name', 'address_1', 'city', 'county', 'phone', 'measure_name']:
            # Use provider_number as key for group-aware imputation
            if 'provider_number' in df.columns:
                df[col] = df.groupby('provider_number')[col].transform(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
                )
        # Fallback to global mode
        mode_val = df[col].mode(dropna=True)
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])

# Clean and convert numeric columns
numeric_columns = ['index', 'score', 'sample']

for col in numeric_columns:
    if col == 'index':
        # Already numeric, just ensure no missing values
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    elif col == 'score':
        # Extract numeric values from percentage strings
        df[col] = df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Cap outliers (0-100 range)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        df[col] = df[col].clip(0, 100)
    elif col == 'sample':
        # Extract numeric values from strings like "33 patients"
        df[col] = df[col].astype(str).str.extract(r'(\d+)', expand=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Cap unrealistic values (e.g., >1000 patients)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        df[col] = df[col].clip(0, 1000)

# Clean phone numbers (treat as text but standardize format)
df['phone'] = df['phone'].astype(str).str.replace(r'[^0-9]', '', regex=True)
# Ensure 10 digits, pad with leading 1 if needed (US numbers)
df['phone'] = df['phone'].apply(lambda x: x.zfill(10) if len(x) == 10 else ('1' + x).zfill(11) if len(x) == 10 else x)
# Impute missing phone numbers with mode
phone_mode = df['phone'].mode(dropna=True)
if not phone_mode.empty:
    df['phone'] = df['phone'].fillna(phone_mode.iloc[0])

# Clean zip codes (treat as text but standardize format)
df['zip'] = df['zip'].astype(str).str.replace(r'[^0-9]', '', regex=True)
df['zip'] = df['zip'].str.zfill(5)  # Ensure 5 digits
# Impute missing zip codes with mode
zip_mode = df['zip'].mode(dropna=True)
if not zip_mode.empty:
    df['zip'] = df['zip'].fillna(zip_mode.iloc[0])

# Clean state_average column (treat as categorical)
df['state_average'] = df['state_average'].str.strip()
state_avg_mode = df['state_average'].mode(dropna=True)
if not state_avg_mode.empty:
    df['state_average'] = df['state_average'].fillna(state_avg_mode.iloc[0])

# Final check for any remaining missing values in text columns
for col in text_columns + ['state_average']:
    if df[col].isna().any():
        mode_val = df[col].mode(dropna=True)
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])