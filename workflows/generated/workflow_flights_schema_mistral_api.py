import pandas as pd
import numpy as np
import re

# Standardize text columns - fix typos and inconsistent categories
text_cols = ['src', 'flight']
for col in text_cols:
    # Strip whitespace and normalize case where appropriate
    df[col] = df[col].astype(str).str.strip()
    # For src column, standardize common variations
    if col == 'src':
        src_mapping = {
            'aa': 'AA',
            'american': 'AA',
            'helloflight': 'HelloFlight',
            'boston': 'Boston',
            'lax': 'LAX',
            'jfk': 'JFK',
            'mia': 'MIA',
            'ord': 'ORD',
            'phx': 'PHX',
            'iah': 'IAH',
            'mco': 'MCO',
            'dfw': 'DFW'
        }
        df[col] = df[col].replace(src_mapping)
    # For flight column, standardize format (keep original if already correct)
    if col == 'flight':
        # Standardize airline codes to uppercase
        df[col] = df[col].str.upper()
        # Fix common flight number formats
        df[col] = df[col].str.replace(r'^([A-Z]{2})(\d)', r'\1-\2', regex=True)
        df[col] = df[col].str.replace(r'([A-Z]{2})-(\d{1,4})([A-Z]{3})', r'\1-\2-\3', regex=True)
        df[col] = df[col].str.replace(r'([A-Z]{3})-([A-Z]{3})$', r'\1-\2', regex=True)
        # Ensure proper format: AA-1234-XXX-XXX
        df[col] = df[col].str.replace(r'^([A-Z]{2})-(\d{1,4})-([A-Z]{3})-([A-Z]{3})$', r'\1-\2-\3-\4')

# Handle missing values in text columns
for col in text_cols:
    # Identify missing values (including empty strings, NA, etc.)
    missing_mask = df[col].isna() | (df[col].astype(str).str.strip() == '') | (df[col].astype(str).str.lower() == 'na') | (df[col].astype(str).str.lower() == 'n/a')
    # Compute mode for imputation
    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
    # Impute missing values
    df.loc[missing_mask, col] = mode_val

# Clean time columns (treat as categorical/text)
time_cols = ['sched_dep_time', 'act_dep_time', 'sched_arr_time', 'act_arr_time']
for col in time_cols:
    # Convert to string and strip whitespace
    df[col] = df[col].astype(str).str.strip()

    # Identify invalid time formats
    valid_time_pattern = re.compile(r'^\d{1,2}:\d{2}\s*[ap]\.?m\.?$', re.IGNORECASE)
    invalid_mask = ~df[col].str.match(valid_time_pattern, na=False)

    # Replace invalid times with mode
    mode_val = df[col].mode()[0] if not df[col].mode().empty else '12:00 p.m.'
    df.loc[invalid_mask, col] = mode_val

    # Handle missing values (including empty strings, NA, etc.)
    missing_mask = df[col].isna() | (df[col] == '') | (df[col].str.lower() == 'na') | (df[col].str.lower() == 'n/a')
    df.loc[missing_mask, col] = mode_val

# Clean numeric column (tuple_id)
if 'tuple_id' in df.columns:
    # Convert to numeric, coercing errors to NaN
    df['tuple_id'] = pd.to_numeric(df['tuple_id'].astype(str).str.replace(r'[^0-9\-]', '', regex=True), errors='coerce')

    # Handle missing values with median
    median_val = df['tuple_id'].median()
    df['tuple_id'] = df['tuple_id'].fillna(median_val)

    # Detect and handle outliers (using IQR method)
    Q1 = df['tuple_id'].quantile(0.25)
    Q3 = df['tuple_id'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap outliers
    df['tuple_id'] = np.where(df['tuple_id'] < lower_bound, lower_bound, df['tuple_id'])
    df['tuple_id'] = np.where(df['tuple_id'] > upper_bound, upper_bound, df['tuple_id'])

# Group-aware imputation for flight-related columns
# For time columns, impute within flight groups where possible
for col in time_cols:
    # First try to impute within flight groups
    if 'flight' in df.columns:
        # Compute group mode
        group_modes = df.groupby('flight')[col].transform(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        )
        # Fill missing values with group mode
        missing_mask = df[col].isna() | (df[col] == '') | (df[col].str.lower() == 'na') | (df[col].str.lower() == 'n/a')
        df.loc[missing_mask, col] = group_modes[missing_mask]

        # For any remaining missing values, use global mode
        global_mode = df[col].mode()[0] if not df[col].mode().empty else '12:00 p.m.'
        df[col] = df[col].fillna(global_mode)

# Final check for any remaining missing values (shouldn't be any at this point)
for col in df.columns:
    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')