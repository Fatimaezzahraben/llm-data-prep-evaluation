import pandas as pd
import numpy as np
import re

# Helper function to detect disguised missing values
def is_disguised_missing(val):
    if pd.isna(val):
        return True
    val = str(val).strip().lower()
    return val in ['', 'na', 'n/a', 'unknown', 'empty']

# 1. Handle missing values (real and disguised) for all columns
for col in df.columns:
    # Convert all values to string first to handle disguised missing
    df[col] = df[col].astype(str).replace('nan', np.nan)

    # Identify missing values (real NaN or disguised)
    missing_mask = df[col].apply(is_disguised_missing)

    if missing_mask.any():
        if pd.api.types.is_numeric_dtype(df[col]):
            # For numeric columns, convert to numeric first
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            # For categorical columns, use mode
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].replace('', np.nan).fillna(mode_val.iloc[0])

# 2. Clean specific columns with high cardinality or known issues

# address_2 and address_3 - all empty, fill with empty string
df['address_2'] = df['address_2'].replace('empty', '')
df['address_3'] = df['address_3'].replace('empty', '')

# type - harmonize near-duplicate categories
type_mapping = {
    'acuxe care hospixals': 'acute care hospitals',
    'acutexcarexhospitals': 'acute care hospitals',
    'acute care hosxitals': 'acute care hospitals',
    'acutx carx hospitals': 'acute care hospitals'
}
df['type'] = df['type'].replace(type_mapping)

# emergency_service - harmonize near-duplicate categories
emergency_mapping = {
    'yxs': 'yes',
    'yex': 'yes',
    'xes': 'yes'
}
df['emergency_service'] = df['emergency_service'].replace(emergency_mapping)

# score - handle empty strings and ensure consistent format
df['score'] = df['score'].replace('empty', np.nan)
score_mode = df['score'].mode()
if not score_mode.empty:
    df['score'] = df['score'].fillna(score_mode.iloc[0])

# sample - handle empty strings
df['sample'] = df['sample'].replace('empty', np.nan)
sample_mode = df['sample'].mode()
if not sample_mode.empty:
    df['sample'] = df['sample'].fillna(sample_mode.iloc[0])

# 3. Ensure numeric columns contain only numeric values
numeric_cols = ['index']
for col in numeric_cols:
    # Clean non-numeric characters
    df[col] = df[col].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Cap outliers (using 1.5*IQR rule)
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower_bound, upper_bound)
    # Fill any remaining NaN with median
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# 4. Clean categorical columns with high cardinality using group-aware imputation
# For provider_number (high cardinality) - use name as grouping key
if 'provider_number' in df.columns and 'name' in df.columns:
    # First fill with group mode
    df['provider_number'] = df.groupby('name')['provider_number'].transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    )
    # Then fill remaining with global mode
    global_mode = df['provider_number'].mode()
    if not global_mode.empty:
        df['provider_number'] = df['provider_number'].fillna(global_mode.iloc[0])

# For measure_code (high cardinality) - use measure_name as grouping key
if 'measure_code' in df.columns and 'measure_name' in df.columns:
    # First fill with group mode
    df['measure_code'] = df.groupby('measure_name')['measure_code'].transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    )
    # Then fill remaining with global mode
    global_mode = df['measure_code'].mode()
    if not global_mode.empty:
        df['measure_code'] = df['measure_code'].fillna(global_mode.iloc[0])

# 5. Clean state column - harmonize invalid state codes
state_mapping = {
    'xl': 'al',  # Assuming 'xl' is a typo for 'al'
    'ax': 'al'   # Assuming 'ax' is a typo for 'al'
}
df['state'] = df['state'].replace(state_mapping)

# 6. Clean phone numbers - ensure consistent format
df['phone'] = df['phone'].astype(str).str.replace(r'[^0-9]', '', regex=True)
# Pad with leading zeros if needed to make 10 digits
df['phone'] = df['phone'].str.zfill(10)
# If still not 10 digits, fill with mode
phone_mode = df['phone'].mode()
if not phone_mode.empty:
    df['phone'] = df['phone'].where(df['phone'].str.len() == 10, phone_mode.iloc[0])

# 7. Clean zip codes - ensure consistent format
df['zip'] = df['zip'].astype(str).str.replace(r'[^0-9]', '', regex=True)
# Pad with leading zeros if needed to make 5 digits
df['zip'] = df['zip'].str.zfill(5)
# If still not 5 digits, fill with mode
zip_mode = df['zip'].mode()
if not zip_mode.empty:
    df['zip'] = df['zip'].where(df['zip'].str.len() == 5, zip_mode.iloc[0])

# 8. Clean text columns - strip whitespace and normalize case where appropriate
text_cols = ['name', 'address_1', 'city', 'county', 'owner', 'condition', 'measure_name']
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()
    # Only normalize case for columns where it makes sense (not for proper nouns)
    if col not in ['name', 'address_1', 'city', 'county']:
        df[col] = df[col].str.lower()

# 9. Clean score column - ensure it's in percentage format
df['score'] = df['score'].astype(str).str.replace(r'[^0-9%]', '', regex=True)
# If no % sign, add it
df['score'] = df['score'].where(df['score'].str.contains('%'), df['score'] + '%')
# If still invalid, fill with mode
score_mode = df['score'].mode()
if not score_mode.empty:
    df['score'] = df['score'].where(df['score'].str.match(r'^\d{1,3}%$'), score_mode.iloc[0])

# Final check - ensure all columns have no disguised missing values
for col in df.columns:
    df[col] = df[col].replace(['', 'na', 'n/a', 'unknown', 'empty'], np.nan)
    if pd.api.types.is_numeric_dtype(df[col]):
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    else:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])