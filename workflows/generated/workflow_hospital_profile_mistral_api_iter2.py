import pandas as pd
import numpy as np
import difflib
import re

# Clean address_2 and address_3 - replace 'empty' with NaN
df['address_2'] = df['address_2'].replace(['empty', ''], np.nan)
df['address_3'] = df['address_3'].replace(['empty', ''], np.nan)

# Clean state column - replace 'xl', 'ax' with 'al' (most common)
df['state'] = df['state'].replace(['xl', 'ax'], 'al')

# Clean type column - fuzzy matching for typos
valid_types = ['acute care hospitals']
def fuzzy_type_correct(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_types]:
        return next(v for v in valid_types if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_types], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_types if v.lower() == match[0])
    return val
df['type'] = df['type'].apply(fuzzy_type_correct)

# Clean emergency_service - replace 'yxs', 'yex', 'xes' with 'yes'
df['emergency_service'] = df['emergency_service'].replace(['yxs', 'yex', 'xes'], 'yes')

# Clean county - fuzzy matching for typos
valid_counties = ['jefferson', 'etowah', 'marion', 'marshall', 'covington', 'madison', 'chickasaw']
def fuzzy_county_correct(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_counties]:
        return next(v for v in valid_counties if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_counties], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_counties if v.lower() == match[0])
    return val
df['county'] = df['county'].apply(fuzzy_county_correct)

# Clean provider_number - replace 'x' with '0' and handle patterns like '1xx36'
def clean_provider_number(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    # Replace 'x' with '0' and handle patterns like '1xx36'
    val = val.replace('x', '0')
    # Handle cases like '1xx36' -> '10036'
    if len(val) > 4 and 'x' in val.replace('0', ''):
        parts = re.split(r'(\d+)', val)
        num_parts = [p for p in parts if p.isdigit()]
        if len(num_parts) >= 2:
            first_part = num_parts[0]
            second_part = num_parts[1]
            if len(first_part) == 1 and len(second_part) == 2:
                return f"{first_part}0{second_part}"
    return val
df['provider_number'] = df['provider_number'].apply(clean_provider_number)

# Clean score column - regex to extract and reconstruct percentage
def clean_score(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    # Extract digits and % sign
    matches = re.findall(r'(\d+)%?', val)
    if matches:
        digits = [int(m) for m in matches if m.isdigit()]
        if len(digits) == 1:
            return f"{digits[0]}%"
        elif len(digits) == 2:
            # Handle cases like '1xx%' -> '100%'
            if 'x' in val:
                first_digit = digits[0]
                if first_digit == 1 and len(str(digits[0])) == 1:
                    return "100%"
            return f"{digits[0]}{digits[1]}%"
        elif len(digits) == 0:
            # Handle cases like 'x7%' -> '7%'
            if 'x' in val:
                return val.replace('x', '0').replace('%', '').strip() + '%'
    return val
df['score'] = df['score'].apply(clean_score)

# Clean sample column - regex to extract number of patients
def clean_sample(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    # Extract digits and 'patients' or 'empty'
    match = re.search(r'(\d+)\s*patients?', val)
    if match:
        return f"{match.group(1)} patients"
    if val.lower() == 'empty':
        return '0 patients'
    return val
df['sample'] = df['sample'].apply(clean_sample)

# Clean state_average - replace 'alx' with 'al_'
df['state_average'] = df['state_average'].str.replace('alx', 'al_')

# Use known grouping keys for EXACT dependencies
# provider_number -> name, address_1, city, state, zip, county, phone, type, owner, emergency_service
df['name'] = df.groupby('provider_number')['name'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['address_1'] = df.groupby('provider_number')['address_1'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['city'] = df.groupby('provider_number')['city'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['state'] = df.groupby('provider_number')['state'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['zip'] = df.groupby('provider_number')['zip'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['county'] = df.groupby('provider_number')['county'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['phone'] = df.groupby('provider_number')['phone'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['type'] = df.groupby('provider_number')['type'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['owner'] = df.groupby('provider_number')['owner'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['emergency_service'] = df.groupby('provider_number')['emergency_service'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)

# measure_code -> measure_name, condition
df['measure_name'] = df.groupby('measure_code')['measure_name'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['condition'] = df.groupby('measure_code')['condition'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)

# zip -> city, state, county
df['city'] = df.groupby('zip')['city'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['state'] = df.groupby('zip')['state'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['county'] = df.groupby('zip')['county'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)

# Clean measure_code - replace 'x' with 'h' or 'c' based on context
def clean_measure_code(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    if 'scip-vtx' in val:
        return val.replace('vtx', 'vte')
    elif 'xf-' in val:
        return val.replace('xf-', 'hf-')
    elif 'xax-' in val:
        return val.replace('xax-', 'cac-')
    elif 'scip-' in val:
        return val.replace('scip-', 'scip-')
    return val
df['measure_code'] = df['measure_code'].apply(clean_measure_code)

# Fill missing values with appropriate statistics
categorical_cols = ['provider_number', 'name', 'address_1', 'city', 'state', 'zip', 'county', 'phone', 'type', 'owner', 'emergency_service', 'condition', 'measure_code', 'measure_name', 'score', 'sample', 'state_average']
for col in categorical_cols:
    if df[col].dtype == 'object':
        _mode = df[col].mode(dropna=True)
        fill_value = _mode.iloc[0] if not _mode.empty else np.nan
        df[col] = df[col].fillna(fill_value)

# Clean county column again after groupby to ensure consistency
df['county'] = df['county'].str.strip().str.lower()
df['county'] = df['county'].replace(['chxrokxx'], 'chickasaw')

# Final clean for score and sample columns
# Ensure score is a valid percentage
df['score'] = df['score'].str.replace('1xx%', '100%')
df['score'] = df['score'].str.replace('x7%', '7%')
df['score'] = df['score'].str.replace('x9%', '9%')
df['score'] = df['score'].str.replace('x8%', '8%')
df['score'] = df['score'].str.replace('x0%', '0%')

# Ensure sample is a valid number of patients
df['sample'] = df['sample'].str.replace('patiexts', 'patients')
df['sample'] = df['sample'].str.replace('empty', '0 patients')

# Clean provider_number to handle patterns like '1xx36' more robustly
def clean_provider_number_final(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    # Replace 'x' with '0' and handle patterns like '1xx36'
    val = val.replace('x', '0')
    # Handle cases like '1xx36' -> '10036'
    if len(val) > 4 and '0' in val:
        parts = re.split(r'(\d+)', val)
        num_parts = [p for p in parts if p.isdigit()]
        if len(num_parts) >= 2:
            first_part = num_parts[0]
            second_part = num_parts[1]
            if len(first_part) == 1 and len(second_part) == 2:
                return f"{first_part}0{second_part}"
    return val
df['provider_number'] = df['provider_number'].apply(clean_provider_number_final)

# Clean measure_code to handle more patterns
def clean_measure_code_final(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    if 'scip-vtx' in val:
        return val.replace('vtx', 'vte')
    elif 'xf-' in val:
        return val.replace('xf-', 'hf-')
    elif 'xax-' in val:
        return val.replace('xax-', 'cac-')
    elif 'scip-' in val:
        return val.replace('scip-', 'scip-')
    elif 'ami-7a' in val:
        return val.replace('ami-7a', 'ami-7')
    return val
df['measure_code'] = df['measure_code'].apply(clean_measure_code_final)