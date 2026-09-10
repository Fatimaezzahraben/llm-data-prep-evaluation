import pandas as pd
import numpy as np
import difflib
import re

# Clean address_2 and address_3 - replace 'empty' with NaN
df['address_2'] = df['address_2'].replace('empty', np.nan)
df['address_3'] = df['address_3'].replace('empty', np.nan)

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
valid_counties = ['jefferson', 'etowah', 'marion', 'marshall', 'covington']
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

# Clean score column - regex to extract percentage
def clean_score(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    # Extract digits and % sign
    match = re.search(r'(\d+)%?', val)
    if match:
        num = match.group(1)
        # Handle cases like '1xx%' -> '100%'
        if len(num) > 2 and 'x' in num:
            num = num.replace('x', '0')
        return f"{num}%"
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
        return 'empty'
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

# Fill missing values with appropriate statistics
# address_2 and address_3 are already handled by replacing 'empty' with NaN
# Fill missing categorical values with mode
categorical_cols = ['provider_number', 'name', 'address_1', 'city', 'state', 'zip', 'county', 'phone', 'type', 'owner', 'emergency_service', 'condition', 'measure_code', 'measure_name', 'score', 'sample', 'state_average']
for col in categorical_cols:
    if df[col].dtype == 'object':
        _mode = df[col].mode(dropna=True)
        fill_value = _mode.iloc[0] if not _mode.empty else np.nan
        df[col] = df[col].fillna(fill_value)

# Fill missing numeric values with median (though none are numeric in this dataset)
# No numeric columns to clean in this dataset

# Final cleaning for score and sample columns
# Fill missing scores with mode of valid scores
score_mode = df['score'].mode(dropna=True)
if not score_mode.empty:
    df['score'] = df['score'].fillna(score_mode.iloc[0])

# Fill missing samples with mode of valid samples
sample_mode = df['sample'].mode(dropna=True)
if not sample_mode.empty:
    df['sample'] = df['sample'].fillna(sample_mode.iloc[0])

# Clean county column again after groupby to ensure consistency
df['county'] = df['county'].str.strip().str.lower()
df['county'] = df['county'].replace(['chxrokxx'], 'chickasaw')

# Final clean for score and sample columns
# Ensure score is a valid percentage
df['score'] = df['score'].str.replace('1xx%', '100%')
df['score'] = df['score'].str.replace('x7%', '7%')
df['score'] = df['score'].str.replace('x9%', '9%')
df['score'] = df['score'].str.replace('x8%', '8%')

# Ensure sample is a valid number of patients
df['sample'] = df['sample'].str.replace('empty', '0 patients')
df['sample'] = df['sample'].str.replace('0 patients', '0 patients')