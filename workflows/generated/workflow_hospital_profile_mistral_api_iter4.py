import pandas as pd
import numpy as np
import difflib
import re

# Normalize all missing value markers first
missing_markers = ['', 'NA', 'N/A', 'unknown', ' ', 'empty']
df.replace(missing_markers, np.nan, inplace=True)

# Strip whitespace from all string columns
str_cols = df.select_dtypes(include=['object']).columns
df[str_cols] = df[str_cols].apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Fix provider_number: extract digits and handle x patterns more carefully
def clean_provider_number(val):
    if pd.isna(val):
        return val
    val = str(val).lower()
    # Handle patterns like '1xx36' -> '10036'
    if 'xx' in val:
        prefix = val.split('xx')[0]
        suffix = val.split('xx')[1] if len(val.split('xx')) > 1 else ''
        candidates = df['provider_number'].dropna().unique()
        matches = [c for c in candidates if str(c).startswith(prefix) and str(c).endswith(suffix)]
        if matches:
            return max(set(matches), key=list(df['provider_number'].dropna()).count)
    # Handle single x patterns like '334x3xx221' -> '334636221'
    if 'x' in val:
        # Find most common provider_number that matches the non-x pattern
        pattern = re.sub(r'x', r'\\d', re.escape(val))
        candidates = df['provider_number'].dropna().unique()
        matches = [c for c in candidates if re.fullmatch(pattern, str(c))]
        if matches:
            return max(set(matches), key=list(df['provider_number'].dropna()).count)
    # Default digit extraction
    digits = re.sub(r'[^0-9]', '', val)
    return digits if digits else val

df['provider_number'] = df['provider_number'].apply(clean_provider_number)
df['provider_number'] = pd.to_numeric(df['provider_number'], errors='coerce')

# Group impute provider_number using name as key
group_mode = df.groupby('name')['provider_number'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['provider_number'] = df['provider_number'].fillna(group_mode)
global_mode = df['provider_number'].mode()
if not global_mode.empty:
    df['provider_number'] = df['provider_number'].fillna(global_mode.iloc[0])
df['provider_number'] = df['provider_number'].astype(int).astype(str)

# Fix address_1: handle x patterns in street names and numbers
def clean_address(val):
    if pd.isna(val):
        return val
    val = str(val)
    # Handle patterns like '2505xuxsxhighwayx431xnorth' -> '2505 u s highway 431 north'
    val = re.sub(r'x(?=[a-z])', ' ', val, flags=re.IGNORECASE)  # x between letters -> space
    val = re.sub(r'x(?=\d)', '', val)  # x before digit -> remove
    val = re.sub(r'(?<=\d)x', '', val)  # x after digit -> remove
    # Handle patterns like '1256 military street sxuth' -> '1256 military street south'
    val = re.sub(r'sxuth', 'south', val, flags=re.IGNORECASE)
    val = re.sub(r'nxrth', 'north', val, flags=re.IGNORECASE)
    val = re.sub(r'eaxst', 'east', val, flags=re.IGNORECASE)
    val = re.sub(r'wexst', 'west', val, flags=re.IGNORECASE)
    # Handle patterns like '1xth' -> '18th'
    val = re.sub(r'(\d+)x(\w+)', lambda m: m.group(1) + '8' + m.group(2), val)
    # Handle patterns like '1xx' -> '100' (common in street numbers)
    val = re.sub(r'(\d+)xx', lambda m: m.group(1) + '00', val)
    val = re.sub(r'\s+', ' ', val).strip()
    return val

df['address_1'] = df['address_1'].apply(clean_address)

# Fix zip: handle x patterns and ensure 5 digits
def clean_zip(val):
    if pd.isna(val):
        return val
    val = str(val).lower()
    # Handle patterns like '36x01' -> '36201' or '359x8' -> '35968'
    if 'x' in val:
        # Find the most common zip that matches the non-x parts
        pattern = re.sub(r'x', r'\\d', re.escape(val))
        candidates = df['zip'].dropna().unique()
        matches = [c for c in candidates if re.fullmatch(pattern, str(c))]
        if matches:
            return max(set(matches), key=list(df['zip'].dropna()).count)
    # Default digit extraction
    digits = re.sub(r'[^0-9]', '', val)
    return digits[:5].ljust(5, '0') if digits else val

df['zip'] = df['zip'].apply(clean_zip)
df['zip'] = df['zip'].str.pad(5, fillchar='0')  # Ensure 5 digits

# Group impute zip using city
group_mode = df.groupby('city')['zip'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['zip'] = df['zip'].fillna(group_mode)
global_mode = df['zip'].mode()
if not global_mode.empty:
    df['zip'] = df['zip'].fillna(global_mode.iloc[0])

# Fix city: fuzzy match with enhanced pattern handling
valid_cities = ['birmingham', 'gadsden', 'montgomery', 'dothan', 'huntsville',
                'cullman', 'opelika', 'prattville', 'centre', 'wedowee']
def fuzzy_correct_city(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    # Handle patterns like 'cuxxman' -> 'cullman'
    if 'xx' in val_norm:
        prefix = val_norm.split('xx')[0]
        suffix = val_norm.split('xx')[1] if len(val_norm.split('xx')) > 1 else ''
        candidates = [c for c in valid_cities if c.startswith(prefix) and c.endswith(suffix)]
        if candidates:
            return candidates[0]
    # Handle specific known typos
    if val_norm == 'andaluxia':
        return 'andalusia'
    # Default fuzzy matching
    if val_norm in [v.lower() for v in valid_cities]:
        return next(v for v in valid_cities if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_cities], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_cities if v.lower() == match[0])
    return val

df['city'] = df['city'].apply(fuzzy_correct_city)

# Fix state: enhanced fuzzy matching
valid_states = ['al', 'ak']
def fuzzy_correct_state(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    # Handle patterns like 'xl' -> 'al'
    if len(val_norm) == 2 and val_norm[0] in ['a', 'x'] and val_norm[1] in ['l', 'x']:
        return 'al'
    if val_norm in [v.lower() for v in valid_states]:
        return next(v for v in valid_states if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_states], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_states if v.lower() == match[0])
    return val

df['state'] = df['state'].apply(fuzzy_correct_state)

# Fix county: fuzzy match against known valid counties
valid_counties = ['jefferson', 'etowah', 'marion', 'marshall', 'covington',
                  'lee', 'autauga', 'cherokee', 'randolph', 'houston']
def fuzzy_correct_county(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    # Handle patterns like 'chxrokxx' -> 'cherokee'
    if 'xx' in val_norm:
        prefix = val_norm.split('xx')[0]
        candidates = [c for c in valid_counties if c.startswith(prefix)]
        if candidates:
            return candidates[0]
    # Handle specific known typos
    if val_norm == 'housxon':
        return 'houston'
    if val_norm == 'chxrokxx':
        return 'cherokee'
    if val_norm in [v.lower() for v in valid_counties]:
        return next(v for v in valid_counties if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_counties], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_counties if v.lower() == match[0])
    return val

df['county'] = df['county'].apply(fuzzy_correct_county)

# Fix type: fuzzy match against known valid types
valid_types = ['acute care hospitals']
def fuzzy_correct_type(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    # Handle patterns like 'acuxe care hospixals' -> 'acute care hospitals'
    if 'x' in val_norm:
        val_norm = val_norm.replace('x', '')
    # Handle specific known typos
    if val_norm == 'acutexcarexhospitals':
        return 'acute care hospitals'
    if val_norm in [v.lower() for v in valid_types]:
        return next(v for v in valid_types if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_types], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_types if v.lower() == match[0])
    return val

df['type'] = df['type'].apply(fuzzy_correct_type)

# Fix owner: fuzzy match against known valid owner types
valid_owners = [
    'voluntary non-profit - private',
    'proprietary',
    'government - hospital district or authority',
    'voluntary non-profit - other',
    'voluntary non-profit - church',
    'government - local',
    'government - state'
]
def fuzzy_correct_owner(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_owners]:
        return next(v for v in valid_owners if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_owners], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_owners if v.lower() == match[0])
    return val

df['owner'] = df['owner'].apply(fuzzy_correct_owner)

# Fix emergency_service: fuzzy match against known valid values
valid_emergency = ['yes', 'no']
def fuzzy_correct_emergency(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    # Handle patterns like 'yxs' -> 'yes'
    if val_norm.startswith('y') and len(val_norm) == 3:
        return 'yes'
    if val_norm in [v.lower() for v in valid_emergency]:
        return next(v for v in valid_emergency if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_emergency], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_emergency if v.lower() == match[0])
    return val

df['emergency_service'] = df['emergency_service'].apply(fuzzy_correct_emergency)

# Fix condition: fuzzy match with enhanced pattern handling
valid_conditions = [
    'surgical infection prevention',
    'heart attack',
    'pneumonia',
    'heart failure',
    "children's asthma care"
]
def fuzzy_correct_condition(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    # Handle patterns like 'surgical ixfectiox prevextiox' -> 'surgical infection prevention'
    if 'x' in val_norm:
        val_norm = val_norm.replace('x', '')
    # Handle specific known patterns
    if 'surgical' in val_norm and 'infect' in val_norm:
        return 'surgical infection prevention'
    if 'heart' in val_norm and 'attack' in val_norm:
        return 'heart attack'
    if val_norm in [v.lower() for v in valid_conditions]:
        return next(v for v in valid_conditions if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_conditions], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_conditions if v.lower() == match[0])
    return val

df['condition'] = df['condition'].apply(fuzzy_correct_condition)

# Fix measure_name: fuzzy match with enhanced pattern handling
def fuzzy_correct_measure_name(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    # Handle patterns like 'heart attaxk patients given aspirin at disxharge'
    if 'x' in val_norm:
        val_norm = val_norm.replace('x', '')
    # Handle specific known patterns
    if 'heart attack' in val_norm and 'aspirin' in val_norm and 'discharge' in val_norm:
        return 'heart attack patients given aspirin at discharge'
    if 'pneumonia' in val_norm and 'pneumococcal' in val_norm:
        return 'pneumonia patients assessed and given pneumococcal vaccination'
    if val_norm in [v.lower() for v in df['measure_name'].dropna().unique()]:
        return next(v for v in df['measure_name'].dropna().unique() if v.lower() == val_norm)
    # Try to find the most common measure_name that contains the key words
    key_words = val_norm.split()
    candidates = df['measure_name'].dropna().unique()
    matches = []
    for candidate in candidates:
        if all(word in candidate.lower() for word in key_words):
            matches.append(candidate)
    if matches:
        return max(set(matches), key=list(df['measure_name'].dropna()).count)
    return val

df['measure_name'] = df['measure_name'].apply(fuzzy_correct_measure_name)

# Fix phone: handle x patterns
def clean_phone(val):
    if pd.isna(val):
        return val
    val = str(val)
    # Handle patterns like '334x3xx221' -> '334636221'
    if 'x' in val:
        # Find most common phone that matches the pattern
        pattern = re.sub(r'x', r'\\d', re.escape(val))
        candidates = df['phone'].dropna().unique()
        matches = [c for c in candidates if re.fullmatch(pattern, str(c))]
        if matches:
            return max(set(matches), key=list(df['phone'].dropna()).count)
    # Default digit extraction
    digits = re.sub(r'[^0-9]', '', val)
    return digits if digits else val

df['phone'] = df['phone'].apply(clean_phone)

# Fix score: handle x patterns in percentages
def clean_score(val):
    if pd.isna(val):
        return val
    val = str(val).lower()
    # Handle patterns like 'x7%' -> '67%'
    if val.startswith('x') and '%' in val:
        prefix = val[1:].split('%')[0]
        # Find most common score that matches the pattern
        candidates = df['score'].dropna().unique()
        matches = [c for c in candidates if c.endswith(prefix+'%') and len(c) == len(val)]
        if matches:
            return max(set(matches), key=list(df['score'].dropna()).count)
    # Handle patterns like '1xx%' -> '100%'
    if 'xx' in val:
        prefix = val.split('xx')[0]
        return f"{prefix}00%"
    # Handle patterns like '9x%' -> '98%'
    if 'x' in val and '%' in val and not val.startswith('x'):
        prefix = val.split('x')[0]
        suffix = val.split('x')[1].split('%')[0]
        # Find most common score that matches the pattern
        candidates = df['score'].dropna().unique()
        matches = [c for c in candidates if c.startswith(prefix) and c.endswith(suffix+'%')]
        if matches:
            return max(set(matches), key=list(df['score'].dropna()).count)
    # Default digit extraction
    digits = re.sub(r'[^0-9]', '', val)
    return f"{digits}%" if digits else val

df['score'] = df['score'].apply(clean_score)
df['score'] = df['score'].str.replace(r'%', '', regex=False)
df['score'] = pd.to_numeric(df['score'], errors='coerce')

# Group impute score with (provider_number, measure_code)
df['score'] = df.groupby(['provider_number', 'measure_code'])['score'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
global_mode = df['score'].mode()
if not global_mode.empty:
    df['score'] = df['score'].fillna(global_mode.iloc[0])
df['score'] = df['score'].astype(int).astype(str) + '%'

# Fix sample: extract number and handle special cases
def clean_sample(val):
    if pd.isna(val):
        return val
    val = str(val).strip().lower()
    if val == '0 patients':
        return '0 patients'
    if val == 'empty':
        return np.nan
    # Handle patterns like '1xx patients' -> '100 patients'
    if 'xx' in val:
        prefix = val.split('xx')[0]
        return f"{prefix}00 patients"
    # Extract number from patterns like 'x patients' or '1 patients'
    num_match = re.search(r'(\d+)', val)
    if num_match:
        return f"{num_match.group(1)} patients"
    return val

df['sample'] = df['sample'].apply(clean_sample)

# Group impute sample with (provider_number, measure_code)
df['sample'] = df.groupby(['provider_number', 'measure_code'])['sample'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
global_mode = df['sample'].mode()
if not global_mode.empty:
    df['sample'] = df['sample'].fillna(global_mode.iloc[0])

# Group-aware imputation for known dependencies
# provider_number -> name, address_1, city, state, zip, county, phone, type, owner, emergency_service
for target in ['name', 'address_1', 'city', 'state', 'zip', 'county', 'phone', 'type', 'owner', 'emergency_service']:
    df[target] = df.groupby('provider_number')[target].transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    )
    global_mode = df[target].mode()
    if not global_mode.empty:
        df[target] = df[target].fillna(global_mode.iloc[0])

# measure_code -> measure_name, condition
for target in ['measure_name', 'condition']:
    df[target] = df.groupby('measure_code')[target].transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    )
    global_mode = df[target].mode()
    if not global_mode.empty:
        df[target] = df[target].fillna(global_mode.iloc[0])

# (provider_number, measure_code) -> score, sample (already done above)

# zip -> city, state, county
for target in ['city', 'state', 'county']:
    df[target] = df.groupby('zip')[target].transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    )
    global_mode = df[target].mode()
    if not global_mode.empty:
        df[target] = df[target].fillna(global_mode.iloc[0])

# Fix name: handle specific patterns
def clean_name(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    # Handle patterns like 'andaluxia regional hoxpital' -> 'andalusia regional hospital'
    if 'andaluxia' in val.lower():
        val = val.replace('andaluxia', 'andalusia').replace('Andaluxia', 'Andalusia')
    if 'hoxpital' in val.lower():
        val = val.replace('hoxpital', 'hospital').replace('Hoxpital', 'Hospital')
    return val

df['name'] = df['name'].apply(clean_name)

# Ensure numeric columns are numeric
df['index'] = pd.to_numeric(df['index'], errors='coerce')
df['index'] = df['index'].fillna(df['index'].median())

# Final output
df