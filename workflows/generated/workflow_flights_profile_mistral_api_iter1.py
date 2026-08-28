import pandas as pd
import numpy as np
import re
import difflib

# tuple_id: numeric, 0% missing, no issues detected
# No cleaning needed beyond ensuring numeric dtype
df['tuple_id'] = pd.to_numeric(df['tuple_id'], errors='coerce')
df['tuple_id'] = df['tuple_id'].fillna(df['tuple_id'].median())

# src: categorical, 0% missing, but check for typos/near-duplicates
# Profile shows 38 unique values - use fuzzy matching for low-cardinality correction
valid_src = ['helloflight', 'boston', 'airtravelcenter', 'flightview', 'panynj']
def fuzzy_correct_src(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_src]:
        return next(v for v in valid_src if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_src], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_src if v.lower() == match[0])
    return val
df['src'] = df['src'].apply(fuzzy_correct_src)

# flight: categorical, 0% missing, no obvious typos in sample
# No cleaning needed beyond basic whitespace/case normalization
df['flight'] = df['flight'].astype(str).str.strip().str.upper()

# sched_dep_time: categorical/text, 33% missing, many near-duplicates
# First normalize disguised missing values
disguised_missing = df['sched_dep_time'].astype(str).str.strip().str.lower().isin(['na', 'n/a', 'unknown', ''])
df.loc[disguised_missing, 'sched_dep_time'] = np.nan

# Apply EXACT dependency: flight -> sched_dep_time
df['sched_dep_time'] = df.groupby('flight')['sched_dep_time'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)

# Global fallback for any remaining missing values
_mode = df['sched_dep_time'].mode(dropna=True)
fill_value = _mode.iloc[0] if not _mode.empty else '7:10 a.m.'
df['sched_dep_time'] = df['sched_dep_time'].fillna(fill_value)

# act_dep_time: categorical/text, 15.82% missing
# First normalize disguised missing values
disguised_missing = df['act_dep_time'].astype(str).str.strip().str.lower().isin(['na', 'n/a', 'unknown', ''])
df.loc[disguised_missing, 'act_dep_time'] = np.nan

# Clean time format issues seen in sample (e.g., '5:58aDec 1')
def clean_time(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    # Handle cases like '5:58aDec 1' -> '5:58 a.m.'
    if re.match(r'^\d{1,2}:\d{2}[ap]m?\s*[a-zA-Z]', val, re.IGNORECASE):
        val = re.sub(r'([ap]m?)\s*[a-zA-Z].*', r'\1', val, flags=re.IGNORECASE)
        val = val.replace('a', ' a.m.').replace('p', ' p.m.')
    # Handle 'Delayed' -> NaN
    if val.lower() == 'delayed':
        return np.nan
    return val

df['act_dep_time'] = df['act_dep_time'].apply(clean_time)

# Global mode imputation
_mode = df['act_dep_time'].mode(dropna=True)
fill_value = _mode.iloc[0] if not _mode.empty else '7:22 a.m.'
df['act_dep_time'] = df['act_dep_time'].fillna(fill_value)

# sched_arr_time: categorical/text, 32.41% missing
# First normalize disguised missing values
disguised_missing = df['sched_arr_time'].astype(str).str.strip().str.lower().isin(['na', 'n/a', 'unknown', ''])
df.loc[disguised_missing, 'sched_arr_time'] = np.nan

# Apply EXACT dependency: flight -> sched_arr_time
df['sched_arr_time'] = df.groupby('flight')['sched_arr_time'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)

# Global fallback for any remaining missing values
_mode = df['sched_arr_time'].mode(dropna=True)
fill_value = _mode.iloc[0] if not _mode.empty else '2:35 p.m.'
df['sched_arr_time'] = df['sched_arr_time'].fillna(fill_value)

# act_arr_time: categorical/text, 16.08% missing
# First normalize disguised missing values
disguised_missing = df['act_arr_time'].astype(str).str.strip().str.lower().isin(['na', 'n/a', 'unknown', ''])
df.loc[disguised_missing, 'act_arr_time'] = np.nan

# Clean time format issues seen in sample (e.g., '9:05aDec 1')
df['act_arr_time'] = df['act_arr_time'].apply(clean_time)

# Global mode imputation
_mode = df['act_arr_time'].mode(dropna=True)
fill_value = _mode.iloc[0] if not _mode.empty else '7:10 p.m.'
df['act_arr_time'] = df['act_arr_time'].fillna(fill_value)

# Final type consistency
for col in ['sched_dep_time', 'act_dep_time', 'sched_arr_time', 'act_arr_time']:
    df[col] = df[col].astype(str).str.strip()
    # Ensure consistent a.m./p.m. formatting
    df[col] = df[col].str.replace(r'\.m\.', 'm', regex=True)
    df[col] = df[col].str.replace(r'(\d)([ap]m)', r'\1 \2', regex=True)
    df[col] = df[col].str.replace(r'(\d)([ap])', r'\1 \2.m.', regex=True)
    df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()

df