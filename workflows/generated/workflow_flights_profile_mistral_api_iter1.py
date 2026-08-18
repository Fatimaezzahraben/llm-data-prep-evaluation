import pandas as pd
import numpy as np
import difflib
import re

# Disguised missing values handling
missing_markers = ["NA", "N/A", "unknown", "", " "]
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].replace(missing_markers, np.nan)
        df[col] = df[col].str.strip().replace('', np.nan)

# tuple_id - numeric, no missing, no action needed (already clean)

# src - categorical, no missing, but check for typos
src_valid = ['helloflight', 'boston', 'airtravelcenter', 'flightview', 'panynj']
def fuzzy_correct_src(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in src_valid]:
        return next(v for v in src_valid if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in src_valid], n=1, cutoff=0.6)
    if match:
        return next(v for v in src_valid if v.lower() == match[0])
    return val
df['src'] = df['src'].apply(fuzzy_correct_src)

# flight - categorical, no missing, but check for format consistency
# Standardize format: AA-123-XXX-XXX (remove any extra spaces or inconsistent separators)
df['flight'] = df['flight'].str.replace(r'[^A-Za-z0-9-]', '', regex=True).str.upper()
df['flight'] = df['flight'].str.replace(r'^([A-Z]+)(\d+)([A-Z]+)([A-Z]+)$', r'\1-\2-\3-\4', regex=True)
df['flight'] = df['flight'].str.replace(r'^([A-Z]+)(\d+)([A-Z]+)$', r'\1-\2-\3', regex=True)

# Time columns: sched_dep_time, act_dep_time, sched_arr_time, act_arr_time
# Treat as categorical (do not convert to datetime), but clean format and impute missing

# Clean time format: standardize to "h:mm a.m./p.m." with consistent spacing and dots
def clean_time(val):
    if pd.isna(val):
        return val
    val = str(val).strip().lower()
    # Handle cases like "6:00aDec 1" -> "6:00 a.m."
    val = re.sub(r'(\d+:\d+)\s*[ap]\.?m?\.?.*', r'\1', val, flags=re.IGNORECASE)
    val = re.sub(r'(\d+:\d+)\s*', r'\1 ', val)
    val = re.sub(r'(\d+:\d+)\s*([ap])\.?m?\.?', r'\1 \2.m.', val, flags=re.IGNORECASE)
    val = val.replace('a.m.', 'a.m.').replace('p.m.', 'p.m.')
    # Reconstruct full format
    if 'a.m.' not in val and 'p.m.' not in val:
        if 'a' in val:
            val = val.replace('a', 'a.m.')
        elif 'p' in val:
            val = val.replace('p', 'p.m.')
    return val.strip()

time_cols = ['sched_dep_time', 'act_dep_time', 'sched_arr_time', 'act_arr_time']
for col in time_cols:
    df[col] = df[col].apply(clean_time)
    # Impute missing with mode (per column)
    mode_val = df[col].mode(dropna=True)
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val.iloc[0])

# Group-aware imputation for time columns using flight as key (if flight determines time)
for col in time_cols:
    group_mode = df.groupby('flight')[col].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    df[col] = df[col].fillna(group_mode)
    # Fallback to global mode if group mode didn't fill all
    global_mode = df[col].mode(dropna=True)
    if not global_mode.empty:
        df[col] = df[col].fillna(global_mode.iloc[0])

# Final output
df