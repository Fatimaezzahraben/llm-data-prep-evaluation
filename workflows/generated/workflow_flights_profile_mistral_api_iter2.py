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

# flight - categorical, no missing, but standardize format
df['flight'] = df['flight'].str.replace(r'[^A-Za-z0-9-]', '', regex=True).str.upper()
df['flight'] = df['flight'].str.replace(r'^([A-Z]+)(\d+)([A-Z]+)([A-Z]+)$', r'\1-\2-\3-\4', regex=True)
df['flight'] = df['flight'].str.replace(r'^([A-Z]+)(\d+)([A-Z]+)$', r'\1-\2-\3', regex=True)

# Time columns: treat as categorical, clean format, and impute missing
time_cols = ['sched_dep_time', 'act_dep_time', 'sched_arr_time', 'act_arr_time']

# First pass: clean time format while preserving a.m./p.m.
def clean_time_format(val):
    if pd.isna(val):
        return val
    val = str(val).strip().lower()

    # Extract time and period
    time_part = re.search(r'(\d{1,2}:\d{2})\s*([ap]\.?m?\.?)', val, re.IGNORECASE)
    if time_part:
        time = time_part.group(1)
        period = time_part.group(2).replace('.', '').strip()
        if period.startswith('a'):
            period = 'a.m.'
        elif period.startswith('p'):
            period = 'p.m.'
        return f"{time} {period}"

    # Handle cases like "6:00aDec 1" -> extract "6:00" and infer period from context
    time_match = re.search(r'(\d{1,2}:\d{2})', val)
    if time_match:
        time = time_match.group(1)
        # Try to infer period from surrounding text (not perfect but better than nothing)
        if 'a' in val.lower() or 'am' in val.lower():
            return f"{time} a.m."
        elif 'p' in val.lower() or 'pm' in val.lower():
            return f"{time} p.m."
        # If no period found, return just the time (will be handled in group imputation)
        return time
    return val

for col in time_cols:
    df[col] = df[col].apply(clean_time_format)

# Second pass: group-aware imputation using flight as key
# We'll create a mapping of flight -> most common time for each time column
flight_time_map = {}
for col in time_cols:
    flight_time_map[col] = df.groupby('flight')[col].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).to_dict()

# Apply the mapping to missing values
for col in time_cols:
    # First fill with flight-specific time if available
    df[col] = df[col].fillna(df['flight'].map(flight_time_map[col]))
    # Then fill remaining with global mode
    global_mode = df[col].mode()
    if not global_mode.empty:
        df[col] = df[col].fillna(global_mode[0])

# Third pass: ensure all times have a.m./p.m. by using the most common period for each hour
for col in time_cols:
    # For times missing period, assign based on hour
    missing_period = df[col].str.contains(r'^\d{1,2}:\d{2}$', na=False)
    if missing_period.any():
        # Extract hour from time
        df['temp_hour'] = df[col].str.extract(r'^(\d{1,2}):\d{2}$')[0].astype(float)
        # Determine most common period for each hour
        hour_period = df[~missing_period].copy()
        hour_period['hour'] = hour_period[col].str.extract(r'^(\d{1,2}):\d{2}')[0].astype(float)
        period_map = hour_period.groupby('hour')[col].agg(lambda x: x.str.extract(r'([ap]\.m\.)')[0].mode()[0])

        # Apply period to missing values
        for hour, period in period_map.items():
            mask = missing_period & (df['temp_hour'] == hour)
            df.loc[mask, col] = df.loc[mask, col] + f" {period}"

        # Clean up temp column
        df.drop('temp_hour', axis=1, inplace=True)

# Final output
df