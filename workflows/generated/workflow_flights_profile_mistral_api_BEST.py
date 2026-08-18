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

    # Handle cases with clear a.m./p.m. indicators
    time_part = re.search(r'(\d{1,2}:\d{2})\s*([ap]\.?m?\.?)', val, re.IGNORECASE)
    if time_part:
        time = time_part.group(1)
        period = time_part.group(2).replace('.', '').strip()
        if period.startswith('a'):
            period = 'a.m.'
        elif period.startswith('p'):
            period = 'p.m.'
        return f"{time} {period}"

    # Handle cases like "6:00aDec 1" -> extract "6:00" and infer period
    time_match = re.search(r'(\d{1,2}:\d{2})', val)
    if time_match:
        time = time_match.group(1)
        # Try to infer period from surrounding text
        if 'a' in val.lower() or 'am' in val.lower():
            return f"{time} a.m."
        elif 'p' in val.lower() or 'pm' in val.lower():
            return f"{time} p.m."
        # If no period found, return just the time (will be handled later)
        return time

    # Handle cases with just numbers (like "1230" for 12:30)
    num_match = re.search(r'^(\d{3,4})$', val)
    if num_match:
        num = num_match.group(1)
        if len(num) == 3:
            time = f"{num[0]}:{num[1:]}"
        else:
            time = f"{num[:2]}:{num[2:]}"
        # Infer period based on hour
        hour = int(time.split(':')[0])
        if hour < 12:
            return f"{time} a.m."
        else:
            return f"{time} p.m."

    return val

for col in time_cols:
    df[col] = df[col].apply(clean_time_format)

# Second pass: create a more robust imputation strategy
# We'll use both flight number and scheduled time as grouping keys
for col in time_cols:
    # First try to impute using both flight and corresponding scheduled time
    if 'sched' in col:
        # For scheduled times, we can't use other scheduled times as keys
        group_key = ['flight']
    else:
        # For actual times, use both flight and corresponding scheduled time
        sched_col = col.replace('act_', 'sched_')
        group_key = ['flight', sched_col]

    # Create group-based imputation
    group_mode = df.groupby(group_key)[col].transform(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    )
    df[col] = df[col].fillna(group_mode)

    # Fallback to global mode for any remaining missing values
    global_mode = df[col].mode()
    if not global_mode.empty:
        df[col] = df[col].fillna(global_mode[0])

    # Final pass: ensure all times have a.m./p.m. by using the most common period for each hour
    missing_period = df[col].str.contains(r'^\d{1,2}:\d{2}$', na=False)
    if missing_period.any():
        # Extract hour from time
        df['temp_hour'] = df[col].str.extract(r'^(\d{1,2}):\d{2}$')[0].astype(float)
        # Determine most common period for each hour
        hour_period = df[~missing_period].copy()
        hour_period['hour'] = hour_period[col].str.extract(r'^(\d{1,2}):\d{2}')[0].astype(float)
        period_map = hour_period.groupby('hour')[col].agg(
            lambda x: x.str.extract(r'([ap]\.m\.)')[0].mode()[0] if not x.str.extract(r'([ap]\.m\.)')[0].mode().empty else 'a.m.'
        )

        # Apply period to missing values
        for hour, period in period_map.items():
            mask = missing_period & (df['temp_hour'] == hour)
            df.loc[mask, col] = df.loc[mask, col] + f" {period}"

        # Clean up temp column
        df.drop('temp_hour', axis=1, inplace=True)

# Third pass: handle special cases where actual times should be close to scheduled times
# For actual departure times, they should be close to scheduled departure times
# For actual arrival times, they should be close to scheduled arrival times
for col in ['act_dep_time', 'act_arr_time']:
    sched_col = col.replace('act_', 'sched_')

    # For rows where both scheduled and actual times are present
    mask = df[col].notna() & df[sched_col].notna()

    # Extract minutes from both times
    def extract_minutes(time_str):
        if pd.isna(time_str):
            return np.nan
        try:
            time_part = re.search(r'(\d{1,2}):(\d{2})\s*([ap]\.m\.)', time_str)
            if time_part:
                hour = int(time_part.group(1))
                minute = int(time_part.group(2))
                period = time_part.group(3)
                if period == 'p.m.' and hour != 12:
                    hour += 12
                elif period == 'a.m.' and hour == 12:
                    hour = 0
                return hour * 60 + minute
            return np.nan
        except:
            return np.nan

    df['temp_actual'] = df[col].apply(extract_minutes)
    df['temp_sched'] = df[sched_col].apply(extract_minutes)

    # For times that are more than 2 hours apart, consider them suspicious
    suspicious = mask & (abs(df['temp_actual'] - df['temp_sched']) > 120)
    if suspicious.any():
        # Replace suspicious actual times with scheduled times (with period adjusted)
        for idx in df[suspicious].index:
            sched_time = df.loc[idx, sched_col]
            if pd.notna(sched_time):
                df.loc[idx, col] = sched_time

    # Clean up temp columns
    df.drop(['temp_actual', 'temp_sched'], axis=1, inplace=True)

# Final output
df