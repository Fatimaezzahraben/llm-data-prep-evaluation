import pandas as pd
import numpy as np
import re
import difflib

# tuple_id: numeric, 0% missing, no issues detected
df['tuple_id'] = pd.to_numeric(df['tuple_id'], errors='coerce')
df['tuple_id'] = df['tuple_id'].fillna(df['tuple_id'].median())

# src: categorical, 0% missing, but check for typos/near-duplicates
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
df['flight'] = df['flight'].astype(str).str.strip().str.upper()

# Time cleaning function that preserves original formatting
def clean_time(val):
    if pd.isna(val):
        return val
    val = str(val).strip()

    # Handle disguised missing values
    if val.lower() in ['na', 'n/a', 'unknown', '']:
        return np.nan

    # Handle 'Delayed' -> NaN
    if val.lower() == 'delayed':
        return np.nan

    # Handle cases like '5:58aDec 1' -> '5:58 a.m.'
    if re.match(r'^\d{1,2}:\d{2}[ap]m?\s*[a-zA-Z]', val, re.IGNORECASE):
        val = re.sub(r'([ap]m?)\s*[a-zA-Z].*', r'\1', val, flags=re.IGNORECASE)
        val = val.replace('a', ' a.m.').replace('p', ' p.m.')

    # Handle cases like '5:58a' -> '5:58 a.m.'
    if re.match(r'^\d{1,2}:\d{2}[ap]m?$', val, re.IGNORECASE):
        val = val.replace('a', ' a.m.').replace('p', ' p.m.')

    # Ensure consistent formatting (a.m./p.m. with dots, space before)
    val = re.sub(r'([ap])(\.?m\.?)', r'\1.m.', val, flags=re.IGNORECASE)
    val = re.sub(r'(\d)([ap]\.m\.)', r'\1 \2', val, flags=re.IGNORECASE)
    val = re.sub(r'\s+', ' ', val).strip()

    return val

# sched_dep_time: categorical/text, 33% missing
disguised_missing = df['sched_dep_time'].astype(str).str.strip().str.lower().isin(['na', 'n/a', 'unknown', ''])
df.loc[disguised_missing, 'sched_dep_time'] = np.nan
df['sched_dep_time'] = df['sched_dep_time'].apply(clean_time)

# Apply EXACT dependency: flight -> sched_dep_time (whole column overwrite)
def get_mode(s):
    mode_vals = s.mode(dropna=True)
    return mode_vals.iloc[0] if not mode_vals.empty else np.nan
df['sched_dep_time'] = df.groupby('flight')['sched_dep_time'].transform(get_mode)

# Global fallback for any remaining missing values
_mode = df['sched_dep_time'].mode(dropna=True)
fill_value = _mode.iloc[0] if not _mode.empty else '7:10 a.m.'
df['sched_dep_time'] = df['sched_dep_time'].fillna(fill_value)

# sched_arr_time: categorical/text, 32.41% missing
disguised_missing = df['sched_arr_time'].astype(str).str.strip().str.lower().isin(['na', 'n/a', 'unknown', ''])
df.loc[disguised_missing, 'sched_arr_time'] = np.nan
df['sched_arr_time'] = df['sched_arr_time'].apply(clean_time)

# Apply EXACT dependency: flight -> sched_arr_time (whole column overwrite)
df['sched_arr_time'] = df.groupby('flight')['sched_arr_time'].transform(get_mode)

# Global fallback for any remaining missing values
_mode = df['sched_arr_time'].mode(dropna=True)
fill_value = _mode.iloc[0] if not _mode.empty else '2:35 p.m.'
df['sched_arr_time'] = df['sched_arr_time'].fillna(fill_value)

# act_dep_time: categorical/text, 15.82% missing
disguised_missing = df['act_dep_time'].astype(str).str.strip().str.lower().isin(['na', 'n/a', 'unknown', ''])
df.loc[disguised_missing, 'act_dep_time'] = np.nan
df['act_dep_time'] = df['act_dep_time'].apply(clean_time)

# Create a mapping from flight to typical departure delay (in minutes)
flight_delay_map = {}
for flight in df['flight'].unique():
    mask = (df['flight'] == flight) & df['sched_dep_time'].notna() & df['act_dep_time'].notna()
    if mask.sum() >= 2:  # Only consider flights with at least 2 valid pairs
        sched_times = df.loc[mask, 'sched_dep_time']
        act_times = df.loc[mask, 'act_dep_time']

        # Convert times to minutes since midnight for calculation
        def time_to_minutes(t):
            if pd.isna(t):
                return np.nan
            t = str(t).strip().lower()
            try:
                if 'a.m.' in t:
                    h, m = re.findall(r'(\d+):(\d+)', t)[0]
                    return int(h) * 60 + int(m)
                elif 'p.m.' in t and '12:' not in t:
                    h, m = re.findall(r'(\d+):(\d+)', t)[0]
                    return (int(h) + 12) * 60 + int(m)
                else:
                    h, m = re.findall(r'(\d+):(\d+)', t)[0]
                    return int(h) * 60 + int(m)
            except (IndexError, ValueError):
                return np.nan

        sched_mins = sched_times.apply(time_to_minutes)
        act_mins = act_times.apply(time_to_minutes)
        delays = (act_mins - sched_mins).dropna()
        if len(delays) > 0:
            flight_delay_map[flight] = int(round(delays.median()))

# Apply the delay mapping to impute missing actual times
def impute_act_time(row, time_col, delay_map):
    if pd.isna(row[time_col]):
        sched_time = row['sched_dep_time'] if time_col == 'act_dep_time' else row['sched_arr_time']
        flight = row['flight']
        if pd.notna(sched_time) and flight in delay_map:
            # Convert scheduled time to minutes
            t = str(sched_time).strip().lower()
            try:
                if 'a.m.' in t:
                    h, m = re.findall(r'(\d+):(\d+)', t)[0]
                    total_mins = int(h) * 60 + int(m) + delay_map[flight]
                elif 'p.m.' in t and '12:' not in t:
                    h, m = re.findall(r'(\d+):(\d+)', t)[0]
                    total_mins = (int(h) + 12) * 60 + int(m) + delay_map[flight]
                else:
                    h, m = re.findall(r'(\d+):(\d+)', t)[0]
                    total_mins = int(h) * 60 + int(m) + delay_map[flight]

                # Handle day rollover
                total_mins %= 1440

                # Convert back to time string
                h = total_mins // 60
                m = total_mins % 60
                if h < 12:
                    period = 'a.m.'
                    if h == 0:
                        h = 12
                else:
                    period = 'p.m.'
                    if h > 12:
                        h -= 12
                return f"{h}:{m:02d} {period}"
            except (IndexError, ValueError):
                pass
    return row[time_col]

# Apply the delay-based imputation first
df['act_dep_time'] = df.apply(lambda row: impute_act_time(row, 'act_dep_time', flight_delay_map), axis=1)

# Then apply composite key dependency for remaining missing values
group_val = df.groupby(['flight', 'sched_dep_time'])['act_dep_time'].transform(get_mode)
df['act_dep_time'] = df['act_dep_time'].fillna(group_val)

# Fallback to flight-only grouping
group_val = df.groupby('flight')['act_dep_time'].transform(get_mode)
df['act_dep_time'] = df['act_dep_time'].fillna(group_val)

# Global fallback for any remaining missing values
_mode = df['act_dep_time'].mode(dropna=True)
fill_value = _mode.iloc[0] if not _mode.empty else '7:22 a.m.'
df['act_dep_time'] = df['act_dep_time'].fillna(fill_value)

# act_arr_time: categorical/text, 16.08% missing
disguised_missing = df['act_arr_time'].astype(str).str.strip().str.lower().isin(['na', 'n/a', 'unknown', ''])
df.loc[disguised_missing, 'act_arr_time'] = np.nan
df['act_arr_time'] = df['act_arr_time'].apply(clean_time)

# Create a mapping from flight to typical arrival delay (in minutes)
flight_arr_delay_map = {}
for flight in df['flight'].unique():
    mask = (df['flight'] == flight) & df['sched_arr_time'].notna() & df['act_arr_time'].notna()
    if mask.sum() >= 2:  # Only consider flights with at least 2 valid pairs
        sched_times = df.loc[mask, 'sched_arr_time']
        act_times = df.loc[mask, 'act_arr_time']

        def time_to_minutes(t):
            if pd.isna(t):
                return np.nan
            t = str(t).strip().lower()
            try:
                if 'a.m.' in t:
                    h, m = re.findall(r'(\d+):(\d+)', t)[0]
                    return int(h) * 60 + int(m)
                elif 'p.m.' in t and '12:' not in t:
                    h, m = re.findall(r'(\d+):(\d+)', t)[0]
                    return (int(h) + 12) * 60 + int(m)
                else:
                    h, m = re.findall(r'(\d+):(\d+)', t)[0]
                    return int(h) * 60 + int(m)
            except (IndexError, ValueError):
                return np.nan

        sched_mins = sched_times.apply(time_to_minutes)
        act_mins = act_times.apply(time_to_minutes)
        delays = (act_mins - sched_mins).dropna()
        if len(delays) > 0:
            flight_arr_delay_map[flight] = int(round(delays.median()))

# Apply the delay-based imputation first
df['act_arr_time'] = df.apply(lambda row: impute_act_time(row, 'act_arr_time', flight_arr_delay_map), axis=1)

# Then apply composite key dependency for remaining missing values
group_val = df.groupby(['flight', 'sched_arr_time'])['act_arr_time'].transform(get_mode)
df['act_arr_time'] = df['act_arr_time'].fillna(group_val)

# Fallback to flight-only grouping
group_val = df.groupby('flight')['act_arr_time'].transform(get_mode)
df['act_arr_time'] = df['act_arr_time'].fillna(group_val)

# Global fallback for any remaining missing values
_mode = df['act_arr_time'].mode(dropna=True)
fill_value = _mode.iloc[0] if not _mode.empty else '7:10 p.m.'
df['act_arr_time'] = df['act_arr_time'].fillna(fill_value)

# Final type consistency and formatting
for col in ['sched_dep_time', 'act_dep_time', 'sched_arr_time', 'act_arr_time']:
    df[col] = df[col].astype(str).str.strip()
    # Ensure consistent a.m./p.m. formatting with dots and space
    df[col] = df[col].str.replace(r'([ap])m\.?', r'\1.m.', regex=True, flags=re.IGNORECASE)
    df[col] = df[col].str.replace(r'(\d)([ap]\.m\.)', r'\1 \2', regex=True)
    df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()

df