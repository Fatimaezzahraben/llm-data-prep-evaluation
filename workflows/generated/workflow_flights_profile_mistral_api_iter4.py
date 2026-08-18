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

    # Handle cases with month/day (like "6:00aDec 1")
    if re.search(r'[a-z]{3}\s*\d+', val):
        time_part = re.search(r'(\d{1,2}:\d{2})', val)
        if time_part:
            time = time_part.group(1)
            period = 'a.m.' if 'a' in val else 'p.m.' if 'p' in val else None
            if period:
                return f"{time} {period}"
        return val

    # Standard time format with period
    time_part = re.search(r'(\d{1,2}:\d{2})\s*([ap]\.?m?\.?)', val, re.IGNORECASE)
    if time_part:
        time = time_part.group(1)
        period = time_part.group(2).replace('.', '').strip()
        if period.startswith('a'):
            period = 'a.m.'
        elif period.startswith('p'):
            period = 'p.m.'
        return f"{time} {period}"

    # Handle cases like "6:00a" or "6:00p"
    time_match = re.search(r'(\d{1,2}:\d{2})([ap])', val, re.IGNORECASE)
    if time_match:
        time = time_match.group(1)
        period = 'a.m.' if time_match.group(2).lower() == 'a' else 'p.m.'
        return f"{time} {period}"

    # Handle cases with just time (no period)
    time_match = re.search(r'^(\d{1,2}:\d{2})$', val)
    if time_match:
        return val  # Will be handled in group imputation

    return val

for col in time_cols:
    df[col] = df[col].apply(clean_time_format)

# Second pass: create a more accurate mapping using both flight and scheduled time
# We'll create a mapping of (flight, sched_time) -> most common actual time
time_mapping = {}
for time_col in ['act_dep_time', 'act_arr_time']:
    # For actual times, we'll map from (flight, scheduled time) to actual time
    if time_col == 'act_dep_time':
        sched_col = 'sched_dep_time'
    else:
        sched_col = 'sched_arr_time'

    # Create a composite key of flight and scheduled time
    df['composite_key'] = df['flight'] + "|" + df[sched_col]

    # Get the most common actual time for each composite key
    mapping = df.groupby('composite_key')[time_col].agg(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    ).to_dict()

    time_mapping[time_col] = mapping

    # Apply the mapping to missing values
    df[time_col] = df[time_col].fillna(df['composite_key'].map(mapping))

    # Fallback to flight-only mapping if composite key fails
    flight_mapping = df.groupby('flight')[time_col].agg(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    ).to_dict()
    df[time_col] = df[time_col].fillna(df['flight'].map(flight_mapping))

    # Final fallback to global mode
    global_mode = df[time_col].mode()
    if not global_mode.empty:
        df[time_col] = df[time_col].fillna(global_mode[0])

    # Clean up
    df.drop('composite_key', axis=1, inplace=True)

# For scheduled times, we'll use flight-only mapping
for time_col in ['sched_dep_time', 'sched_arr_time']:
    # First try to fill with flight mapping
    flight_mapping = df.groupby('flight')[time_col].agg(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    ).to_dict()
    df[time_col] = df[time_col].fillna(df['flight'].map(flight_mapping))

    # Then fill remaining with global mode
    global_mode = df[time_col].mode()
    if not global_mode.empty:
        df[time_col] = df[time_col].fillna(global_mode[0])

# Third pass: ensure all times have a.m./p.m. by using the most common period for each hour
for col in time_cols:
    # For times missing period, assign based on hour
    missing_period = df[col].str.match(r'^\d{1,2}:\d{2}$', na=False)
    if missing_period.any():
        # Extract hour from time
        hours = df[col].str.extract(r'^(\d{1,2}):\d{2}$')[0].astype(float)

        # Determine most common period for each hour
        period_data = df[~missing_period].copy()
        period_data['hour'] = period_data[col].str.extract(r'^(\d{1,2}):\d{2}')[0].astype(float)
        period_map = period_data.groupby('hour')[col].agg(
            lambda x: x.str.extract(r'([ap]\.m\.)')[0].mode()[0] if not x.str.extract(r'([ap]\.m\.)')[0].mode().empty else 'a.m.'
        )

        # Apply period to missing values
        for hour, period in period_map.items():
            mask = missing_period & (hours == hour)
            df.loc[mask, col] = df.loc[mask, col] + f" {period}"

# Final pass: round times to nearest 5 minutes where appropriate
def round_time(val):
    if pd.isna(val):
        return val
    if not re.match(r'^\d{1,2}:\d{2} [ap]\.m\.$', val):
        return val

    time_part, period = val.split()
    hour, minute = map(int, time_part.split(':'))
    total_minutes = hour * 60 + minute

    # Round to nearest 5 minutes
    rounded_minutes = round(total_minutes / 5) * 5
    if rounded_minutes >= 720:  # Handle 12:00 p.m. case
        rounded_minutes -= 720
    rounded_hour = rounded_minutes // 60
    rounded_minute = rounded_minutes % 60

    # Handle 12-hour format
    if period == 'a.m.' and rounded_hour == 0:
        rounded_hour = 12
    elif period == 'p.m.' and rounded_hour < 12:
        rounded_hour += 12
    elif period == 'a.m.' and rounded_hour == 12:
        rounded_hour = 0

    return f"{rounded_hour}:{rounded_minute:02d} {period}"

for col in time_cols:
    df[col] = df[col].apply(round_time)

# Final output
df