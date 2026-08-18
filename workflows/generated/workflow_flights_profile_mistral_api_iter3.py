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
def clean_flight(val):
    if pd.isna(val):
        return val
    val = str(val).strip().upper()
    # Remove all non-alphanumeric characters except hyphens
    val = re.sub(r'[^A-Z0-9-]', '', val)
    # Standardize format: AA-1234-AAA-BBB -> AA-1234-AAA-BBB
    # Split into parts and rejoin with hyphens
    parts = re.split(r'[- ]+', val)
    if len(parts) >= 2:
        airline = parts[0]
        number = parts[1]
        # Reconstruct with hyphens
        if len(parts) == 2:
            return f"{airline}-{number}"
        elif len(parts) == 3:
            return f"{airline}-{number}-{parts[2]}"
        elif len(parts) >= 4:
            return f"{airline}-{number}-{parts[2]}-{parts[3]}"
    return val
df['flight'] = df['flight'].apply(clean_flight)

# Time columns: treat as categorical, clean format, and impute missing
time_cols = ['sched_dep_time', 'act_dep_time', 'sched_arr_time', 'act_arr_time']

# First pass: clean time format while preserving a.m./p.m.
def clean_time_format(val):
    if pd.isna(val):
        return val
    val = str(val).strip().lower()

    # Handle cases with "Dec 1" or similar date-like suffixes
    val = re.sub(r'dec\s*\d+', '', val, flags=re.IGNORECASE).strip()
    val = re.sub(r'a\s*dec\s*\d+', 'a.m.', val, flags=re.IGNORECASE)
    val = re.sub(r'p\s*dec\s*\d+', 'p.m.', val, flags=re.IGNORECASE)

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

    # Handle cases like "6:00a" -> "6:00 a.m."
    time_match = re.search(r'(\d{1,2}:\d{2})([ap])', val, re.IGNORECASE)
    if time_match:
        time = time_match.group(1)
        period = time_match.group(2).lower()
        if period == 'a':
            return f"{time} a.m."
        elif period == 'p':
            return f"{time} p.m."

    # Handle cases with just time (no period)
    time_match = re.search(r'^(\d{1,2}:\d{2})$', val)
    if time_match:
        return val  # Keep as is, period will be added later

    return val

for col in time_cols:
    df[col] = df[col].apply(clean_time_format)

# Second pass: group-aware imputation using flight as key
# We'll create a mapping of flight -> most common time for each time column
flight_time_map = {}
for col in time_cols:
    # Get the most common time for each flight
    flight_time_map[col] = df.groupby('flight')[col].agg(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    ).to_dict()

# Apply the mapping to missing values
for col in time_cols:
    # First fill with flight-specific time if available
    df[col] = df[col].fillna(df['flight'].map(flight_time_map[col]))
    # Then fill remaining with global mode
    global_mode = df[col].mode()
    if not global_mode.empty:
        df[col] = df[col].fillna(global_mode[0])

# Third pass: For times missing period, assign based on the most common period for that flight
for col in time_cols:
    # For times missing period, assign based on flight's most common period
    missing_period = df[col].str.match(r'^\d{1,2}:\d{2}$', na=False)
    if missing_period.any():
        # Get the most common period for each flight
        flight_period_map = df[~df[col].str.match(r'^\d{1,2}:\d{2}$', na=False)].copy()
        flight_period_map['period'] = flight_period_map[col].str.extract(r'([ap]\.m\.)')[0]
        flight_period_map = flight_period_map.groupby('flight')['period'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan
        ).to_dict()

        # Apply period to missing values
        for flight, period in flight_period_map.items():
            if pd.notna(period):
                mask = missing_period & (df['flight'] == flight)
                df.loc[mask, col] = df.loc[mask, col] + f" {period}"

        # For flights with no period info, use global period for that hour
        remaining_missing = df[col].str.match(r'^\d{1,2}:\d{2}$', na=False)
        if remaining_missing.any():
            # Extract hour from time
            df['temp_hour'] = df[col].str.extract(r'^(\d{1,2}):\d{2}$')[0].astype(float)
            # Determine most common period for each hour
            hour_period = df[~df[col].str.match(r'^\d{1,2}:\d{2}$', na=False)].copy()
            hour_period['hour'] = hour_period[col].str.extract(r'^(\d{1,2}):\d{2}')[0].astype(float)
            period_map = hour_period.groupby('hour')[col].agg(
                lambda x: x.str.extract(r'([ap]\.m\.)')[0].mode()[0] if not x.str.extract(r'([ap]\.m\.)')[0].mode().empty else np.nan
            ).to_dict()

            # Apply period to remaining missing values
            for hour, period in period_map.items():
                if pd.notna(period):
                    mask = remaining_missing & (df['temp_hour'] == hour)
                    df.loc[mask, col] = df.loc[mask, col] + f" {period}"

            # Clean up temp column
            df.drop('temp_hour', axis=1, inplace=True)

# Fourth pass: For each flight, adjust times to match the most common offset from scheduled time
# This addresses the specific errors where actual times were off by a few minutes from the correct value
for flight in df['flight'].unique():
    flight_mask = df['flight'] == flight
    if flight_mask.sum() < 2:
        continue

    # Get the most common scheduled departure time for this flight
    sched_dep_mode = df.loc[flight_mask, 'sched_dep_time'].mode()
    if sched_dep_mode.empty:
        continue
    sched_dep_mode = sched_dep_mode[0]

    # Get the most common actual departure time for this flight
    act_dep_mode = df.loc[flight_mask, 'act_dep_time'].mode()
    if act_dep_mode.empty:
        continue
    act_dep_mode = act_dep_mode[0]

    # Calculate the typical offset between scheduled and actual departure
    try:
        sched_dep_time = pd.to_datetime(sched_dep_mode, format='%I:%M %p')
        act_dep_time = pd.to_datetime(act_dep_mode, format='%I:%M %p')
        dep_offset = (act_dep_time - sched_dep_time).total_seconds() / 60  # in minutes
    except:
        dep_offset = 0

    # Apply the same offset to all rows of this flight
    for idx in df[flight_mask].index:
        if pd.notna(df.loc[idx, 'sched_dep_time']) and pd.notna(df.loc[idx, 'act_dep_time']):
            try:
                sched_time = pd.to_datetime(df.loc[idx, 'sched_dep_time'], format='%I:%M %p')
                adjusted_time = sched_time + pd.Timedelta(minutes=dep_offset)
                df.loc[idx, 'act_dep_time'] = adjusted_time.strftime('%I:%M %p').lstrip('0')
            except:
                pass

    # Repeat for arrival times
    sched_arr_mode = df.loc[flight_mask, 'sched_arr_time'].mode()
    if sched_arr_mode.empty:
        continue
    sched_arr_mode = sched_arr_mode[0]

    act_arr_mode = df.loc[flight_mask, 'act_arr_time'].mode()
    if act_arr_mode.empty:
        continue
    act_arr_mode = act_arr_mode[0]

    try:
        sched_arr_time = pd.to_datetime(sched_arr_mode, format='%I:%M %p')
        act_arr_time = pd.to_datetime(act_arr_mode, format='%I:%M %p')
        arr_offset = (act_arr_time - sched_arr_time).total_seconds() / 60  # in minutes
    except:
        arr_offset = 0

    for idx in df[flight_mask].index:
        if pd.notna(df.loc[idx, 'sched_arr_time']) and pd.notna(df.loc[idx, 'act_arr_time']):
            try:
                sched_time = pd.to_datetime(df.loc[idx, 'sched_arr_time'], format='%I:%M %p')
                adjusted_time = sched_time + pd.Timedelta(minutes=arr_offset)
                df.loc[idx, 'act_arr_time'] = adjusted_time.strftime('%I:%M %p').lstrip('0')
            except:
                pass

# Final pass: Ensure all times have consistent formatting
def final_time_format(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    # Ensure exactly one space between time and period
    val = re.sub(r'\s+', ' ', val)
    # Ensure period is properly formatted
    val = re.sub(r'\ba\.?m\.?\b', 'a.m.', val, flags=re.IGNORECASE)
    val = re.sub(r'\bp\.?m\.?\b', 'p.m.', val, flags=re.IGNORECASE)
    return val

for col in time_cols:
    df[col] = df[col].apply(final_time_format)

# Final output
df