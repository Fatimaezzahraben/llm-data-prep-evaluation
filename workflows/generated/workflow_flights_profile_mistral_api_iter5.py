import pandas as pd
import difflib
import numpy as np
import re

# Clean tuple_id (numeric, no missing values)
df['tuple_id'] = pd.to_numeric(df['tuple_id'], errors='coerce')

# Clean src (categorical, no missing values, standardize)
src_mapping = {
    'helloflight': 'helloflight',
    'boston': 'boston',
    'airtravelcenter': 'airtravelcenter',
    'flightview': 'flightview',
    'panynj': 'panynj',
    'businesstravellogue': 'businesstravellogue',
    'flylouisville': 'flylouisville',
    'orbitz': 'orbitz',
    'myrateplan': 'myrateplan',
    'flightstats': 'flightstats',
    'ua': 'ua'
}
df['src'] = df['src'].replace(src_mapping)

# Clean flight (categorical, no missing values, standardize format)
flight_pattern = re.compile(r'^([A-Z]{2}-\d{1,4}-[A-Z]{3}-[A-Z]{3})$')
def clean_flight(flight):
    if pd.isna(flight):
        return flight
    flight = str(flight).strip()
    if not re.match(r'^[A-Z]{2}-\d{1,4}-[A-Z]{3}-[A-Z]{3}$', flight, re.IGNORECASE):
        return flight
    return flight.upper()
df['flight'] = df['flight'].apply(clean_flight)

# Clean sched_dep_time (categorical, 33% missing)
# First handle disguised missing values
disguised_missing = ['Not Available', 'Delayed', 'NA', 'N/A', 'unknown', '', ' ', 'NaN', 'nan']
df['sched_dep_time'] = df['sched_dep_time'].replace(disguised_missing, np.nan)

# Clean time strings by removing date suffixes and parenthetical notes
def clean_time(time_str):
    if pd.isna(time_str):
        return time_str
    time_str = str(time_str).strip()
    # Remove date suffixes (e.g., 'aDec 1')
    time_str = re.sub(r'\s+\w+\s*\d+', '', time_str)
    # Remove parenthetical notes (e.g., '(Estimated)')
    time_str = re.sub(r'\s*\(.*\)', '', time_str)
    # Standardize time format (e.g., '6:00a' -> '6:00 a.m.')
    time_str = re.sub(r'(\d+):(\d{2})([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    # Standardize time format (e.g., '6:00a' -> '6:00 a.m.')
    time_str = re.sub(r'(\d+):(\d{2})([ap])', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    # Standardize time format (e.g., '6:00am' -> '6:00 a.m.')
    time_str = re.sub(r'(\d+):(\d{2})([ap])$', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    # Standardize time format (e.g., '6:00am' -> '6:00 a.m.')
    time_str = re.sub(r'(\d+):(\d{2})([ap])$', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    return time_str

df['sched_dep_time'] = df['sched_dep_time'].apply(clean_time)

# Use exact key to impute sched_dep_time
df['sched_dep_time'] = df.groupby('flight')['sched_dep_time'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['sched_dep_time'] = df['sched_dep_time'].fillna(df['sched_dep_time'].mode().iloc[0] if not df['sched_dep_time'].mode().empty else np.nan)

# Clean act_dep_time (categorical, 15.82% missing)
# First handle disguised missing values
df['act_dep_time'] = df['act_dep_time'].replace(disguised_missing, np.nan)

# Clean time strings by removing date suffixes and parenthetical notes
df['act_dep_time'] = df['act_dep_time'].apply(clean_time)

# Use exact key to impute act_dep_time
df['act_dep_time'] = df.groupby(['flight', 'sched_dep_time'])['act_dep_time'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['act_dep_time'] = df['act_dep_time'].fillna(df['act_dep_time'].mode().iloc[0] if not df['act_dep_time'].mode().empty else np.nan)

# Clean sched_arr_time (categorical, 32.41% missing)
# First handle disguised missing values
df['sched_arr_time'] = df['sched_arr_time'].replace(disguised_missing, np.nan)

# Clean time strings by removing date suffixes and parenthetical notes
df['sched_arr_time'] = df['sched_arr_time'].apply(clean_time)

# Use exact key to impute sched_arr_time
df['sched_arr_time'] = df.groupby('flight')['sched_arr_time'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['sched_arr_time'] = df['sched_arr_time'].fillna(df['sched_arr_time'].mode().iloc[0] if not df['sched_arr_time'].mode().empty else np.nan)

# Clean act_arr_time (categorical, 16.08% missing)
# First handle disguised missing values
df['act_arr_time'] = df['act_arr_time'].replace(disguised_missing, np.nan)

# Clean time strings by removing date suffixes and parenthetical notes
df['act_arr_time'] = df['act_arr_time'].apply(clean_time)

# Use exact key to impute act_arr_time
df['act_arr_time'] = df.groupby(['flight', 'sched_arr_time'])['act_arr_time'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['act_arr_time'] = df['act_arr_time'].fillna(df['act_arr_time'].mode().iloc[0] if not df['act_arr_time'].mode().empty else np.nan)

# Final cleaning of time columns to ensure consistent formatting
def final_time_clean(time_str):
    if pd.isna(time_str):
        return time_str
    time_str = str(time_str).strip()
    # Standardize time format (e.g., '6:00 a.m.' -> '6:00 a.m.')
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    # Standardize time format (e.g., '6:00am' -> '6:00 a.m.')
    time_str = re.sub(r'(\d+):(\d{2})([ap])$', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    return time_str

df['sched_dep_time'] = df['sched_dep_time'].apply(final_time_clean)
df['act_dep_time'] = df['act_dep_time'].apply(final_time_clean)
df['sched_arr_time'] = df['sched_arr_time'].apply(final_time_clean)
df['act_arr_time'] = df['act_arr_time'].apply(final_time_clean)