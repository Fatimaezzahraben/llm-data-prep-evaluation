import pandas as pd
import numpy as np
import re

# Helper function to detect disguised missing values
def is_disguised_missing(val):
    if pd.isna(val):
        return True
    val = str(val).strip().lower()
    return val in ['', 'na', 'n/a', 'unknown', 'null']

# Clean tuple_id (numeric, no missing, but ensure numeric)
df['tuple_id'] = pd.to_numeric(df['tuple_id'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')
# Cap outliers (using 1.5*IQR rule)
q1 = df['tuple_id'].quantile(0.25)
q3 = df['tuple_id'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df['tuple_id'] = df['tuple_id'].clip(lower_bound, upper_bound)
# Fill any NaN from conversion with median
df['tuple_id'] = df['tuple_id'].fillna(df['tuple_id'].median())

# Clean src (categorical, no missing, but standardize case and whitespace)
df['src'] = df['src'].astype(str).str.strip().str.lower()
# No high cardinality - no need for group-aware imputation

# Clean flight (categorical, no missing, but standardize format)
df['flight'] = df['flight'].astype(str).str.strip().str.upper()
# Check for common patterns and standardize
flight_pattern = re.compile(r'^([A-Z]{2})-(\d+)-([A-Z]{3})-([A-Z]{3})$')
def standardize_flight(val):
    val = str(val).strip().upper()
    match = flight_pattern.match(val)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}-{match.group(4)}"
    return val
df['flight'] = df['flight'].apply(standardize_flight)

# Clean sched_dep_time (categorical/text with 33% missing)
# First mark all disguised missing as NaN
df['sched_dep_time'] = df['sched_dep_time'].apply(lambda x: np.nan if is_disguised_missing(x) else x)
# Group-aware imputation using flight as key (n_unique=100 vs 2376 rows)
df['sched_dep_time'] = df.groupby('flight')['sched_dep_time'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
# Fallback to global mode
global_mode = df['sched_dep_time'].mode()
if not global_mode.empty:
    df['sched_dep_time'] = df['sched_dep_time'].fillna(global_mode.iloc[0])

# Clean act_dep_time (categorical/text with 15.82% missing)
df['act_dep_time'] = df['act_dep_time'].apply(lambda x: np.nan if is_disguised_missing(x) else x)
# Group-aware imputation using flight as key
df['act_dep_time'] = df.groupby('flight')['act_dep_time'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
# Fallback to global mode
global_mode = df['act_dep_time'].mode()
if not global_mode.empty:
    df['act_dep_time'] = df['act_dep_time'].fillna(global_mode.iloc[0])

# Clean sched_arr_time (categorical/text with 32.41% missing)
df['sched_arr_time'] = df['sched_arr_time'].apply(lambda x: np.nan if is_disguised_missing(x) else x)
# Group-aware imputation using flight as key
df['sched_arr_time'] = df.groupby('flight')['sched_arr_time'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
# Fallback to global mode
global_mode = df['sched_arr_time'].mode()
if not global_mode.empty:
    df['sched_arr_time'] = df['sched_arr_time'].fillna(global_mode.iloc[0])

# Clean act_arr_time (categorical/text with 16.08% missing)
df['act_arr_time'] = df['act_arr_time'].apply(lambda x: np.nan if is_disguised_missing(x) else x)
# Group-aware imputation using flight as key
df['act_arr_time'] = df.groupby('flight')['act_arr_time'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
# Fallback to global mode
global_mode = df['act_arr_time'].mode()
if not global_mode.empty:
    df['act_arr_time'] = df['act_arr_time'].fillna(global_mode.iloc[0])

# Final validation for time columns (ensure they look like times)
time_pattern = re.compile(r'^\d{1,2}:\d{2}\s*[ap]\.?m\.?$', re.IGNORECASE)
for col in ['sched_dep_time', 'act_dep_time', 'sched_arr_time', 'act_arr_time']:
    # Replace any values that don't match the time pattern with mode
    invalid_mask = ~df[col].astype(str).str.match(time_pattern, na=False)
    if invalid_mask.any():
        mode_val = df[col].mode()
        if not mode_val.empty:
            df.loc[invalid_mask, col] = mode_val.iloc[0]