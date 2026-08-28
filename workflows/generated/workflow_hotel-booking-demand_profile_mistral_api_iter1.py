import pandas as pd
import numpy as np
import difflib
import re

# Helper: safe mode for any column
def safe_mode(series, fallback=None):
    _mode = series.mode(dropna=True)
    return _mode.iloc[0] if not _mode.empty else fallback

# Helper: fuzzy correction for very-low-cardinality columns
def fuzzy_correct_low_card(col, valid_values):
    def correct(val):
        if pd.isna(val):
            return val
        val_norm = str(val).strip().lower()
        if val_norm in [v.lower() for v in valid_values]:
            return next(v for v in valid_values if v.lower() == val_norm)
        match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_values], n=1, cutoff=0.6)
        if match:
            return next(v for v in valid_values if v.lower() == match[0])
        return val
    return col.apply(correct)

# Disguised missing -> NaN
disguised_missing = ["NA", "N/A", "unknown", "", " ", "NaN", "nan"]
for col in df.columns:
    df[col] = df[col].astype(str).replace({v: np.nan for v in disguised_missing}, regex=False)

# --- hotel (categorical, 0% missing, 82 unique) ---
valid_hotel = ['City Hotel', 'Resort Hotel']
df['hotel'] = fuzzy_correct_low_card(df['hotel'], valid_hotel)
df['hotel'] = df['hotel'].str.strip()

# --- is_canceled (numeric, 0% missing) ---
# Already numeric, no missing

# --- lead_time (categorical, 0% missing, 3708 unique) ---
# Convert to numeric
df['lead_time'] = pd.to_numeric(df['lead_time'].astype(str).str.replace(r'[^0-9]', '', regex=True), errors='coerce')
df['lead_time'] = df['lead_time'].fillna(df['lead_time'].median())

# --- arrival_date_year (numeric, 0% missing) ---
# Already numeric, no missing

# --- arrival_date_month (categorical, 0% missing, 12 unique) ---
month_map = {
    'January': 'January', 'February': 'February', 'March': 'March', 'April': 'April',
    'May': 'May', 'June': 'June', 'July': 'July', 'August': 'August',
    'September': 'September', 'October': 'October', 'November': 'November', 'December': 'December'
}
df['arrival_date_month'] = df['arrival_date_month'].str.strip().replace(month_map)

# --- arrival_date_week_number (numeric, 0% missing) ---
# Already numeric, no missing

# --- arrival_date_day_of_month (numeric, 0% missing) ---
# Already numeric, no missing

# --- stays_in_weekend_nights (numeric, 0% missing) ---
# Already numeric, no missing

# --- stays_in_week_nights (numeric, 0% missing, 792 unique) ---
# Cap extreme outlier (999 -> NaN, then median)
df.loc[df['stays_in_week_nights'] > 30, 'stays_in_week_nights'] = np.nan
df['stays_in_week_nights'] = df['stays_in_week_nights'].fillna(df['stays_in_week_nights'].median())

# --- adults (categorical, 0% missing, 45 unique) ---
# Convert to numeric, handle typos like '2O' -> '2'
df['adults'] = pd.to_numeric(df['adults'].astype(str).str.replace(r'[^0-9]', '', regex=True), errors='coerce')
df['adults'] = df['adults'].fillna(df['adults'].median())

# --- children (categorical, 10% missing, 7 unique) ---
# Convert to numeric, handle 'unknown' and empty
df['children'] = pd.to_numeric(df['children'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
df['children'] = df['children'].fillna(df['children'].median())

# --- babies (numeric, 0% missing) ---
# Cap extreme outlier (49 -> NaN, then median)
df.loc[df['babies'] > 10, 'babies'] = np.nan
df['babies'] = df['babies'].fillna(df['babies'].median())

# --- meal (categorical, 10% missing, 7 unique) ---
valid_meal = ['BB', 'HB', 'SC']
df['meal'] = fuzzy_correct_low_card(df['meal'], valid_meal)
df['meal'] = df['meal'].fillna(safe_mode(df['meal']))

# --- country (categorical, 10.37% missing, 176 unique) ---
# Normalize case and strip
df['country'] = df['country'].str.strip().str.upper()
df['country'] = df['country'].fillna(safe_mode(df['country']))

# --- market_segment (categorical, 10% missing, 10 unique) ---
valid_segment = ['Online TA', 'Offline TA/TO', 'Groups', 'Direct', 'Corporate']
df['market_segment'] = fuzzy_correct_low_card(df['market_segment'], valid_segment)
df['market_segment'] = df['market_segment'].fillna(safe_mode(df['market_segment']))

# --- distribution_channel (categorical, 0% missing, 5 unique) ---
valid_channel = ['TA/TO', 'Direct', 'Corporate', 'GDS', 'Undefined']
df['distribution_channel'] = fuzzy_correct_low_card(df['distribution_channel'], valid_channel)

# --- is_repeated_guest (numeric, 0% missing) ---
# Already numeric, no missing

# --- previous_cancellations (numeric, 0% missing) ---
# Already numeric, no missing

# --- previous_bookings_not_canceled (numeric, 0% missing) ---
# Already numeric, no missing

# --- reserved_room_type (categorical, 0% missing, 10 unique) ---
df['reserved_room_type'] = df['reserved_room_type'].str.strip().str.upper()
df['reserved_room_type'] = df['reserved_room_type'].fillna(safe_mode(df['reserved_room_type']))

# --- assigned_room_type (categorical, 0% missing, 12 unique) ---
df['assigned_room_type'] = df['assigned_room_type'].str.strip().str.upper()
df['assigned_room_type'] = df['assigned_room_type'].fillna(safe_mode(df['assigned_room_type']))

# --- booking_changes (numeric, 0% missing) ---
# Already numeric, no missing

# --- deposit_type (categorical, 0% missing, 82 unique) ---
valid_deposit = ['No Deposit', 'Non Refund']
df['deposit_type'] = fuzzy_correct_low_card(df['deposit_type'], valid_deposit)
df['deposit_type'] = df['deposit_type'].fillna(safe_mode(df['deposit_type']))

# --- agent (categorical, 22.33% missing, 327 unique) ---
# Convert to numeric
df['agent'] = pd.to_numeric(df['agent'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
# Group-aware impute (fillna only)
group_val = df.groupby(['country', 'market_segment', 'distribution_channel'])['agent'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['agent'] = df['agent'].fillna(group_val)
df['agent'] = df['agent'].fillna(df['agent'].median())

# --- company (numeric, 94.31% missing) ---
# Convert to numeric
df['company'] = pd.to_numeric(df['company'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
# Leave as NaN (very high missing rate)

# --- days_in_waiting_list (numeric, 0% missing) ---
# Cap extreme outlier (8996 -> NaN, then median)
df.loc[df['days_in_waiting_list'] > 365, 'days_in_waiting_list'] = np.nan
df['days_in_waiting_list'] = df['days_in_waiting_list'].fillna(df['days_in_waiting_list'].median())

# --- customer_type (categorical, 0% missing, 133 unique) ---
valid_customer = ['Transient', 'Transient-Party', 'Contract', 'Group']
df['customer_type'] = fuzzy_correct_low_card(df['customer_type'], valid_customer)
df['customer_type'] = df['customer_type'].fillna(safe_mode(df['customer_type']))

# --- adr (numeric, 0% missing) ---
# Cap extreme outliers (negative and > 1000 -> NaN, then median)
df.loc[(df['adr'] < 0) | (df['adr'] > 1000), 'adr'] = np.nan
# Group-aware impute (fillna only)
group_val = df.groupby(['hotel', 'market_segment', 'reserved_room_type'])['adr'].transform(
    lambda s: s.median() if not s.empty else np.nan
)
df['adr'] = df['adr'].fillna(group_val)
df['adr'] = df['adr'].fillna(df['adr'].median())

# --- required_car_parking_spaces (numeric, 0% missing) ---
# Cap extreme outlier (8 -> NaN, then median)
df.loc[df['required_car_parking_spaces'] > 3, 'required_car_parking_spaces'] = np.nan
df['required_car_parking_spaces'] = df['required_car_parking_spaces'].fillna(df['required_car_parking_spaces'].median())

# --- total_of_special_requests (numeric, 0% missing) ---
# Already numeric, no missing

# --- reservation_status (categorical, 0% missing, 3 unique) ---
valid_status = ['Check-Out', 'Canceled', 'No-Show']
df['reservation_status'] = fuzzy_correct_low_card(df['reservation_status'], valid_status)

# --- reservation_status_date (categorical, 0% missing, 5361 unique) ---
# Parse multi-format dates
parsed = pd.Series(pd.NaT, index=df.index)
raw = df['reservation_status_date'].astype(str)
for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d-%b-%Y', '%Y/%m/%d']:
    still_missing = parsed.isna()
    parsed.loc[still_missing] = pd.to_datetime(raw[still_missing], format=fmt, errors='coerce')
df['reservation_status_date'] = parsed.dt.strftime('%Y-%m-%d')
df['reservation_status_date'] = df['reservation_status_date'].fillna(safe_mode(df['reservation_status_date']))

df