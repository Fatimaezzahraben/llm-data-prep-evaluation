import pandas as pd
import numpy as np
import re

# Helper function to detect disguised missing values
def is_disguised_missing(val):
    if pd.isna(val):
        return True
    val = str(val).strip().lower()
    return val in ['', 'na', 'n/a', 'unknown', 'null', 'none']

# 1. Handle missing values (real and disguised) for all columns
missing_cols = {
    'children': 'categorical',
    'meal': 'categorical',
    'country': 'categorical',
    'market_segment': 'categorical',
    'agent': 'categorical',
    'company': 'numeric'
}

for col, col_type in missing_cols.items():
    # Convert to string and check for disguised missing
    df[col] = df[col].astype(str).replace('nan', np.nan)
    mask = df[col].apply(is_disguised_missing)
    df.loc[mask, col] = np.nan

    if col_type == 'categorical':
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])
    else:
        # Convert to numeric first
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# 2. Clean and standardize categorical columns with high cardinality
# hotel column
hotel_mapping = {
    '  City Hotel ': 'City Hotel',
    '  Resort Hotel ': 'Resort Hotel',
    'CITY hotel': 'City Hotel',
    'RESORT HOTEL': 'Resort Hotel',
    'city hotel': 'City Hotel',
    'resort hotel': 'Resort Hotel'
}
df['hotel'] = df['hotel'].str.strip()
df['hotel'] = df['hotel'].replace(hotel_mapping)

# adults column - fix typos like '2O' to '2'
df['adults'] = df['adults'].astype(str).str.replace(r'[^0-9]', '', regex=True)
df['adults'] = pd.to_numeric(df['adults'], errors='coerce')
median_adults = df['adults'].median()
df['adults'] = df['adults'].fillna(median_adults)

# lead_time - convert to numeric
df['lead_time'] = pd.to_numeric(df['lead_time'], errors='coerce')
median_lead_time = df['lead_time'].median()
df['lead_time'] = df['lead_time'].fillna(median_lead_time)

# arrival_date_month - standardize
month_mapping = {
    'January': 'January', 'january': 'January', 'Jan': 'January',
    'February': 'February', 'february': 'February', 'Feb': 'February',
    'March': 'March', 'march': 'March', 'Mar': 'March',
    'April': 'April', 'april': 'April', 'Apr': 'April',
    'May': 'May', 'may': 'May',
    'June': 'June', 'june': 'June', 'Jun': 'June',
    'July': 'July', 'july': 'July', 'Jul': 'July',
    'August': 'August', 'august': 'August', 'Aug': 'August',
    'September': 'September', 'september': 'September', 'Sep': 'September',
    'October': 'October', 'october': 'October', 'Oct': 'October',
    'November': 'November', 'november': 'November', 'Nov': 'November',
    'December': 'December', 'december': 'December', 'Dec': 'December'
}
df['arrival_date_month'] = df['arrival_date_month'].str.capitalize()
df['arrival_date_month'] = df['arrival_date_month'].replace(month_mapping)

# deposit_type
deposit_mapping = {
    '  No Deposit ': 'No Deposit',
    'NO deposit': 'No Deposit',
    '  Non Refund ': 'Non Refund',
    'non refund': 'Non Refund',
    'NonRefund': 'Non Refund',
    'NoDeposit': 'No Deposit'
}
df['deposit_type'] = df['deposit_type'].str.strip()
df['deposit_type'] = df['deposit_type'].replace(deposit_mapping)

# customer_type
customer_mapping = {
    '  Transient ': 'Transient',
    'transient': 'Transient',
    'Transient-Party': 'Transient-Party',
    'Contract': 'Contract',
    'Group': 'Group'
}
df['customer_type'] = df['customer_type'].str.strip()
df['customer_type'] = df['customer_type'].replace(customer_mapping)

# 3. Ensure numeric columns contain only numeric values
numeric_cols = [
    'is_canceled', 'arrival_date_year', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights',
    'stays_in_week_nights', 'babies', 'is_repeated_guest',
    'previous_cancellations', 'previous_bookings_not_canceled',
    'booking_changes', 'days_in_waiting_list', 'adr',
    'required_car_parking_spaces', 'total_of_special_requests'
]

for col in numeric_cols:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# 4. Cap outliers for numeric columns
def cap_outliers(series, min_val, max_val):
    return series.clip(lower=min_val, upper=max_val)

numeric_ranges = {
    'lead_time': (0, 730),  # 2 years max
    'stays_in_weekend_nights': (0, 14),
    'stays_in_week_nights': (0, 30),
    'adults': (1, 4),  # Based on top values
    'babies': (0, 10),
    'previous_cancellations': (0, 10),
    'previous_bookings_not_canceled': (0, 20),
    'booking_changes': (0, 10),
    'days_in_waiting_list': (0, 365),
    'adr': (0, 1000),
    'required_car_parking_spaces': (0, 3),
    'total_of_special_requests': (0, 5)
}

for col, (min_val, max_val) in numeric_ranges.items():
    if col in df.columns:
        df[col] = cap_outliers(df[col], min_val, max_val)

# 5. Clean reservation_status_date (date column)
df['reservation_status_date'] = pd.to_datetime(
    df['reservation_status_date'],
    errors='coerce',
    format='mixed'
)
# For any remaining unparseable dates, impute with mode
mode_date = df['reservation_status_date'].mode()
if not mode_date.empty:
    df['reservation_status_date'] = df['reservation_status_date'].fillna(mode_date.iloc[0])

# 6. Group-aware imputation for high-cardinality columns
# For agent column (high cardinality), impute using market_segment as group
if 'agent' in df.columns and 'market_segment' in df.columns:
    # First convert agent to numeric
    df['agent'] = pd.to_numeric(df['agent'], errors='coerce')

    # Group by market_segment and impute with group mode
    df['agent'] = df.groupby('market_segment')['agent'].transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    )

    # Fallback to global mode
    global_mode = df['agent'].mode()
    if not global_mode.empty:
        df['agent'] = df['agent'].fillna(global_mode.iloc[0])

# For company column (very high missing rate), impute with 0 (no company)
if 'company' in df.columns:
    df['company'] = df['company'].fillna(0)

# 7. Final pass to ensure all numeric columns are properly typed
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        continue
    if col in numeric_cols or col in ['lead_time', 'adults', 'children', 'agent', 'company']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)