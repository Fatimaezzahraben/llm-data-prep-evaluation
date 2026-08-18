import pandas as pd
import numpy as np
import re

# Standardize date formats in reservation_status_date
df['reservation_status_date'] = pd.to_datetime(
    df['reservation_status_date'].str.replace(r'/', '-', regex=True),
    errors='coerce'
).fillna(pd.to_datetime(df['reservation_status_date'], format='%Y-%m-%d', errors='coerce'))

# Handle missing values and fix text columns
text_cols = [
    'hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
    'distribution_channel', 'reserved_room_type', 'assigned_room_type',
    'deposit_type', 'agent', 'customer_type', 'reservation_status'
]

# Clean text columns: strip whitespace, fix typos, and standardize categories
clean_mappings = {
    'hotel': {
        'Resoort Hotel': 'Resort Hotel',
        'RESOrt hotel': 'Resort Hotel',
        'Resort hotel': 'Resort Hotel',
        'City Hotel': 'City Hotel'
    },
    'arrival_date_month': {
        'July': 'July', 'August': 'August', 'September': 'September',
        'October': 'October', 'November': 'November', 'December': 'December',
        'January': 'January', 'February': 'February', 'March': 'March',
        'April': 'April', 'May': 'May', 'June': 'June'
    },
    'meal': {
        'BB': 'BB', 'HB': 'HB', 'FB': 'FB', 'SC': 'SC',
        'Undefined': 'Undefined', 'unknown': 'Undefined'
    },
    'deposit_type': {
        'No Deposit': 'No Deposit',
        'NoD eposit': 'No Deposit',
        'Non Refund': 'Non Refund',
        'Refundable': 'Refundable'
    },
    'customer_type': {
        'Transient': 'Transient',
        'Transient-Party': 'Transient-Party',
        'Contract': 'Contract',
        'Group': 'Group',
        '   Transient ': 'Transient',
        'CONtract': 'Contract'
    },
    'reservation_status': {
        'Check-Out': 'Check-Out',
        'Canceled': 'Canceled',
        'No-Show': 'No-Show'
    }
}

for col in text_cols:
    # Strip whitespace
    df[col] = df[col].astype(str).str.strip()

    # Apply specific mappings if available
    if col in clean_mappings:
        df[col] = df[col].replace(clean_mappings[col])

    # Fill missing values with mode
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col] = df[col].replace(['', 'nan', 'NaN', 'None', 'unknown'], np.nan)
        df[col] = df[col].fillna(mode_val.iloc[0])

# Handle numeric columns that are stored as text
numeric_text_cols = ['lead_time', 'adults', 'children', 'agent']

for col in numeric_text_cols:
    # Clean and convert to numeric
    df[col] = df[col].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing with median
    if pd.api.types.is_numeric_dtype(df[col]):
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# Handle purely numeric columns
numeric_cols = [
    'is_canceled', 'arrival_date_year', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'babies', 'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes', 'company',
    'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
    'total_of_special_requests'
]

for col in numeric_cols:
    if col in df.columns:
        # Ensure numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle missing values
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

        # Handle outliers by capping at 99th percentile
        if pd.api.types.is_numeric_dtype(df[col]):
            q_low = df[col].quantile(0.01)
            q_hi = df[col].quantile(0.99)
            df[col] = df[col].clip(q_low, q_hi)

# Special handling for children column (has empty strings and floats)
df['children'] = df['children'].replace('', np.nan)
df['children'] = pd.to_numeric(df['children'], errors='coerce')
df['children'] = df['children'].fillna(df['children'].median())

# Special handling for country column (high cardinality - use group-aware imputation)
if 'market_segment' in df.columns:
    # First try to impute by market_segment
    df['country'] = df.groupby('market_segment')['country'].transform(
        lambda x: x.replace('', np.nan).fillna(x.mode()[0] if not x.mode().empty else np.nan)
    )
    # Then fill remaining with global mode
    global_mode = df['country'].mode()
    if not global_mode.empty:
        df['country'] = df['country'].fillna(global_mode.iloc[0])

# Special handling for agent column (high cardinality - use group-aware imputation)
if 'market_segment' in df.columns:
    # First try to impute by market_segment
    df['agent'] = df.groupby('market_segment')['agent'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
    )
    # Then fill remaining with global mode
    global_mode = df['agent'].mode()
    if not global_mode.empty:
        df['agent'] = df['agent'].fillna(global_mode.iloc[0])

# Ensure all text columns are properly typed
for col in text_cols:
    df[col] = df[col].astype(str)

# Final check for any remaining missing values
for col in df.columns:
    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])