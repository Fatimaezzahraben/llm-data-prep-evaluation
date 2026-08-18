import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('hotel_bookings.csv')

# 1. Fix inconsistent date formats in 'reservation_status_date'
# Convert to datetime and standardize format
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce').dt.strftime('%Y-%m-%d')

# 2. Handle missing values appropriately
# For numeric columns with missing values, fill with median (less sensitive to outliers)
numeric_cols = ['children', 'agent', 'company']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Ensure numeric
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# For categorical columns with missing values, fill with mode
categorical_cols = ['country', 'meal', 'market_segment', 'distribution_channel',
                    'reserved_room_type', 'assigned_room_type', 'deposit_type',
                    'customer_type', 'reservation_status']
for col in categorical_cols:
    if col in df.columns:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

# Special case for 'children' - if missing and adults=0, set to 0
df.loc[(df['children'].isna()) & (df['adults'] == 0), 'children'] = 0

# 3. Remove duplicate rows (keeping first occurrence)
initial_rows = len(df)
df = df.drop_duplicates()
final_rows = len(df)
if initial_rows != final_rows:
    print(f"Removed {initial_rows - final_rows} duplicate rows")

# 4. Correct obvious typos and inconsistent categories
# Standardize country codes (uppercase)
if 'country' in df.columns:
    df['country'] = df['country'].str.upper()

# Standardize meal categories
meal_mapping = {
    'Undefined': 'SC',
    'SC': 'SC',
    'HB': 'HB',
    'FB': 'FB',
    'Breakfast': 'HB'  # Assuming Breakfast is same as HB
}
if 'meal' in df.columns:
    df['meal'] = df['meal'].replace(meal_mapping)

# Standardize market_segment categories
segment_mapping = {
    'Online TA': 'Online TA',
    'Offline TA/TO': 'Offline TA/TO',
    'Direct': 'Direct',
    'Corporate': 'Corporate',
    'Groups': 'Groups',
    'Complementary': 'Complementary',
    'Aviation': 'Aviation',
    'Undefined': 'Direct'  # Assuming undefined is direct
}
if 'market_segment' in df.columns:
    df['market_segment'] = df['market_segment'].replace(segment_mapping)

# Standardize distribution_channel
channel_mapping = {
    'TA/TO': 'TA/TO',
    'Direct': 'Direct',
    'Corporate': 'Corporate',
    'GDS': 'GDS',
    'Undefined': 'Direct'  # Assuming undefined is direct
}
if 'distribution_channel' in df.columns:
    df['distribution_channel'] = df['distribution_channel'].replace(channel_mapping)

# Standardize deposit_type
deposit_mapping = {
    'No Deposit': 'No Deposit',
    'Refundable': 'Refundable',
    'Non Refund': 'Non Refund'
}
if 'deposit_type' in df.columns:
    df['deposit_type'] = df['deposit_type'].replace(deposit_mapping)

# Standardize customer_type
customer_mapping = {
    'Transient': 'Transient',
    'Contract': 'Contract',
    'Transient-Party': 'Transient-Party',
    'Group': 'Group'
}
if 'customer_type' in df.columns:
    df['customer_type'] = df['customer_type'].replace(customer_mapping)

# 5. Ensure numeric columns contain only numeric values
numeric_cols = ['lead_time', 'arrival_date_year', 'arrival_date_week_number',
                'arrival_date_day_of_month', 'stays_in_weekend_nights',
                'stays_in_week_nights', 'adults', 'children', 'babies',
                'is_repeated_guest', 'previous_cancellations',
                'previous_bookings_not_canceled', 'booking_changes',
                'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
                'total_of_special_requests']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 6. Final check for impossible values
# Ensure no negative values in numeric columns
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].clip(lower=0)

# Ensure adults + children + babies > 0 (at least one person per booking)
df = df[(df['adults'] + df['children'] + df['babies']) > 0]

# Report final row count
print(f"Final row count: {len(df)} (original: {initial_rows})")
if len(df) < initial_rows:
    print(f"Intentionally removed {initial_rows - len(df)} rows due to data quality issues")

# Save cleaned dataset
df.to_csv('hotel_bookings_cleaned.csv', index=False)