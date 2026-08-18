import pandas as pd
import numpy as np
import difflib

# Disguised missing values handling
disguised_missing = ["NA", "N/A", "unknown", "", " "]

# 1. hotel - clean typos and whitespace
df['hotel'] = df['hotel'].astype(str).str.strip()
valid_hotels = ['City Hotel', 'Resort Hotel']
hotel_mapping = {
    'CITY hotel': 'City Hotel',
    '  City Hotel ': 'City Hotel',
    '  Resort Hotel ': 'Resort Hotel'
}
df['hotel'] = df['hotel'].replace(hotel_mapping)
# Fuzzy match remaining typos
def fuzzy_hotel(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_hotels]:
        return next(v for v in valid_hotels if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_hotels], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_hotels if v.lower() == match[0])
    return val
df['hotel'] = df['hotel'].apply(fuzzy_hotel)

# 2. is_canceled - already numeric, no missing

# 3. lead_time - convert to numeric, impute missing with median
df['lead_time'] = pd.to_numeric(df['lead_time'].astype(str).str.replace(r'[^0-9]', '', regex=True), errors='coerce')
df['lead_time'] = df['lead_time'].fillna(df['lead_time'].median())

# 4. arrival_date_year - already numeric, no missing

# 5. arrival_date_month - standardize month names
month_mapping = {
    'January': 'January', 'February': 'February', 'March': 'March', 'April': 'April',
    'May': 'May', 'June': 'June', 'July': 'July', 'August': 'August',
    'September': 'September', 'October': 'October', 'November': 'November', 'December': 'December'
}
df['arrival_date_month'] = df['arrival_date_month'].astype(str).str.strip().replace(month_mapping)

# 6. arrival_date_week_number - already numeric, no missing
# 7. arrival_date_day_of_month - already numeric, no missing

# 8. stays_in_weekend_nights - already numeric, no missing
# 9. stays_in_week_nights - cap extreme outlier (999)
df['stays_in_week_nights'] = pd.to_numeric(df['stays_in_week_nights'], errors='coerce')
is_outlier = df['stays_in_week_nights'] > 30  # plausible max
df.loc[is_outlier, 'stays_in_week_nights'] = np.nan
df['stays_in_week_nights'] = df['stays_in_week_nights'].fillna(df['stays_in_week_nights'].median())

# 10. adults - clean typos and convert to numeric
adults_mapping = {'2O': '2', '1O': '1'}
df['adults'] = df['adults'].astype(str).replace(adults_mapping)
df['adults'] = pd.to_numeric(df['adults'].astype(str).str.replace(r'[^0-9]', '', regex=True), errors='coerce')
df['adults'] = df['adults'].fillna(df['adults'].median())

# 11. children - handle missing and disguised missing
df['children'] = df['children'].astype(str).replace(disguised_missing, np.nan)
df['children'] = pd.to_numeric(df['children'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
df['children'] = df['children'].fillna(df['children'].median())

# 12. babies - already numeric, no missing

# 13. meal - handle missing and standardize
df['meal'] = df['meal'].astype(str).replace(disguised_missing, np.nan)
meal_mapping = {' ': np.nan}
df['meal'] = df['meal'].replace(meal_mapping)
df['meal'] = df['meal'].fillna(df['meal'].mode()[0] if not df['meal'].mode().empty else 'BB')

# 14. country - standardize case and handle missing
df['country'] = df['country'].astype(str).str.strip().str.upper()
df['country'] = df['country'].replace(disguised_missing, np.nan)
df['country'] = df['country'].fillna(df['country'].mode()[0] if not df['country'].mode().empty else 'PRT')

# 15. market_segment - standardize
df['market_segment'] = df['market_segment'].astype(str).str.strip()
segment_mapping = {
    'Online TA': 'Online TA',
    'Offline TA/TO': 'Offline TA/TO',
    'Groups': 'Groups',
    'Direct': 'Direct',
    'Corporate': 'Corporate'
}
df['market_segment'] = df['market_segment'].replace(segment_mapping)
df['market_segment'] = df['market_segment'].fillna(df['market_segment'].mode()[0] if not df['market_segment'].mode().empty else 'Online TA')

# 16. distribution_channel - standardize
df['distribution_channel'] = df['distribution_channel'].astype(str).str.strip()
channel_mapping = {
    'TA/TO': 'TA/TO',
    'Direct': 'Direct',
    'Corporate': 'Corporate',
    'GDS': 'GDS',
    'Undefined': 'Undefined'
}
df['distribution_channel'] = df['distribution_channel'].replace(channel_mapping)

# 17. is_repeated_guest - already numeric, no missing
# 18. previous_cancellations - already numeric, no missing
# 19. previous_bookings_not_canceled - already numeric, no missing

# 20. reserved_room_type - standardize
df['reserved_room_type'] = df['reserved_room_type'].astype(str).str.strip()

# 21. assigned_room_type - standardize
df['assigned_room_type'] = df['assigned_room_type'].astype(str).str.strip()

# 22. booking_changes - already numeric, no missing

# 23. deposit_type - clean typos and whitespace
df['deposit_type'] = df['deposit_type'].astype(str).str.strip()
deposit_mapping = {
    '  No Deposit ': 'No Deposit',
    'NO deposit': 'No Deposit',
    '  Non Refund ': 'Non Refund'
}
df['deposit_type'] = df['deposit_type'].replace(deposit_mapping)
valid_deposits = ['No Deposit', 'Non Refund']
def fuzzy_deposit(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_deposits]:
        return next(v for v in valid_deposits if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_deposits], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_deposits if v.lower() == match[0])
    return val
df['deposit_type'] = df['deposit_type'].apply(fuzzy_deposit)

# 24. agent - convert to numeric, impute missing
df['agent'] = df['agent'].astype(str).replace(disguised_missing, np.nan)
df['agent'] = pd.to_numeric(df['agent'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
# Group-aware imputation: agent is often constant per hotel+market_segment
group_val = df.groupby(['hotel', 'market_segment'])['agent'].transform(
    lambda s: s.mode()[0] if not s.mode().empty else np.nan
)
df['agent'] = df['agent'].fillna(group_val)
df['agent'] = df['agent'].fillna(df['agent'].median())

# 25. company - high missing rate, impute only with group mode
df['company'] = pd.to_numeric(df['company'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
group_val = df.groupby(['hotel', 'market_segment'])['company'].transform(
    lambda s: s.mode()[0] if not s.mode().empty else np.nan
)
df['company'] = df['company'].fillna(group_val)
# Leave remaining as NaN (high missing rate)

# 26. days_in_waiting_list - cap extreme outlier (8996)
df['days_in_waiting_list'] = pd.to_numeric(df['days_in_waiting_list'], errors='coerce')
is_outlier = df['days_in_waiting_list'] > 365  # plausible max
df.loc[is_outlier, 'days_in_waiting_list'] = np.nan
df['days_in_waiting_list'] = df['days_in_waiting_list'].fillna(df['days_in_waiting_list'].median())

# 27. customer_type - clean typos
df['customer_type'] = df['customer_type'].astype(str).str.strip()
customer_mapping = {
    '  Transient ': 'Transient'
}
df['customer_type'] = df['customer_type'].replace(customer_mapping)
valid_customers = ['Transient', 'Transient-Party', 'Contract', 'Group']
def fuzzy_customer(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_customers]:
        return next(v for v in valid_customers if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_customers], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_customers if v.lower() == match[0])
    return val
df['customer_type'] = df['customer_type'].apply(fuzzy_customer)

# 28. adr - cap extreme outliers
df['adr'] = pd.to_numeric(df['adr'], errors='coerce')
is_outlier = (df['adr'] < -100) | (df['adr'] > 1000)  # plausible range
df.loc[is_outlier, 'adr'] = np.nan
df['adr'] = df['adr'].fillna(df['adr'].median())

# 29. required_car_parking_spaces - already numeric, no missing
# 30. total_of_special_requests - already numeric, no missing

# 31. reservation_status - standardize
df['reservation_status'] = df['reservation_status'].astype(str).str.strip()
status_mapping = {
    'Check-Out': 'Check-Out',
    'Canceled': 'Canceled',
    'No-Show': 'No-Show'
}
df['reservation_status'] = df['reservation_status'].replace(status_mapping)

# 32. reservation_status_date - standardize date formats
df['reservation_status_date'] = df['reservation_status_date'].astype(str)
parsed = pd.Series(pd.NaT, index=df.index)
raw = df['reservation_status_date']
date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%d-%b-%Y']
for fmt in date_formats:
    still_missing = parsed.isna()
    parsed.loc[still_missing] = pd.to_datetime(raw[still_missing], format=fmt, errors='coerce')
df['reservation_status_date'] = parsed.dt.strftime('%Y-%m-%d')
df['reservation_status_date'] = df['reservation_status_date'].replace('NaT', np.nan)
df['reservation_status_date'] = df['reservation_status_date'].fillna(df['reservation_status_date'].mode()[0] if not df['reservation_status_date'].mode().empty else '2015-10-21')

df