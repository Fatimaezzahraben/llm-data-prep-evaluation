import pandas as pd
import numpy as np
import difflib
import re

# Disguised missing values handling
disguised_missing = ["NA", "N/A", "unknown", "", " ", "nan", "NaN"]

# 1. hotel - clean typos and whitespace
df['hotel'] = df['hotel'].astype(str).str.strip()
valid_hotels = ['City Hotel', 'Resort Hotel']
hotel_mapping = {
    'CITY hotel': 'City Hotel',
    '  City Hotel ': 'City Hotel',
    '  Resort Hotel ': 'Resort Hotel'
}
df['hotel'] = df['hotel'].replace(hotel_mapping)
def fuzzy_hotel(val):
    if pd.isna(val) or str(val).lower() in ['nan', 'none']:
        return np.nan
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_hotels]:
        return next(v for v in valid_hotels if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_hotels], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_hotels if v.lower() == match[0])
    return np.nan
df['hotel'] = df['hotel'].apply(fuzzy_hotel)
df['hotel'] = df['hotel'].fillna(df['hotel'].mode()[0])

# 2. is_canceled - already numeric, no missing

# 3. lead_time - convert to numeric, impute missing with median
df['lead_time'] = pd.to_numeric(df['lead_time'].astype(str).str.replace(r'[^0-9]', '', regex=True), errors='coerce')
df['lead_time'] = df['lead_time'].fillna(df['lead_time'].median())

# 4. arrival_date_year - already numeric, no missing

# 5. arrival_date_month - standardize month names
month_mapping = {
    'January': 'January', 'February': 'February', 'March': 'March', 'April': 'April',
    'May': 'May', 'June': 'June', 'July': 'July', 'August': 'August',
    'September': 'September', 'October': 'October', 'November': 'November', 'December': 'December',
    'Jan': 'January', 'Feb': 'February', 'Mar': 'March', 'Apr': 'April',
    'Jun': 'June', 'Jul': 'July', 'Aug': 'August', 'Sep': 'September',
    'Oct': 'October', 'Nov': 'November', 'Dec': 'December'
}
df['arrival_date_month'] = df['arrival_date_month'].astype(str).str.strip().replace(month_mapping)
df['arrival_date_month'] = df['arrival_date_month'].fillna(df['arrival_date_month'].mode()[0])

# 6. arrival_date_week_number - already numeric, no missing
# 7. arrival_date_day_of_month - already numeric, no missing
# 8. stays_in_weekend_nights - already numeric, no missing

# 9. stays_in_week_nights - cap extreme outliers and impute
df['stays_in_week_nights'] = pd.to_numeric(df['stays_in_week_nights'], errors='coerce')
is_outlier = (df['stays_in_week_nights'] > 30) | (df['stays_in_week_nights'] < 0)
df.loc[is_outlier, 'stays_in_week_nights'] = np.nan
group_val = df.groupby(['hotel', 'customer_type'])['stays_in_week_nights'].transform(
    lambda s: s.median() if not pd.isna(s.median()) else np.nan
)
df['stays_in_week_nights'] = df['stays_in_week_nights'].fillna(group_val)
df['stays_in_week_nights'] = df['stays_in_week_nights'].fillna(df['stays_in_week_nights'].median())

# 10. adults - clean typos and convert to numeric
adults_mapping = {'2O': '2', '1O': '1', '0': '0'}
df['adults'] = df['adults'].astype(str).replace(adults_mapping)
df['adults'] = pd.to_numeric(df['adults'].astype(str).str.replace(r'[^0-9]', '', regex=True), errors='coerce')
is_outlier = (df['adults'] > 10) | (df['adults'] < 0)
df.loc[is_outlier, 'adults'] = np.nan
df['adults'] = df['adults'].fillna(df['adults'].median())

# 11. children - handle missing and disguised missing
df['children'] = df['children'].astype(str).replace(disguised_missing, np.nan)
df['children'] = pd.to_numeric(df['children'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
is_outlier = (df['children'] > 10) | (df['children'] < 0)
df.loc[is_outlier, 'children'] = np.nan
df['children'] = df['children'].fillna(df['children'].median())

# 12. babies - cap extreme outliers
df['babies'] = pd.to_numeric(df['babies'], errors='coerce')
is_outlier = (df['babies'] > 4) | (df['babies'] < 0)
df.loc[is_outlier, 'babies'] = np.nan
df['babies'] = df['babies'].fillna(0)

# 13. meal - handle missing and standardize
df['meal'] = df['meal'].astype(str).replace(disguised_missing, np.nan)
meal_mapping = {
    ' ': np.nan, 'Undefined': np.nan, 'No Meal': 'SC',
    'Breakfast': 'BB', 'Half Board': 'HB', 'Full Board': 'FB'
}
df['meal'] = df['meal'].replace(meal_mapping)
valid_meals = ['BB', 'HB', 'SC', 'FB']
def fuzzy_meal(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().upper()
    if val_norm in valid_meals:
        return val_norm
    match = difflib.get_close_matches(val_norm, valid_meals, n=1, cutoff=0.6)
    if match:
        return match[0]
    return np.nan
df['meal'] = df['meal'].apply(fuzzy_meal)
df['meal'] = df['meal'].fillna(df['meal'].mode()[0])

# 14. country - standardize case and handle missing with improved group-aware imputation
df['country'] = df['country'].astype(str).str.strip().str.upper()
df['country'] = df['country'].replace(disguised_missing, np.nan)

# First try imputation by market_segment and hotel (stronger relationship)
group_val = df.groupby(['market_segment', 'hotel'])['country'].transform(
    lambda s: s.mode()[0] if not s.mode().empty else np.nan
)
df['country'] = df['country'].fillna(group_val)

# Then try imputation by agent (booking agent often serves specific countries)
group_val = df.groupby('agent')['country'].transform(
    lambda s: s.mode()[0] if not s.mode().empty else np.nan
)
df['country'] = df['country'].fillna(group_val)

# Fallback to global mode
df['country'] = df['country'].fillna(df['country'].mode()[0])

# 15. market_segment - standardize and handle missing with improved imputation
df['market_segment'] = df['market_segment'].astype(str).str.strip()
segment_mapping = {
    'Online TA': 'Online TA',
    'Offline TA/TO': 'Offline TA/TO',
    'Groups': 'Groups',
    'Direct': 'Direct',
    'Corporate': 'Corporate',
    'Complementary': 'Complementary',
    'Aviation': 'Aviation',
    'Undefined': np.nan,
    ' ': np.nan
}
df['market_segment'] = df['market_segment'].replace(segment_mapping)
valid_segments = ['Online TA', 'Offline TA/TO', 'Groups', 'Direct', 'Corporate', 'Complementary', 'Aviation']
def fuzzy_segment(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_segments]:
        return next(v for v in valid_segments if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_segments], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_segments if v.lower() == match[0])
    return np.nan
df['market_segment'] = df['market_segment'].apply(fuzzy_segment)

# Improved group-aware imputation
group_val = df.groupby(['hotel', 'distribution_channel'])['market_segment'].transform(
    lambda s: s.mode()[0] if not s.mode().empty else np.nan
)
df['market_segment'] = df['market_segment'].fillna(group_val)

# Fallback to global mode
df['market_segment'] = df['market_segment'].fillna(df['market_segment'].mode()[0])

# 16. distribution_channel - standardize
df['distribution_channel'] = df['distribution_channel'].astype(str).str.strip()
channel_mapping = {
    'TA/TO': 'TA/TO',
    'Direct': 'Direct',
    'Corporate': 'Corporate',
    'GDS': 'GDS',
    'Undefined': 'Undefined',
    ' ': np.nan
}
df['distribution_channel'] = df['distribution_channel'].replace(channel_mapping)
df['distribution_channel'] = df['distribution_channel'].fillna(df['distribution_channel'].mode()[0])

# 17. is_repeated_guest - already numeric, no missing
# 18. previous_cancellations - already numeric, no missing
# 19. previous_bookings_not_canceled - already numeric, no missing

# 20. reserved_room_type - standardize
df['reserved_room_type'] = df['reserved_room_type'].astype(str).str.strip().str.upper()
df['reserved_room_type'] = df['reserved_room_type'].replace(disguised_missing, np.nan)
df['reserved_room_type'] = df['reserved_room_type'].fillna(df['reserved_room_type'].mode()[0])

# 21. assigned_room_type - standardize
df['assigned_room_type'] = df['assigned_room_type'].astype(str).str.strip().str.upper()
df['assigned_room_type'] = df['assigned_room_type'].replace(disguised_missing, np.nan)
df['assigned_room_type'] = df['assigned_room_type'].fillna(df['assigned_room_type'].mode()[0])

# 22. booking_changes - already numeric, no missing

# 23. deposit_type - clean typos and whitespace
df['deposit_type'] = df['deposit_type'].astype(str).str.strip()
deposit_mapping = {
    '  No Deposit ': 'No Deposit',
    'NO deposit': 'No Deposit',
    '  Non Refund ': 'Non Refund',
    'Refundable': 'Non Refund',
    'Non Ref': 'Non Refund',
    ' ': np.nan,
    'oN Deposit': 'No Deposit'  # Added from sample data
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
    return np.nan
df['deposit_type'] = df['deposit_type'].apply(fuzzy_deposit)
df['deposit_type'] = df['deposit_type'].fillna(df['deposit_type'].mode()[0])

# 24. agent - convert to numeric, impute missing with improved group-aware approach
df['agent'] = df['agent'].astype(str).replace(disguised_missing, np.nan)
df['agent'] = pd.to_numeric(df['agent'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

# First try imputation by hotel and market_segment
group_val = df.groupby(['hotel', 'market_segment'])['agent'].transform(
    lambda s: s.mode()[0] if not s.mode().empty else np.nan
)
df['agent'] = df['agent'].fillna(group_val)

# Then try imputation by country (agents often serve specific countries)
group_val = df.groupby('country')['agent'].transform(
    lambda s: s.mode()[0] if not s.mode().empty else np.nan
)
df['agent'] = df['agent'].fillna(group_val)

# Fallback to global mode
df['agent'] = df['agent'].fillna(df['agent'].mode()[0])

# 25. company - high missing rate, impute only with group mode
df['company'] = pd.to_numeric(df['company'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
group_val = df.groupby(['hotel', 'market_segment'])['company'].transform(
    lambda s: s.mode()[0] if not s.mode().empty else np.nan
)
df['company'] = df['company'].fillna(group_val)
# Leave remaining as NaN (high missing rate)

# 26. days_in_waiting_list - cap extreme outlier
df['days_in_waiting_list'] = pd.to_numeric(df['days_in_waiting_list'], errors='coerce')
is_outlier = df['days_in_waiting_list'] > 365
df.loc[is_outlier, 'days_in_waiting_list'] = np.nan
df['days_in_waiting_list'] = df['days_in_waiting_list'].fillna(df['days_in_waiting_list'].median())

# 27. customer_type - clean typos
df['customer_type'] = df['customer_type'].astype(str).str.strip()
customer_mapping = {
    '  Transient ': 'Transient',
    'Transient-Party': 'Transient-Party',
    'Contract': 'Contract',
    'Group': 'Group',
    ' ': np.nan
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
    return np.nan
df['customer_type'] = df['customer_type'].apply(fuzzy_customer)
df['customer_type'] = df['customer_type'].fillna(df['customer_type'].mode()[0])

# 28. adr - cap extreme outliers with more conservative bounds
df['adr'] = pd.to_numeric(df['adr'], errors='coerce')
is_outlier = (df['adr'] < -100) | (df['adr'] > 1000)
df.loc[is_outlier, 'adr'] = np.nan

# Improved group-aware imputation
group_val = df.groupby(['hotel', 'customer_type', 'reserved_room_type'])['adr'].transform(
    lambda s: s.median() if not pd.isna(s.median()) else np.nan
)
df['adr'] = df['adr'].fillna(group_val)

# Fallback to global median
df['adr'] = df['adr'].fillna(df['adr'].median())

# 29. required_car_parking_spaces - already numeric, no missing
# 30. total_of_special_requests - already numeric, no missing

# 31. reservation_status - standardize
df['reservation_status'] = df['reservation_status'].astype(str).str.strip()
status_mapping = {
    'Check-Out': 'Check-Out',
    'Canceled': 'Canceled',
    'No-Show': 'No-Show',
    'Check Out': 'Check-Out',
    'Cancel': 'Canceled'
}
df['reservation_status'] = df['reservation_status'].replace(status_mapping)
df['reservation_status'] = df['reservation_status'].fillna(df['reservation_status'].mode()[0])

# 32. reservation_status_date - improved date parsing with Unix timestamp handling
df['reservation_status_date'] = df['reservation_status_date'].astype(str)
parsed = pd.Series(pd.NaT, index=df.index)
raw = df['reservation_status_date']

# First handle Unix timestamps (numeric values)
numeric_mask = raw.str.match(r'^\d+$', na=False)
if numeric_mask.any():
    unix_dates = pd.to_numeric(raw[numeric_mask], errors='coerce')
    parsed.loc[numeric_mask] = pd.to_datetime(unix_dates, unit='s', errors='coerce')

# Then handle regular date formats
date_formats = [
    '%Y-%m-%d', '%d/%m/%Y', '%d-%b-%Y', '%d-%m-%Y', '%Y/%m/%d',
    '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y', '%m/%d/%Y'
]

for fmt in date_formats:
    still_missing = parsed.isna()
    parsed.loc[still_missing] = pd.to_datetime(raw[still_missing], format=fmt, errors='coerce')

# Handle month names in different languages
month_names = {
    'January': 'January', 'February': 'February', 'March': 'March', 'April': 'April',
    'May': 'May', 'June': 'June', 'July': 'July', 'August': 'August',
    'September': 'September', 'October': 'October', 'November': 'November', 'December': 'December',
    'Jan': 'January', 'Feb': 'February', 'Mar': 'March', 'Apr': 'April',
    'Jun': 'June', 'Jul': 'July', 'Aug': 'August', 'Sep': 'September',
    'Oct': 'October', 'Nov': 'November', 'Dec': 'December'
}

# For remaining unparsed dates, try to extract components
still_missing = parsed.isna()
if still_missing.any():
    for idx in raw[still_missing].index:
        val = str(raw[idx]).strip()
        if not val or val in disguised_missing:
            continue

        # Try to extract components
        parts = re.split(r'[/\-,\s]+', val)
        if len(parts) >= 3:
            try:
                # Handle day-month-year or month-day-year formats
                if parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                    day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        parsed_date = pd.to_datetime(f"{year}-{month:02d}-{day:02d}", errors='coerce')
                        if not pd.isna(parsed_date):
                            parsed[idx] = parsed_date
                    elif 1 <= day <= 12 and 1 <= month <= 31:  # Try month-day-year
                        parsed_date = pd.to_datetime(f"{year}-{day:02d}-{month:02d}", errors='coerce')
                        if not pd.isna(parsed_date):
                            parsed[idx] = parsed_date
                else:
                    # Handle month names
                    day = int(parts[0]) if parts[0].isdigit() else None
                    month = parts[1]
                    year = int(parts[2]) if parts[2].isdigit() else None

                    if day and year and month in month_names:
                        month = month_names[month]
                        parsed_date = pd.to_datetime(f"{year}-{month}-{day}", errors='coerce')
                        if not pd.isna(parsed_date):
                            parsed[idx] = parsed_date
            except:
                continue

# Convert to consistent format
df['reservation_status_date'] = parsed.dt.strftime('%Y-%m-%d')
df['reservation_status_date'] = df['reservation_status_date'].replace('NaT', np.nan)

# Improved group-aware imputation for dates
group_val = df.groupby(['hotel', 'arrival_date_year', 'arrival_date_month', 'reservation_status'])['reservation_status_date'].transform(
    lambda s: s.mode()[0] if not s.mode().empty else np.nan
)
df['reservation_status_date'] = df['reservation_status_date'].fillna(group_val)

# Fallback to global mode
df['reservation_status_date'] = df['reservation_status_date'].fillna(df['reservation_status_date'].mode()[0])

df