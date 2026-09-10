import pandas as pd
import numpy as np
import difflib
import re

# Clean hotel column - standardize whitespace and case
df['hotel'] = df['hotel'].str.strip().str.title().replace(['City Hotel', 'Resort Hotel'], ['City Hotel', 'Resort Hotel'])

# Clean lead_time - convert to numeric, handle disguised missing values
df['lead_time'] = pd.to_numeric(df['lead_time'].replace([' ', 'NA', 'N/A', 'unknown'], np.nan), errors='coerce')
df['lead_time'] = df['lead_time'].fillna(df.groupby('market_segment')['lead_time'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan))
df['lead_time'] = df['lead_time'].fillna(df['lead_time'].median())

# Clean arrival_date_month - standardize month names
month_mapping = {
    'august': 'August', 'july': 'July', 'may': 'May', 'october': 'October', 'april': 'April',
    'january': 'January', 'february': 'February', 'march': 'March', 'november': 'November', 'december': 'December',
    'jan': 'January', 'feb': 'February', 'mar': 'March', 'apr': 'April', 'jun': 'June', 'jul': 'July',
    'aug': 'August', 'sep': 'September', 'oct': 'October', 'nov': 'November', 'dec': 'December'
}
df['arrival_date_month'] = df['arrival_date_month'].str.strip().str.title().replace(month_mapping)

# Clean adults - fuzzy matching for numeric values with 'O' suffix
def clean_adults(val):
    if pd.isna(val):
        return val
    val = str(val).strip().lower()
    if val in ['1', '2', '3', '4', '5']:
        return val
    if val.endswith('o'):
        return val[:-1]
    return val
df['adults'] = df['adults'].apply(clean_adults)
df['adults'] = pd.to_numeric(df['adults'].replace([' ', 'NA', 'N/A', 'unknown'], np.nan), errors='coerce')
df['adults'] = df['adults'].fillna(df['adults'].median())

# Clean children - fuzzy matching for numeric values with 'O' suffix
def clean_children(val):
    if pd.isna(val):
        return val
    val = str(val).strip().lower()
    if val in ['0', '1', '2', '3', '4', '5']:
        return val
    if val.endswith('o'):
        return val[:-1]
    return val
df['children'] = df['children'].apply(clean_children)
df['children'] = pd.to_numeric(df['children'].replace([' ', 'NA', 'N/A', 'unknown'], np.nan), errors='coerce')
_mode = df['children'].mode(dropna=True)
fill_value = _mode.iloc[0] if not _mode.empty else 0
df['children'] = df['children'].fillna(fill_value)

# Clean meal - standardize meal types
meal_mapping = {
    'bb': 'BB', 'bb ': 'BB', 'hb': 'HB', 'hb ': 'HB', 'sc': 'SC', 'sc ': 'SC',
    ' ': np.nan, 'unknown': np.nan, 'na': np.nan, 'n/a': np.nan
}
df['meal'] = df['meal'].str.strip().replace(meal_mapping)

# Clean country - standardize country codes
country_mapping = {
    'prt': 'PRT', 'prt ': 'PRT', 'gb': 'GBR', 'gb ': 'GBR', 'fra': 'FRA', 'fra ': 'FRA',
    'esp': 'ESP', 'esp ': 'ESP', 'deu': 'DEU', 'deu ': 'DEU', 'svn': 'SVN', 'svn ': 'SVN',
    'swe': 'SWE', 'swe ': 'SWE', 'bra': 'BRA', 'bra ': 'BRA'
}
df['country'] = df['country'].str.strip().replace(country_mapping)

# Clean market_segment - standardize market segments
market_mapping = {
    'online ta': 'Online TA', 'online ta ': 'Online TA', 'offline ta/to': 'Offline TA/TO',
    'offline ta/to ': 'Offline TA/TO', 'groups': 'Groups', 'direct': 'Direct',
    'corporate': 'Corporate', ' ': np.nan, 'na': np.nan, 'n/a': np.nan
}
df['market_segment'] = df['market_segment'].str.strip().replace(market_mapping)

# Clean distribution_channel - standardize channel types
channel_mapping = {
    'ta/to': 'TA/TO', 'ta/to ': 'TA/TO', 'direct': 'Direct', 'corporate': 'Corporate',
    'gds': 'GDS', 'undefined': 'Undefined', ' ': np.nan, 'na': np.nan, 'n/a': np.nan
}
df['distribution_channel'] = df['distribution_channel'].str.strip().replace(channel_mapping)

# Clean deposit_type - standardize deposit types
deposit_mapping = {
    'no deposit': 'No Deposit', 'no deposit ': 'No Deposit', 'non refund': 'Non Refund',
    'non refund ': 'Non Refund', ' ': np.nan, 'na': np.nan, 'n/a': np.nan
}
df['deposit_type'] = df['deposit_type'].str.strip().replace(deposit_mapping)

# Clean agent - convert to numeric, handle disguised missing values
df['agent'] = pd.to_numeric(df['agent'].replace([' ', 'NA', 'N/A', 'unknown', 'oN Deposit'], np.nan), errors='coerce')
group_val = df.groupby(['country', 'market_segment', 'distribution_channel'])['agent'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['agent'] = df['agent'].fillna(group_val)
_mode = df['agent'].mode(dropna=True)
fill_value = _mode.iloc[0] if not _mode.empty else 0
df['agent'] = df['agent'].fillna(fill_value)

# Clean company - convert to numeric, handle disguised missing values
df['company'] = pd.to_numeric(df['company'].replace([' ', 'NA', 'N/A', 'unknown'], np.nan), errors='coerce')
df['company'] = df['company'].fillna(df['company'].median())

# Clean days_in_waiting_list - convert to numeric, handle outliers
df['days_in_waiting_list'] = pd.to_numeric(df['days_in_waiting_list'].replace([' ', 'NA', 'N/A', 'unknown'], np.nan), errors='coerce')
df.loc[(df['days_in_waiting_list'] < 0) | (df['days_in_waiting_list'] > 8996), 'days_in_waiting_list'] = np.nan
df['days_in_waiting_list'] = df['days_in_waiting_list'].fillna(df['days_in_waiting_list'].median())

# Clean adr - convert to numeric, handle outliers and group imputation
df['adr'] = pd.to_numeric(df['adr'].replace([' ', 'NA', 'N/A', 'unknown'], np.nan), errors='coerce')
group_val = df.groupby(['hotel', 'market_segment', 'reserved_room_type'])['adr'].transform(
    lambda s: s.median() if not s.median().isna() else np.nan)
df['adr'] = df['adr'].fillna(group_val)
df['adr'] = df['adr'].fillna(df['adr'].median())
df.loc[(df['adr'] < -493.26) | (df['adr'] > 9993.79), 'adr'] = np.nan
df['adr'] = df['adr'].fillna(df['adr'].median())

# Clean required_car_parking_spaces - convert to numeric, handle outliers
df['required_car_parking_spaces'] = pd.to_numeric(df['required_car_parking_spaces'].replace([' ', 'NA', 'N/A', 'unknown'], np.nan), errors='coerce')
df.loc[(df['required_car_parking_spaces'] < 0) | (df['required_car_parking_spaces'] > 8), 'required_car_parking_spaces'] = np.nan
df['required_car_parking_spaces'] = df['required_car_parking_spaces'].fillna(df['required_car_parking_spaces'].median())

# Clean total_of_special_requests - convert to numeric, handle outliers
df['total_of_special_requests'] = pd.to_numeric(df['total_of_special_requests'].replace([' ', 'NA', 'N/A', 'unknown'], np.nan), errors='coerce')
df.loc[(df['total_of_special_requests'] < 0) | (df['total_of_special_requests'] > 5), 'total_of_special_requests'] = np.nan
df['total_of_special_requests'] = df['total_of_special_requests'].fillna(df['total_of_special_requests'].median())

# Clean reservation_status_date - standardize date format
def clean_date(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    if not val:
        return np.nan
    try:
        if '/' in val:
            date_parts = val.split('/')
            if len(date_parts) == 3:
                return f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
        elif '-' in val:
            return val
        elif ' ' in val:
            date_parts = val.split()
            if len(date_parts) == 2:
                return f"{date_parts[1]}-{date_parts[0]}"
    except:
        pass
    return np.nan
df['reservation_status_date'] = df['reservation_status_date'].apply(clean_date)
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
df['reservation_status_date'] = df['reservation_status_date'].dt.strftime('%Y-%m-%d')

# Clean babies - convert to numeric, handle disguised missing values
df['babies'] = pd.to_numeric(df['babies'].replace([' ', 'NA', 'N/A', 'unknown'], np.nan), errors='coerce')
df['babies'] = df['babies'].fillna(df['babies'].median())

# Clean customer_type - standardize customer types
customer_mapping = {
    'transient': 'Transient', 'transient-party': 'Transient-Party', 'contract': 'Contract',
    'group': 'Group', 'transient ': 'Transient', 'transient-party ': 'Transient-Party',
    'contract ': 'Contract', 'group ': 'Group', ' ': np.nan, 'na': np.nan, 'n/a': np.nan
}
df['customer_type'] = df['customer_type'].str.strip().replace(customer_mapping)

# Clean is_canceled - convert to numeric
df['is_canceled'] = pd.to_numeric(df['is_canceled'], errors='coerce')

# Clean is_repeated_guest - convert to numeric
df['is_repeated_guest'] = pd.to_numeric(df['is_repeated_guest'], errors='coerce')

# Clean previous_cancellations - convert to numeric
df['previous_cancellations'] = pd.to_numeric(df['previous_cancellations'], errors='coerce')

# Clean previous_bookings_not_canceled - convert to numeric
df['previous_bookings_not_canceled'] = pd.to_numeric(df['previous_bookings_not_canceled'], errors='coerce')

# Clean booking_changes - convert to numeric
df['booking_changes'] = pd.to_numeric(df['booking_changes'], errors='coerce')

# Clean stays_in_weekend_nights - convert to numeric, handle outliers
df['stays_in_weekend_nights'] = pd.to_numeric(df['stays_in_weekend_nights'], errors='coerce')
df.loc[(df['stays_in_weekend_nights'] < 0) | (df['stays_in_weekend_nights'] > 19), 'stays_in_weekend_nights'] = np.nan
df['stays_in_weekend_nights'] = df['stays_in_weekend_nights'].fillna(df['stays_in_weekend_nights'].median())

# Clean stays_in_week_nights - convert to numeric, handle outliers
df['stays_in_week_nights'] = pd.to_numeric(df['stays_in_week_nights'], errors='coerce')
df.loc[(df['stays_in_week_nights'] < 0) | (df['stays_in_week_nights'] > 999), 'stays_in_week_nights'] = np.nan
df['stays_in_week_nights'] = df['stays_in_week_nights'].fillna(df['stays_in_week_nights'].median())