import pandas as pd
import numpy as np
import difflib
import re

# Hotel column cleaning
df['hotel'] = df['hotel'].str.strip().str.title().replace({
    'City Hotel': 'City Hotel',
    'Resort Hotel': 'Resort Hotel'
})

# Lead time cleaning
df['lead_time'] = pd.to_numeric(df['lead_time'], errors='coerce')
df['lead_time'] = df['lead_time'].fillna(df['lead_time'].median())

# Adults column cleaning
def fuzzy_adults(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    valid_values = ['1', '2', '3', '1o', '2o']
    if val_norm in [v.lower() for v in valid_values]:
        return next(v for v in valid_values if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_values], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_values if v.lower() == match[0])
    return val
df['adults'] = df['adults'].apply(fuzzy_adults)
df['adults'] = pd.to_numeric(df['adults'], errors='coerce')
df['adults'] = df['adults'].fillna(df['adults'].median())

# Children column cleaning
df['children'] = df['children'].str.strip().replace({
    'unknown': np.nan,
    ' ': np.nan
})
df['children'] = pd.to_numeric(df['children'], errors='coerce')
df['children'] = df['children'].fillna(df['children'].mode().iloc[0] if not df['children'].mode().empty else 0.0)

# Meal column cleaning
df['meal'] = df['meal'].str.strip().replace({
    ' ': np.nan,
    'unknown': np.nan
})
df['meal'] = df['meal'].fillna(df['meal'].mode().iloc[0] if not df['meal'].mode().empty else 'BB')

# Country column cleaning
df['country'] = df['country'].str.strip().replace({
    ' ': np.nan
})
df['country'] = df['country'].fillna(df['country'].mode().iloc[0] if not df['country'].mode().empty else 'PRT')

# Market segment column cleaning
df['market_segment'] = df['market_segment'].str.strip().replace({
    ' ': np.nan
})
df['market_segment'] = df['market_segment'].fillna(df['market_segment'].mode().iloc[0] if not df['market_segment'].mode().empty else 'Online TA')

# Agent column cleaning
df['agent'] = df['agent'].str.strip().replace({
    ' ': np.nan
})
df['agent'] = pd.to_numeric(df['agent'], errors='coerce')
df['agent'] = df['agent'].fillna(df['agent'].median())

# Company column cleaning
df['company'] = pd.to_numeric(df['company'], errors='coerce')
df['company'] = df['company'].fillna(df['company'].median())

# Days in waiting list cleaning
df['days_in_waiting_list'] = pd.to_numeric(df['days_in_waiting_list'], errors='coerce')
df['days_in_waiting_list'] = df['days_in_waiting_list'].fillna(df['days_in_waiting_list'].median())

# ADR column cleaning
df['adr'] = pd.to_numeric(df['adr'], errors='coerce')
df['adr'] = df['adr'].fillna(df.groupby(['hotel', 'market_segment', 'reserved_room_type'])['adr'].transform(
    lambda s: s.median() if not s.median().isna() else np.nan))
df['adr'] = df['adr'].fillna(df['adr'].median())

# Deposit type cleaning
df['deposit_type'] = df['deposit_type'].str.strip().replace({
    'No Deposit': 'No Deposit',
    'Non Refund': 'Non Refund',
    '  No Deposit ': 'No Deposit',
    'NO deposit': 'No Deposit',
    '  Non Refund ': 'Non Refund'
})
df['deposit_type'] = df['deposit_type'].fillna(df['deposit_type'].mode().iloc[0] if not df['deposit_type'].mode().empty else 'No Deposit')

# Customer type cleaning
df['customer_type'] = df['customer_type'].str.strip().replace({
    '  Transient ': 'Transient'
})
df['customer_type'] = df['customer_type'].fillna(df['customer_type'].mode().iloc[0] if not df['customer_type'].mode().empty else 'Transient')

# Date columns cleaning
def clean_date(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    if val == '':
        return np.nan
    try:
        return pd.to_datetime(val, format='%Y-%m-%d', errors='coerce')
    except:
        try:
            return pd.to_datetime(val, format='%d/%m/%Y', errors='coerce')
        except:
            try:
                return pd.to_datetime(val, format='%d-%m-%Y', errors='coerce')
            except:
                try:
                    return pd.to_datetime(val, format='%m/%d/%Y', errors='coerce')
                except:
                    return np.nan

df['arrival_date_year'] = df['arrival_date_year'].astype(int)
df['arrival_date_month'] = df['arrival_date_month'].str.strip().replace({
    'December': 'December',
    'July': 'July',
    'May': 'May',
    'October': 'October',
    'April': 'April'
})
df['arrival_date_week_number'] = pd.to_numeric(df['arrival_date_week_number'], errors='coerce')
df['arrival_date_day_of_month'] = pd.to_numeric(df['arrival_date_day_of_month'], errors='coerce')

# Create a combined date column for validation
df['arrival_date'] = df.apply(lambda row: clean_date(f"{row['arrival_date_year']}-{row['arrival_date_month']}-{row['arrival_date_day_of_month']}"), axis=1)
df['arrival_date'] = df['arrival_date'].dt.strftime('%Y-%m-%d')

# Stays columns cleaning
df['stays_in_weekend_nights'] = pd.to_numeric(df['stays_in_weekend_nights'], errors='coerce')
df['stays_in_week_nights'] = pd.to_numeric(df['stays_in_week_nights'], errors='coerce')

# Numeric columns outlier handling
numeric_cols = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
                'stays_in_weekend_nights', 'stays_in_week_nights', 'babies',
                'previous_cancellations', 'previous_bookings_not_canceled',
                'booking_changes', 'agent', 'days_in_waiting_list', 'adr',
                'required_car_parking_spaces', 'total_of_special_requests']

for col in numeric_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        min_val = df[col].quantile(0.01)
        max_val = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=min_val, upper=max_val)

# Reservation status date cleaning
df['reservation_status_date'] = df['reservation_status_date'].str.strip().replace({
    '16/07/2015': '2015-07-16',
    '30-Dec-2016': '2016-12-30',
    '2016-02-26': '2016-02-26'
})
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
df['reservation_status_date'] = df['reservation_status_date'].dt.strftime('%Y-%m-%d')

# Group aware imputation for agent
group_val = df.groupby(['country', 'market_segment', 'distribution_channel'])['agent'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['agent'] = df['agent'].fillna(group_val)

# Group aware imputation for lead_time
group_val = df.groupby(['market_segment'])['lead_time'].transform(
    lambda s: s.median() if not s.median().isna() else np.nan)
df['lead_time'] = df['lead_time'].fillna(group_val)

# Final cleaning for numeric columns
for col in numeric_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())

# Final cleaning for categorical columns
categorical_cols = ['hotel', 'lead_time', 'arrival_date_month', 'adults', 'children',
                    'meal', 'country', 'market_segment', 'distribution_channel',
                    'reserved_room_type', 'assigned_room_type', 'deposit_type',
                    'agent', 'customer_type', 'reservation_status']

for col in categorical_cols:
    if col in ['lead_time', 'adults', 'children', 'agent']:
        continue
    if pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')