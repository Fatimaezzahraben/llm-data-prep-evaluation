import pandas as pd
import numpy as np
from datetime import datetime

# Load the dataset
data = pd.read_csv('hotel-booking-demand.csv')

# Standardize inconsistent date formats and convert to a consistent format (YYYY/MM/DD)
def standardize_date(x):
    if isinstance(x, str):
        try:
            return datetime.strptime(x, '%Y/%m/%d').strftime('%Y/%m/%d')
        except ValueError:
            return np.nan
    else:
        return x

data['reservation_status_date'] = data['reservation_status_date'].apply(standardize_date)

# Handle missing values per column, using a method appropriate to its type
data.fillna({
    'hotel': data['hotel'].mode()[0],
    'is_canceled': 0,
    'lead_time': 0,
    'arrival_date_year': 0,
    'arrival_date_month': 'January',
    'arrival_date_week_number': np.nan,
    'arrival_date_day_of_month': 1,
    'stays_in_weekend_nights': 0,
    'stays_in_week_nights': 0,
    'adults': 2,
    'children': 0,
    'babies': 0,
    'meal': 'unknown',
    'country': '',
    'market_segment': 'Direct',
    'distribution_channel': 'Direct',
    'is_repeated_guest': 0,
    'previous_cancellations': 0,
    'previous_bookings_not_canceled': 0,
    'reserved_room_type': 'C',
    'assigned_room_type': 'C',
    'booking_changes': 0,
    'deposit_type': 'No Deposit',
    'agent': np.nan,
    'company': np.nan,
    'days_in_waiting_list': 0,
    'customer_type': 'Transient',
    'total_of_special_requests': 0,
    'reservation_status': 'Check-Out',
}, inplace=True)

# Remove duplicate rows if any
data.drop_duplicates(inplace=True)

# Fix typos and inconsistent categories in text columns
data['hotel'] = data['hotel'].str.replace('Resort', 'Resort Hotel').str.replace('RESOrt', 'Resort Hotel')
data['country'] = data['country'].replace(['', np.nan], 'Unknown')
data['market_segment'] = data['market_segment'].replace(['Direct', 'direct'], 'Direct')
data['distribution_channel'] = data['distribution_channel'].replace(['Direct', 'direct'], 'Direct')
data['customer_type'] = data['customer_type'].replace(['Transient', 'transient'], 'Transient')
data['customer_type'] = data['customer_type'].replace(['CONtract'], 'Contract')
data['deposit_type'] = data['deposit_type'].str.strip()

# Ensure numeric columns contain only numeric values, converting or flagging any non-numeric entries found
def handle_non_numeric(x):
    try:
        return float(x)
    except ValueError:
        return np.nan

data['stays_in_weekend_nights'] = data['stays_in_weekend_nights'].apply(handle_non_numeric)
data['stays_in_week_nights'] = data['stays_in_week_nights'].apply(handle_non_numeric)
data['adults'] = data['adults'].apply(handle_non_numeric)
data['children'] = data['children'].apply(handle_non_numeric)
data['babies'] = data['babies'].apply(handle_non_numeric)
data['meal'] = data['meal'].replace({'unknown': np.nan, 'HB': 0, 'BB': 1})
data['is_canceled'] = data['is_canceled'].astype(int)
data['company'] = data['company'].apply(handle_non_numeric)
data['agent'] = data['agent'].apply(handle_non_numeric)
data['days_in_waiting_list'] = data['days_in_waiting_list'].astype(int)
data['total_of_special_requests'] = data['total_of_special_requests'].astype(int)
data['is_repeated_guest'] = data['is_repeated_guest'].astype(int)
data['previous_cancellations'] = data['previous_cancellations'].astype(int)
data['previous_bookings_not_canceled'] = data['previous_bookings_not_canceled'].astype(int)
data['required_car_parking_spaces'] = data['required_car_parking_spaces'].astype(int)
data['booking_changes'] = data['booking_changes'].astype(int)

# Detect and handle unrealistic outlier values in numeric columns (using IQR method)
def detect_outliers(col):
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    return (col > q3 + 1.5 * (q3 - q1)) | (col < q1 - 1.5 * (q3 - q1))

outlier_cols = ['stays_in_week_nights', 'adults', 'children', 'babies', 'company', 'agent', 'days_in_waiting_list',
                 'total_of_special_requests', 'is_repeated_guest', 'previous_cancellations',
                 'previous_bookings_not_canceled', 'required_car_parking_spaces']

for col in outlier_cols:
    data[col] = np.where(detect_outliers(data[col]), np.nan, data[col])

# Return the cleaned dataframe
cleaned_data = data.dropna()