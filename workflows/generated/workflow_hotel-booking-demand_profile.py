import pandas as pd
import re
import numpy as np

# Load the data
data = pd.read_csv('hotel-booking-demand.csv')

# Define functions for handling missing values, date standardization, category harmonization and outlier detection
def handle_missing(s):
    # Replace all types of missing values with NaN
    s = s.str.replace(['NA', 'N/A', 'unknown', '', '\s+'], np.nan)
    return s

def standardize_date(s, format='%Y-%m-%d'):
    # Standardize date formats
    s = pd.to_datetime(s, infer_datetime_format=True).dt.strftime(format)
    return s

def harmonize_categories(s, categories):
    # Harmonize inconsistent categories
    s = s.replace({k: v for k, v in categories.items()})
    return s

def outlier_detection(s, q1=0.25, q3=0.75):
    # Detect potential outliers by calculating the IQR and flagging values outside the range (q1 - 1.5 * IQR, q3 + 1.5 * IQR)
    Q1 = s.quantile(q1)
    Q3 = s.quantile(q3)
    IQR = Q3 - Q1
    return (s < (Q1 - 1.5 * IQR)) | (s > (Q3 + 1.5 * IQR))

# Apply the functions to each column
data['hotel'] = harmonize_categories(data['hotel'], {'City Hotel': 'city_hotel', 'Resort Hotel': 'resort_hotel', 'CITY hotel': 'city_hotel', '  City Hotel ': 'city_hotel', '  Resort Hotel ': 'resort_hotel', 'CITY hotel': 'city_hotel'})
data['lead_time'] = data['lead_time'].apply(lambda x: int(x) if x.isdigit() else np.nan)
data['arrival_date_year'] = standardize_date(data['arrival_date_year'])
data['arrival_date_month'] = harmonize_categories(data['arrival_date_month'], {'August': '08', 'July': '07', 'May': '05', 'October': '10', 'April': '04'})
data['arrival_date_week_number'] = data['arrival_date_week_number'].apply(lambda x: int(x))
data['arrival_date_day_of_month'] = data['arrival_date_day_of_month'].apply(lambda x: int(x))
data['stays_in_weekend_nights'] = data['stays_in_weekend_nights'].apply(lambda x: int(x) if not isinstance(x, str) else np.nan)
data['stays_in_week_nights'] = data['stays_in_week_nights'].apply(outlier_detection, q1=0.25, q3=0.75)
data['adults'] = harmonize_categories(data['adults'], {'2': '2', '1': '1', '3': '3', '2O': '2', '1O': '1'})
data['children'] = data['children'].apply(lambda x: int(x.strip('."')) if re.match(r'\d+\.\d+', x) else np.nan)
data['babies'] = data['babies'].apply(outlier_detection, q1=0.25, q3=0.75)
data['meal'] = harmonize_categories(data['meal'], {'BB': 'BB', 'HB': 'HB', 'SC': 'SC', 'unknown': 'Unknown'})
data['country'] = harmonize_categories(data['country'], {'PRT': 'Portugal', 'GBR': 'United Kingdom', 'FRA': 'France', 'ESP': 'Spain', 'DEU': 'Germany'})
data['market_segment'] = harmonize_categories(data['market_segment'], {'Online TA': 'OnlineTA', 'Offline TA/TO': 'OfflineTATO', 'Groups': 'Groups', 'Direct': 'Direct', 'Corporate': 'Corporate'})
data['distribution_channel'] = harmonize_categories(data['distribution_channel'], {'TA/TO': 'TA_TO', 'Direct': 'Direct', 'Corporate': 'Corporate', 'GDS': 'GDS', 'Undefined': 'Undefined'})
data['is_repeated_guest'] = data['is_repeated_guest'].apply(outlier_detection, q1=0.25, q3=0.75)
data['previous_cancellations'] = data['previous_cancellations'].apply(outlier_detection, q1=0.25, q3=0.75)
data['previous_bookings_not_canceled'] = data['previous_bookings_not_canceled'].apply(outlier_detection, q1=0.25, q3=0.75)
data['reserved_room_type'] = harmonize_categories(data['reserved_room_type'], {'A': 'A', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G'})
data['assigned_room_type'] = harmonize_categories(data['assigned_room_type'], {'A': 'A', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G'})
data['booking_changes'] = data['booking_changes'].apply(outlier_detection, q1=0.25, q3=0.75)
data['deposit_type'] = harmonize_categories(data['deposit_type'], {'No Deposit': 'NoDeposit', 'Non Refund': 'NonRefund', '  No Deposit ': 'NoDeposit', 'NO deposit': 'NoDeposit', '  Non Refund ': 'NonRefund'})
data['agent'] = harmonize_categories(data['agent'], {'9.0': 'Agent_9', '10.0': 'Agent_10', '11.0': 'Agent_11', '12.0': 'Agent_12', '13.0': 'Agent_13'})
data['reservation_status'] = harmonize_categories(data['reservation_status'], {'Check-Out': 'CheckOut', 'Canceled': 'Canceled', 'No-Show': 'NoShow'})