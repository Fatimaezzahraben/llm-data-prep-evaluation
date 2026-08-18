import pandas as pd
import numpy as np

# Load and preprocess data
data = pd.read_csv("hotel-booking-demand.csv")

# Treat missing values consistently across columns
data.fillna(value={'hotel': '', 'country': '', 'company': 0, 'agent': ''}, inplace=True)
data['children'] = data['children'].astype('int')
data['babies'] = data['babies'].astype('int')
data['stays_in_weekend_nights'] = data['stays_in_weekend_nights'].astype('int')
data['stays_in_week_nights'] = data['stays_in_week_nights'].astype('int')
data['adults'] = data['adults'].astype('int')
data['assigned_room_type'] = data['assigned_room_type'].astype('category')
data['reserved_room_type'] = data['reserved_room_type'].astype('category')
data['deposit_type'] = data['deposit_type'].astype('category')
data['market_segment'] = data['market_segment'].astype('category')
data['distribution_channel'] = data['distribution_channel'].astype('category')
data['is_repeated_guest'] = data['is_repeated_guest'].astype('int')
data['previous_cancellations'] = data['previous_cancellations'].asttype('int')
data['previous_bookings_not_canceled'] = data['previous_bookings_not_canceled'].astype('int')
data['booking_changes'] = data['booking_changes'].astype('int')
data['required_car_parking_spaces'] = data['required_car_parking_spaces'].astype('int')
data['total_of_special_requests'] = data['total_of_special_requests'].astype('int')

# Standardize date formats
data['arrival_date_year'] = pd.to_datetime(data['arrival_date_year'], format="%Y", errors='coerce').dt.year
data['arrival_date_month'] = pd.to_datetime(data['arrival_date_month'], format="%B", errors='coerce').dt.isocalendar().month
data['arrival_date_week_number'] = data['arrival_date_day_of_month'].astype('int') + data['lead_time'].astype('int') - ((data['arrival_date_year'] - 1900) * 365.25 * 7)
data['arrival_date_day_of_week'] = data['arrival_date_day_of_month'].astype('int') % 7 + 1

# Correct typos and harmonize categories
data['hotel'] = data['hotel'].str.lower().replace({'city hotel': 'hotel', 'resort hotel': 'hotel', 'cITY hotel': 'hotel', ' city hotel ': 'hotel', ' resort hotel ': 'hotel', 'CITY hotel': 'hotel'})
data['market_segment'] = data['market_segment'].cat.rename_categories({'Online TA': 'Online Travel Agency', 'Offline TA/TO': 'Offline Travel Agency / Tour Operator', 'Groups': 'Group Booking', 'Direct': 'Direct Booking', 'Corporate': 'Corporate Booking'})
data['distribution_channel'] = data['distribution_channel'].cat.rename_categories({'TA/TO': 'Travel Agency / Tour Operator', 'Undefined': ''})
data['deposit_type'] = data['deposit_type'].cat.rename_categories({'No Deposit': 0, 'Non Refund': 1, ' No Deposit ': 0, 'NO deposit': 0, ' Non Refund ': 1})
data['agent'] = data['agent'].str.lower().replace({'9.0': 'agent_9.0', '240.0': 'agent_240.0', '1.0': 'agent_1.0', 'unknown': '', ' ': ''})
data['company'] = data['company'].fillna(0)

# Ensure numeric columns contain only numeric values and flag potential outliers
numeric_columns = ['lead_time', 'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month',
                    'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'meal', 'company',
                    'days_in_waiting_list', 'adr', 'required_car_parking_spaces', 'total_of_special_requests']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
    data.loc[outliers, col] = np.nan