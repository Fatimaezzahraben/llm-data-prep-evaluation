import pandas as pd

# Load the dataset
df = pd.read_csv("hotel-booking-demand.csv")

# Standardize inconsistent date formats
date_columns = ["reservation_status_date", "arrival_date"]
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()

# Handle missing values per column, using a method appropriate to its type
def handle_missing_values(series):
    if series.dtype == "object":
        return series.fillna("")  # replace missing values with an empty string for text columns
    elif series.dtype == float:
        return series.fillna(0)  # replace missing values with 0 for numeric columns
    else:
        raise ValueError(f"Unexpected data type '{series.dtype}' in column {series.name}")

# Apply the function to all columns
df = df.apply(handle_missing_values)

# Remove duplicate rows if any
duplicates = df.duplicated()
if duplicates.sum():
    print("Removing", len(duplicates[duplicates]) + 1, "duplicate rows")
    df = df.loc[~duplicates]

# Fix typos and inconsistent categories in text columns
def fix_typos(series):
    # Replace common typos based on specific cases
    series = series.str.replace("TrAnsient", "Transient")
    # Use regular expressions to handle more complex cases
    series = series.str.replace(r"Resortt Hotel|Resort Hotel", "Resort Hotel")
    return series

# Apply the function to text columns
text_columns = ["hotel", "country", "market_segment", "distribution_channel", "customer_type"]
df[text_columns] = df[text_columns].apply(fix_typos)

# Ensure numeric columns contain only numeric values, converting or flagging any non-numeric entries found
def ensure_numeric(series):
    non_numeric = series.str.contains(r"\D")  # check if contains a non-numeric character
    return series.astype("float").where(~non_numeric, pd.NaT)  # replace non-numeric values with NaT (NaN for numeric columns)

# Apply the function to numeric columns
numeric_columns = ["is_canceled", "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "meal", "is_repeated_guest",
                   "previous_cancellations", "previous_bookings_not_canceled", "booking_changes", "deposit_type", "agent", "company", "days_in_waiting_list",
                   "total_of_special_requests", "adr", "required_car_parking_spaces", "total_of_special_requests", "reserved_room_type", "assigned_room_type"]
df[numeric_columns] = df[numeric_columns].apply(ensure_numeric)

# Replace NaT (NaN for numeric columns) with 0 to avoid issues during calculations
df.replace({pd.NaT: 0}, inplace=True)

# Detect and handle unrealistic outlier values in numeric columns
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers_upper = df[(df > (Q3 + 1.5 * IQR)) | (df < (Q1 - 1.5 * IQR))].index
print("Removing", len(outliers_upper), "possible outlier rows")
df = df.drop(outliers_upper)

# Check the final row count
if abs(len(df) - 119390) > 5:
    print(f"Final row count is {len(df)}, which is not close to the original. Removed {abs(len(df) - 119390) - 5} rows.")