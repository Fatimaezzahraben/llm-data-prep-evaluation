import pandas as pd

# Load the dataset
df = pd.read_csv('hotel-booking-demand.csv')

# Fix inconsistent date formats (assuming dates are in YYYY-MM-DD format)
df['check_in'] = pd.to_datetime(df['check_in'], dayfirst=True, errors='coerce')
df['check_out'] = pd.to_datetime(df['check_out'], dayfirst=True, errors='coerce')

# Handle missing values appropriately (except for date columns)
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Remove duplicate rows if any
duplicates = df.duplicated()
if duplicates.sum() > 0:
    print("Removing", duplicates.sum(), "duplicate rows.")
    df.drop_duplicates(inplace=True)

# Correct obvious typos and inconsistent categories (based on common typo examples, adjust as needed)
# Replace "November" with "11" in month column
df['month'] = df['month'].str.replace('November', '11', regex=True)
# Replace "Januaray" with "January" in month column
df['month'] = df['month'].str.replace('Januaray', 'January', regex=True)
# Replace "February" with "02" in month column
df['month'] = df['month'].apply(lambda x: '0' + x if x.startswith('Febr') else x, axis=1)
# Replace "March" with "03" in month column
df['month'] = df['month'].apply(lambda x: '0' + x if x.startswith('Marc') else x, axis=1)

# Make sure numeric columns contain only numeric values (no letters mixed in)
for col in numeric_columns:
    df[col] = df[col].str.replace(r'\D', '', regex=True)

# Check the final row count and report if any rows are removed
if df.shape[0] != len(df):
    print("Removed", len(df) - df.shape[0], "rows with non-numeric values in numeric columns.")

# Save the cleaned dataset to a CSV file
df.to_csv('cleaned_hotel-booking-demand.csv', index=False)