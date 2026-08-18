import pandas as pd
import numpy as np

# Load the dataset if not already loaded
# df = pd.read_csv('path/to/your/dataset.csv')

# Fix inconsistent date formats and convert to datetime format
df['date'] = pd.to_datetime(df['date'].str.replace('/', '-').astype(str), dayfirst=True, infer_datetime_format=True)

# Handle missing values (replacing NaN with the median for numerical columns)
numerical_columns = df.select_dtypes(include='number').columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# Handle categorical missing values by replacing them with a constant value (e.g., '-')
categorical_columns = df.select_dtypes(include='object').columns
df[categorical_columns] = df[categorical_columns].fillna('-')

# Remove duplicate rows (assuming no duplicates exist after handling missing values)
df.drop_duplicates(inplace=True)

# Correct obvious typos and inconsistent categories
# This step might require manual inspection and correction of data, which is beyond the scope of this script.

# Ensure numeric columns contain only numeric values (no letters mixed in)
for column in numerical_columns:
    df[column] = df[column].str.replace('[^0-9.]', '', regex=True)

# Save the cleaned dataset to a new CSV file
df.to_csv('cleaned_hotel-booking-demand.csv', index=False)