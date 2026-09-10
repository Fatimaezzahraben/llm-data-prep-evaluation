import pandas as pd
import numpy as np
import re

# Load the dataset
df = pd.read_csv('titanic.csv')

# Define a function to handle inconsistent date formats
def standardize_dates(series):
    return pd.to_datetime(series, infer_datetime_format=True)

# Define a function to handle inconsistent categories and typos
def harmonize_categories(series, n_unique_threshold=1000):
    # Find and remove duplicates
    series = series.drop_duplicates()

    # Find and correct common typos
    series = series.str.replace({
        'male': 'm',
        'female': 'f',
        'mle': 'm',
        '  male ': 'm',
        'FEMAle': 'm',
        'unknown': np.nan
    })

    # Find and correct categories with too many unique values
    if len(series.unique()) > n_unique_threshold:
        series = series.value_counts().index[:n_unique_threshold]
        series = series.to_series().reindex(series.index.union(series.dropna()))

    return series

# Define a function to handle numeric columns and cap values outside the observed range
def cap_values(series, min_val=None, max_val=None):
    series = series.astype(float)

    if min_val is not None:
        series = series.clip(min_val, None)

    if max_val is not None:
        series = series.clip(None, max_val)

    return series

# Define a function to handle missing values using mode for categorical and median for numeric
def impute_missing_values(series):
    if series.dtype == 'object':
        series = series.fillna(series.mode()[0])
    else:
        series = series.fillna(series.median())

    return series

# Apply the cleaning functions to each column
df['Name'] = harmonize_categories(df['Name'])
df['Sex'] = harmonize_categories(df['Sex'])
df['Age'] = cap_values(df['Age'], min_val=0, max_val=100)
df['Age'] = impute_missing_values(df['Age'])
df['SibSp'] = cap_values(df['SibSp'])
df['Parch'] = cap_values(df['Parch'])
df['Fare'] = cap_values(df['Fare'], min_val=0, max_val=520)
df['Fare'] = impute_missing_values(df['Fare'])
df['Cabin'] = impute_missing_values(df['Cabin'])
df['Embarked'] = harmonize_categories(df['Embarked'])

# Standardize dates
df['Embarked'] = standardize_dates(df['Embarked'])

# Fill missing Embarked values using mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Fill missing Ticket values using mode, but only after grouping by Ticket
df['Ticket'] = df.groupby('Ticket')['Ticket'].transform(lambda x: x.fillna(x.mode()[0]))

# Save the cleaned dataset
df.to_csv('cleaned_titanic.csv', index=False)