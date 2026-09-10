import pandas as pd
import numpy as np
import re

# Load the dataset
df = pd.read_csv('titanic.csv')

# Define functions for handling inconsistent categories and date formats
def handle_categories(series):
    # Remove duplicates and harmonize inconsistent categories
    series = series.str.lower()
    series = series.str.replace(r'\s+', ' ', regex=True)
    series = series.str.strip()
    series = series.apply(lambda x: x.replace('mr.', 'mr.') if x.startswith('mr.') else x)
    series = series.apply(lambda x: x.replace('miss.', 'miss.') if x.startswith('miss.') else x)
    series = series.apply(lambda x: x.replace('mle', 'mrs.') if x.startswith('mle') else x)
    series = series.apply(lambda x: x.replace('female', 'female.') if x.endswith('.') else x)
    series = series.apply(lambda x: x.replace('ale', 'mrs.') if x.startswith('ale') else x)
    series = series.apply(lambda x: x.replace('  male ', 'male') if x.startswith('  male ') else x)
    series = series.apply(lambda x: x.replace('FEMAle', 'female') if x == 'FEMAle' else x)
    series = series.apply(lambda x: x.replace('male ', 'male') if x.startswith('male ') else x)
    series = series.apply(lambda x: x.replace('male', 'male.') if x.endswith('.') else x)
    series = series.apply(lambda x: x.replace('unknown', np.nan) if x == 'unknown' else x)
    series = series.apply(lambda x: x.replace('', np.nan) if x == '' else x)
    series = series.apply(lambda x: x.replace('NA', np.nan) if x == 'NA' else x)
    series = series.apply(lambda x: x.replace('N/A', np.nan) if x == 'N/A' else x)
    series = series.apply(lambda x: x.replace(' ', np.nan) if re.search(r'\s+', x) else x)
    return series

def handle_dates(series):
    # Standardize inconsistent date formats
    series = pd.to_datetime(series, errors='coerce')
    series = series.dt.strftime('%Y-%m-%d')
    return series

# Apply the functions to the relevant columns
df['Name'] = handle_categories(df['Name'])
df['Embarked'] = handle_categories(df['Embarked'])
df['Cabin'] = handle_categories(df['Cabin'])
df['Age'] = handle_dates(df['Age'])

# Handle missing values and outliers in Age and Fare columns
df.loc[(df['Age'] < 0) | (df['Age'] > 100), 'Age'] = np.nan
df.loc[(df['Fare'] < 0) | (df['Fare'] > 520), 'Fare'] = np.nan

# Fill missing values using mode for categorical columns and median for numeric columns
df.fillna({'Sex': df['Sex'].mode()[0],
           'Pclass': df['Pclass'].median(),
           'SibSp': df['SibSp'].median(),
           'Parch': df['Parch'].median(),
           'Ticket': df['Ticket'].median(),
           'Cabin': df['Cabin'].mode()[0]}, inplace=True)

# Ensure numeric columns contain only numeric values
df['Pclass'] = pd.to_numeric(df['Pclass'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['SibSp'] = pd.to_numeric(df['SibSp'], errors='coerce')
df['Parch'] = pd.to_numeric(df['Parch'], errors='coerce')
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')

# Cap values that fall far outside the observed min/max range for each numeric column
df['Pclass'] = df['Pclass'].clip(df['Pclass'].min(), df['Pclass'].max())
df['Age'] = df['Age'].clip(df['Age'].min(), df['Age'].max())
df['SibSp'] = df['SibSp'].clip(df['SibSp'].min(), df['SibSp'].max())
df['Parch'] = df['Parch'].clip(df['Parch'].min(), df['Parch'].max())
df['Fare'] = df['Fare'].clip(df['Fare'].min(), df['Fare'].max())