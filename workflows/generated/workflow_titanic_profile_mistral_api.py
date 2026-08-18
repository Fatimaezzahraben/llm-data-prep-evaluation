import pandas as pd
import numpy as np
import re

# Define a function to detect disguised missing values
def is_disguised_missing(val):
    if pd.isna(val):
        return True
    val = str(val).strip().lower()
    return val in ['', 'na', 'n/a', 'unknown', 'null', 'none']

# Clean each column according to its profile

# PassengerId - numeric, no missing, no issues
# No cleaning needed

# Survived - numeric, no missing, no issues
# No cleaning needed

# Pclass - numeric, no missing, no issues
# No cleaning needed

# Name - categorical/text, no missing, but high cardinality (891 unique)
# No cleaning needed beyond ensuring no disguised missing values
df['Name'] = df['Name'].apply(lambda x: np.nan if is_disguised_missing(x) else x)

# Sex - categorical/text, no missing but has inconsistent values
# Create mapping for known variations
sex_mapping = {
    'male': 'male',
    'female': 'female',
    'mle': 'male',
    '  male ': 'male',
    'FEMAle': 'female'
}
df['Sex'] = df['Sex'].str.strip().replace(sex_mapping)
# Fill any remaining disguised missing with mode
mode_sex = df['Sex'].mode()[0]
df['Sex'] = df['Sex'].fillna(mode_sex)

# Age - categorical/text with 27.16% missing and many unique values
# First convert to numeric
df['Age'] = pd.to_numeric(df['Age'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
# Cap outliers (using 1st and 99th percentiles)
q_low = df['Age'].quantile(0.01)
q_hi = df['Age'].quantile(0.99)
df['Age'] = df['Age'].clip(q_low, q_hi)
# Fill missing with median
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)

# SibSp - numeric, no missing, but has extreme values (max 48)
# Cap outliers
q_low = df['SibSp'].quantile(0.01)
q_hi = df['SibSp'].quantile(0.99)
df['SibSp'] = df['SibSp'].clip(q_low, q_hi)

# Parch - numeric, no missing, but has extreme values (max 47)
# Cap outliers
q_low = df['Parch'].quantile(0.01)
q_hi = df['Parch'].quantile(0.99)
df['Parch'] = df['Parch'].clip(q_low, q_hi)

# Ticket - categorical/text, no missing but high cardinality (681 unique)
# No cleaning needed beyond ensuring no disguised missing values
df['Ticket'] = df['Ticket'].apply(lambda x: np.nan if is_disguised_missing(x) else x)

# Fare - categorical/text with no missing but many unique values
# Convert to numeric
df['Fare'] = pd.to_numeric(df['Fare'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
# Cap outliers
q_low = df['Fare'].quantile(0.01)
q_hi = df['Fare'].quantile(0.99)
df['Fare'] = df['Fare'].clip(q_low, q_hi)
# Fill any missing with median
median_fare = df['Fare'].median()
df['Fare'] = df['Fare'].fillna(median_fare)

# Cabin - categorical/text with 79.46% missing
# First handle disguised missing values
df['Cabin'] = df['Cabin'].apply(lambda x: np.nan if is_disguised_missing(x) else x)
# For imputation, we'll use group-aware imputation by Pclass (since cabin is likely related to class)
# First fill with group mode
df['Cabin'] = df.groupby('Pclass')['Cabin'].transform(
    lambda s: s.mode()[0] if not s.mode().empty else np.nan
)
# Then fill remaining with global mode
mode_cabin = df['Cabin'].mode()[0]
df['Cabin'] = df['Cabin'].fillna(mode_cabin)

# Embarked - categorical/text with 9.88% missing
# First handle disguised missing values
df['Embarked'] = df['Embarked'].apply(lambda x: np.nan if is_disguised_missing(x) else x)
# Create mapping for known variations
embarked_mapping = {
    'S': 'S',
    'C': 'C',
    'Q': 'Q',
    ' ': np.nan,
    'unknown': np.nan
}
df['Embarked'] = df['Embarked'].replace(embarked_mapping)
# Fill missing with mode
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)

# Final check: ensure all numeric columns are actually numeric
numeric_cols = ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
for col in numeric_cols:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill any new NaN with median
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)