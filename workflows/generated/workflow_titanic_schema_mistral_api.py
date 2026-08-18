import pandas as pd
import numpy as np
import re

# Fix typos and inconsistent categories in text columns
# Sex column
df['Sex'] = df['Sex'].str.strip().str.lower()
sex_mapping = {
    'male': 'male',
    'female': 'female',
    'femal': 'female',
    'femmale': 'female',
    'femail': 'female'
}
df['Sex'] = df['Sex'].replace(sex_mapping)

# Embarked column
df['Embarked'] = df['Embarked'].str.strip().str.upper()
embarked_mapping = {
    'S': 'S',
    'C': 'C',
    'Q': 'Q',
    '': np.nan,
    ' ': np.nan
}
df['Embarked'] = df['Embarked'].replace(embarked_mapping)

# Name column - just strip whitespace
df['Name'] = df['Name'].str.strip()

# Ticket column - just strip whitespace
df['Ticket'] = df['Ticket'].str.strip()

# Cabin column - standardize format and handle missing values
df['Cabin'] = df['Cabin'].str.strip().str.upper()
df['Cabin'] = df['Cabin'].replace({'': np.nan, ' ': np.nan, 'N/A': np.nan, 'NA': np.nan})

# Ensure numeric columns contain only numeric values
# Age column
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
age_median = df['Age'].median()
df['Age'] = df['Age'].fillna(age_median)

# Fare column
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
fare_median = df['Fare'].median()
df['Fare'] = df['Fare'].fillna(fare_median)

# Handle unrealistic outlier values in numeric columns
# Age outliers - cap at reasonable values (0-100)
df['Age'] = df['Age'].clip(0, 100)

# Fare outliers - cap at reasonable values (0-500)
df['Fare'] = df['Fare'].clip(0, 500)

# Handle missing values per column
# Cabin - high cardinality, use group-aware imputation with Pclass as key
if df['Cabin'].isna().any():
    # First try group-wise imputation
    df['Cabin'] = df.groupby('Pclass')['Cabin'].transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    )
    # Fallback to global mode if group imputation didn't work
    global_cabin_mode = df['Cabin'].mode(dropna=True)
    if not global_cabin_mode.empty:
        df['Cabin'] = df['Cabin'].fillna(global_cabin_mode.iloc[0])

# Embarked - low cardinality, use global mode
embarked_mode = df['Embarked'].mode(dropna=True)
if not embarked_mode.empty:
    df['Embarked'] = df['Embarked'].fillna(embarked_mode.iloc[0])

# Ensure all numeric columns are properly typed
numeric_cols = ['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch']
for col in numeric_cols:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# Final check for any remaining missing values in categorical columns
categorical_cols = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
for col in categorical_cols:
    if df[col].isna().any():
        mode_val = df[col].mode(dropna=True)
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])

# Ensure no missing values remain in any column
for col in df.columns:
    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])