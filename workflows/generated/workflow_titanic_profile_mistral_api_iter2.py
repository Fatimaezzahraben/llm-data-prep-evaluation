import pandas as pd
import numpy as np
import difflib
import re

# ====================== MISSING VALUE HANDLING ======================
# First, convert all disguised missing values to np.nan for all columns
disguised_missing = ['', ' ', 'NA', 'N/A', 'unknown', 'nan', 'NaN', 'None', 'null']
for col in df.columns:
    df[col] = df[col].replace(disguised_missing, np.nan)
    # Also handle whitespace-only strings
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()
        df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

# ====================== SEX COLUMN ======================
valid_sex = ['male', 'female']
def fuzzy_correct_sex(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    if val_norm in [v.lower() for v in valid_sex]:
        return next(v for v in valid_sex if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_sex], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_sex if v.lower() == match[0])
    return val

df['Sex'] = df['Sex'].apply(fuzzy_correct_sex)
df['Sex'] = df['Sex'].str.lower()

# Final imputation with mode
sex_mode = df['Sex'].mode()[0]
df['Sex'] = df['Sex'].fillna(sex_mode)
df['Sex'] = df['Sex'].replace({'male': 'male', 'female': 'female'})  # Ensure only valid values

# ====================== AGE COLUMN ======================
# Convert to numeric, handling various formats
def clean_age(age_str):
    if pd.isna(age_str):
        return np.nan
    age_str = str(age_str).strip()
    # Remove any non-numeric characters except decimal point and minus
    age_str = re.sub(r'[^0-9.-]', '', age_str)
    if not age_str or age_str == '.':
        return np.nan
    try:
        return float(age_str)
    except ValueError:
        return np.nan

df['Age'] = df['Age'].apply(clean_age)

# Cap outliers (plausible range: 0-100)
df.loc[(df['Age'] < 0) | (df['Age'] > 100), 'Age'] = np.nan

# Group-aware imputation: Pclass -> Age (known grouping key)
age_by_pclass = df.groupby('Pclass')['Age'].transform(lambda s: s.median())
df['Age'] = df['Age'].fillna(age_by_pclass)

# Composite key imputation: (Pclass, Sex) -> Age (stronger grouping)
age_by_pclass_sex = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda s: s.median())
df['Age'] = df['Age'].fillna(age_by_pclass_sex)

# Global median fallback
age_median = df['Age'].median()
df['Age'] = df['Age'].fillna(age_median)

# ====================== FARE COLUMN ======================
# Convert to numeric
def clean_fare(fare_str):
    if pd.isna(fare_str):
        return np.nan
    fare_str = str(fare_str).strip()
    # Remove any non-numeric characters except decimal point
    fare_str = re.sub(r'[^0-9.]', '', fare_str)
    if not fare_str or fare_str == '.':
        return np.nan
    try:
        return float(fare_str)
    except ValueError:
        return np.nan

df['Fare'] = df['Fare'].apply(clean_fare)

# Cap outliers (plausible range: 0-500)
df.loc[(df['Fare'] < 0) | (df['Fare'] > 500), 'Fare'] = np.nan

# Group-aware imputation: Ticket -> Fare (known grouping key)
fare_by_ticket = df.groupby('Ticket')['Fare'].transform(lambda s: s.median())
df['Fare'] = df['Fare'].fillna(fare_by_ticket)

# Group-aware imputation: Pclass -> Fare (known grouping key)
fare_by_pclass = df.groupby('Pclass')['Fare'].transform(lambda s: s.median())
df['Fare'] = df['Fare'].fillna(fare_by_pclass)

# Global median fallback
fare_median = df['Fare'].median()
df['Fare'] = df['Fare'].fillna(fare_median)

# ====================== EMBARKED COLUMN ======================
valid_embarked = ['S', 'C', 'Q']
def fuzzy_correct_embarked(val):
    if pd.isna(val):
        return np.nan
    val_norm = str(val).strip().upper()
    if val_norm in valid_embarked:
        return val_norm
    match = difflib.get_close_matches(val_norm, valid_embarked, n=1, cutoff=0.6)
    if match:
        return match[0]
    return np.nan

df['Embarked'] = df['Embarked'].apply(fuzzy_correct_embarked)

# Group-aware imputation: Ticket -> Embarked (known grouping key)
embarked_by_ticket = df.groupby('Ticket')['Embarked'].transform(lambda s: s.mode()[0] if not s.mode().empty else np.nan)
df['Embarked'] = df['Embarked'].fillna(embarked_by_ticket)

# Global mode fallback
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(embarked_mode)

# ====================== CABIN COLUMN ======================
# High missing rate - only fill when we have a strong signal
# Group-aware imputation: Ticket -> Cabin (strong grouping)
cabin_by_ticket = df.groupby('Ticket')['Cabin'].transform(lambda s: s.mode()[0] if not s.mode().empty else np.nan)
df['Cabin'] = df['Cabin'].fillna(cabin_by_ticket)

# Group-aware imputation: (Pclass, Embarked) -> Cabin (strong grouping)
cabin_by_pclass_embarked = df.groupby(['Pclass', 'Embarked'])['Cabin'].transform(
    lambda s: s.mode()[0] if not s.mode().empty else np.nan)
df['Cabin'] = df['Cabin'].fillna(cabin_by_pclass_embarked)

# Leave remaining as NaN (high missing rate)

# ====================== SIBSP AND PARCH COLUMNS ======================
# Cap outliers (plausible range: 0-10)
for col in ['SibSp', 'Parch']:
    df.loc[df[col] > 10, col] = np.nan
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# ====================== NAME COLUMN ======================
# Standardize formatting (no content changes)
name_mode = df['Name'].mode()[0]
df['Name'] = df['Name'].fillna(name_mode)

# ====================== TICKET COLUMN ======================
# Standardize formatting (no content changes)
ticket_mode = df['Ticket'].mode()[0]
df['Ticket'] = df['Ticket'].fillna(ticket_mode)

# ====================== NUMERIC COLUMNS ======================
for col in ['PassengerId', 'Survived', 'Pclass']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# ====================== FINAL CHECK ======================
critical_cols = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
for col in critical_cols:
    if df[col].isna().any():
        if df[col].dtype == 'object':
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
        else:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

df