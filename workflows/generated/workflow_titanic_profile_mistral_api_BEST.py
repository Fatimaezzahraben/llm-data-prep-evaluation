import pandas as pd
import numpy as np
import difflib

# --- Clean 'Sex' column (categorical, 29 unique values but should be 2) ---
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

# Normalize whitespace and case for 'Sex'
df['Sex'] = df['Sex'].str.strip().str.lower()
df['Sex'] = df['Sex'].replace({'mle': 'male', 'femalem': 'female', 'femmale': 'female'})

# Final impute with mode
sex_mode = df['Sex'].mode()[0]
df['Sex'] = df['Sex'].fillna(sex_mode)
df['Sex'] = df['Sex'].replace({'male': 'male', 'female': 'female'})  # Ensure only valid values

# --- Clean 'Age' column (currently text, should be numeric) ---
# Convert to numeric, coercing errors
df['Age'] = pd.to_numeric(df['Age'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

# Cap outliers (min=0, max=100 for age)
df.loc[(df['Age'] < 0) | (df['Age'] > 100), 'Age'] = np.nan

# Group-aware imputation: Pclass + Title -> Age (composite key)
# Extract title from Name
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
title_age_map = {
    'Mr': 30, 'Mrs': 35, 'Miss': 22, 'Master': 5, 'Dr': 40,
    'Rev': 45, 'Col': 50, 'Major': 50, 'Mlle': 22, 'Ms': 28,
    'Lady': 35, 'Sir': 50, 'Don': 40, 'Mme': 35, 'Capt': 50,
    'Countess': 40, 'Jonkheer': 35
}
df['Title'] = df['Title'].replace(title_age_map)

# Group by Pclass and Title for age imputation
age_by_pclass_title = df.groupby(['Pclass', 'Title'])['Age'].transform(
    lambda s: s.median() if not pd.isna(s.median()) else np.nan
)
df['Age'] = df['Age'].fillna(age_by_pclass_title)

# Fallback to Pclass-only imputation
age_by_pclass = df.groupby('Pclass')['Age'].transform(lambda s: s.median())
df['Age'] = df['Age'].fillna(age_by_pclass)

# Global median fallback
age_median = df['Age'].median()
df['Age'] = df['Age'].fillna(age_median)

# --- Clean 'Fare' column (currently text, should be numeric) ---
df['Fare'] = pd.to_numeric(df['Fare'].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

# Cap outliers (min=0, max=500 for fare)
df.loc[(df['Fare'] < 0) | (df['Fare'] > 500), 'Fare'] = np.nan

# Group-aware imputation: Ticket -> Fare
fare_by_ticket = df.groupby('Ticket')['Fare'].transform(lambda s: s.median())
df['Fare'] = df['Fare'].fillna(fare_by_ticket)

# Global median fallback
fare_median = df['Fare'].median()
df['Fare'] = df['Fare'].fillna(fare_median)

# --- Clean 'Embarked' column (categorical, 9 unique values but should be 3) ---
valid_embarked = ['S', 'C', 'Q']
def fuzzy_correct_embarked(val):
    if pd.isna(val) or str(val).strip() in ['', 'unknown', 'nan']:
        return np.nan
    val_norm = str(val).strip().upper()
    if val_norm in valid_embarked:
        return val_norm
    match = difflib.get_close_matches(val_norm, valid_embarked, n=1, cutoff=0.6)
    if match:
        return match[0]
    return np.nan
df['Embarked'] = df['Embarked'].apply(fuzzy_correct_embarked)

# Group-aware imputation: Ticket -> Embarked
embarked_by_ticket = df.groupby('Ticket')['Embarked'].transform(
    lambda s: s.mode()[0] if not s.mode().empty else np.nan
)
df['Embarked'] = df['Embarked'].fillna(embarked_by_ticket)

# Global mode fallback
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(embarked_mode)

# --- Clean 'Cabin' column (very high missing rate, 79.46%) ---
# Only fill when we have a specific signal (Ticket -> Cabin)
df['Cabin'] = df['Cabin'].replace(['', ' ', 'unknown'], np.nan)

# Group-aware imputation: Ticket -> Cabin
cabin_by_ticket = df.groupby('Ticket')['Cabin'].transform(
    lambda s: s.mode()[0] if not s.mode().empty else np.nan
)
df['Cabin'] = df['Cabin'].fillna(cabin_by_ticket)

# For remaining missing, leave as NaN (high missing rate)

# --- Clean 'SibSp' and 'Parch' (numeric, but have extreme outliers) ---
# Cap outliers (SibSp max=10, Parch max=10)
df.loc[df['SibSp'] > 10, 'SibSp'] = np.nan
df.loc[df['Parch'] > 10, 'Parch'] = np.nan

# Impute with median
sibsp_median = df['SibSp'].median()
parch_median = df['Parch'].median()
df['SibSp'] = df['SibSp'].fillna(sibsp_median)
df['Parch'] = df['Parch'].fillna(parch_median)

# --- Clean 'Name' column (no missing, but standardize formatting) ---
df['Name'] = df['Name'].replace(['', ' ', 'unknown'], np.nan)
name_mode = df['Name'].mode()[0]
df['Name'] = df['Name'].fillna(name_mode)

# --- Clean 'Ticket' column (no missing, but standardize formatting) ---
df['Ticket'] = df['Ticket'].replace(['', ' ', 'unknown'], np.nan)
ticket_mode = df['Ticket'].mode()[0]
df['Ticket'] = df['Ticket'].fillna(ticket_mode)

# --- Ensure 'PassengerId', 'Survived', 'Pclass' are numeric and have no missing ---
for col in ['PassengerId', 'Survived', 'Pclass']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# --- Final check: no NaN in critical columns ---
critical_cols = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
for col in critical_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode()[0] if df[col].dtype == 'object' else df[col].median())

# Drop temporary 'Title' column
df.drop('Title', axis=1, inplace=True, errors='ignore')

df