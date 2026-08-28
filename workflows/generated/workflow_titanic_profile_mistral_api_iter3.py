import pandas as pd
import numpy as np
import difflib

# --- 1. Disguised missing values ---
missing_markers = ['NA', 'N/A', 'unknown', '', ' ', 'NaN', 'nan', '?', 'None', '/A']
for col in ['Name', 'Sex', 'Age', 'Ticket', 'Fare', 'Cabin', 'Embarked']:
    df[col] = df[col].astype(str).replace({m: np.nan for m in missing_markers})

# --- 2. Numeric columns ---
# Convert Age to numeric
df['Age'] = pd.to_numeric(df['Age'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')
# Convert Fare to numeric
df['Fare'] = pd.to_numeric(df['Fare'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')

# --- 3. Outlier handling for numeric columns ---
# Age: treat values > 100 or < 0 as outliers
age_outliers = (df['Age'] > 100) | (df['Age'] < 0)
df.loc[age_outliers, 'Age'] = np.nan

# SibSp: cap at 8 (realistic max for siblings/spouses)
sibsp_outliers = (df['SibSp'] > 8)
df.loc[sibsp_outliers, 'SibSp'] = np.nan
df['SibSp'] = df['SibSp'].fillna(df['SibSp'].median())

# Parch: cap at 6 (realistic max for parents/children)
parch_outliers = (df['Parch'] > 6)
df.loc[parch_outliers, 'Parch'] = np.nan
df['Parch'] = df['Parch'].fillna(df['Parch'].median())

# --- 4. Categorical columns ---
# Sex: fuzzy correction
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

# Embarked: fuzzy correction
valid_embarked = ['S', 'C', 'Q']
def fuzzy_correct_embarked(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().upper()
    if val_norm in [v.upper() for v in valid_embarked]:
        return next(v for v in valid_embarked if v.upper() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.upper() for v in valid_embarked], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_embarked if v.upper() == match[0])
    return val
df['Embarked'] = df['Embarked'].apply(fuzzy_correct_embarked)

# --- 5. Group-aware imputation with improved strategies ---
# First pass: impute with strong group signals (using fillna only)

# Age imputation - composite key (Ticket + Pclass + Sex) for better accuracy
age_by_ticket_pclass_sex = df.groupby(['Ticket', 'Pclass', 'Sex'])['Age'].transform(
    lambda s: s.median() if not pd.isna(s.median()) else np.nan
)
df['Age'] = df['Age'].fillna(age_by_ticket_pclass_sex)

# Age imputation - Pclass + Sex
age_by_pclass_sex = df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
df['Age'] = df['Age'].fillna(age_by_pclass_sex)

# Age imputation - Pclass only
age_by_pclass = df.groupby('Pclass')['Age'].transform('median')
df['Age'] = df['Age'].fillna(age_by_pclass)

# Age imputation - SibSp (children often travel with siblings)
age_by_sibsp = df.groupby('SibSp')['Age'].transform('median')
df['Age'] = df['Age'].fillna(age_by_sibsp)

# Age imputation - Parch (children often travel with parents)
age_by_parch = df.groupby('Parch')['Age'].transform('median')
df['Age'] = df['Age'].fillna(age_by_parch)

# Fare imputation - Ticket
fare_by_ticket = df.groupby('Ticket')['Fare'].transform('median')
df['Fare'] = df['Fare'].fillna(fare_by_ticket)

# Fare imputation - Pclass
fare_by_pclass = df.groupby('Pclass')['Fare'].transform('median')
df['Fare'] = df['Fare'].fillna(fare_by_pclass)

# Embarked imputation - Ticket (strong signal)
embarked_by_ticket = df.groupby('Ticket')['Embarked'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['Embarked'] = df['Embarked'].fillna(embarked_by_ticket)

# Embarked imputation - Pclass (strong signal)
embarked_by_pclass = df.groupby('Pclass')['Embarked'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['Embarked'] = df['Embarked'].fillna(embarked_by_pclass)

# Cabin imputation - Ticket (only fill if at least 2 non-null values in group)
cabin_by_ticket = df.groupby('Ticket')['Cabin'].transform(
    lambda s: s.mode().iloc[0] if (not s.mode().empty and s.count() >= 2) else np.nan
)
df['Cabin'] = df['Cabin'].fillna(cabin_by_ticket)

# Second pass: global imputation only for columns with remaining missing values

# Age: global median (after all group imputations)
age_median = df['Age'].median()
df['Age'] = df['Age'].fillna(age_median)

# Fare: global median
fare_median = df['Fare'].median()
df['Fare'] = df['Fare'].fillna(fare_median)

# Sex: global mode
sex_mode = df['Sex'].mode(dropna=True)
fill_value = sex_mode.iloc[0] if not sex_mode.empty else 'male'
df['Sex'] = df['Sex'].fillna(fill_value)

# Embarked: global mode (after all group imputations)
embarked_mode = df['Embarked'].mode(dropna=True)
fill_value = embarked_mode.iloc[0] if not embarked_mode.empty else 'S'
df['Embarked'] = df['Embarked'].fillna(fill_value)

# Cabin: leave as NaN (high missing rate, weak imputation signal)
# Only fill if we have a very strong signal (at least 3 non-null in group)
cabin_by_ticket_strong = df.groupby('Ticket')['Cabin'].transform(
    lambda s: s.mode().iloc[0] if (not s.mode().empty and s.count() >= 3) else np.nan
)
df['Cabin'] = df['Cabin'].fillna(cabin_by_ticket_strong)

# --- 6. Special handling for very young children ---
# For passengers with Parch > 0 and Age still missing, assume they are infants
df.loc[(df['Parch'] > 0) & (df['Age'].isna()), 'Age'] = 0.75

# --- 7. Final cleanup ---
# Ensure numeric columns are properly typed
numeric_cols = ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
for col in numeric_cols:
    if col in ['Age', 'Fare']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        df[col] = df[col].astype(float)

# Final DataFrame
df