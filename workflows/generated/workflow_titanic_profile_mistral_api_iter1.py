import pandas as pd
import numpy as np
import difflib

# --- 1. Disguised missing values ---
# Define universal missing markers
missing_markers = ['NA', 'N/A', 'unknown', '', ' ', 'NaN', 'nan']

# Apply to all categorical/text columns
for col in ['Name', 'Sex', 'Age', 'Ticket', 'Fare', 'Cabin', 'Embarked']:
    df[col] = df[col].astype(str).replace({m: np.nan for m in missing_markers})

# --- 2. Numeric columns ---
# PassengerId, Survived, Pclass, SibSp, Parch are already numeric per profile
# Age and Fare are stored as text but should be numeric

# Age: convert to numeric
df['Age'] = pd.to_numeric(df['Age'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')
# Impute missing Age with median (after conversion)
age_median = df['Age'].median()
df['Age'] = df['Age'].fillna(age_median)

# Fare: convert to numeric
df['Fare'] = pd.to_numeric(df['Fare'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')
# Impute missing Fare with median (after conversion)
fare_median = df['Fare'].median()
df['Fare'] = df['Fare'].fillna(fare_median)

# --- 3. Outlier handling for numeric columns ---
# Pclass: min=1, max=3 (already valid)
# SibSp: min=0, max=48 (profile shows 48, but 8 is realistic max for family size)
sibsp_outliers = (df['SibSp'] > 8)
df.loc[sibsp_outliers, 'SibSp'] = np.nan
df['SibSp'] = df['SibSp'].fillna(df['SibSp'].median())

# Parch: min=0, max=47 (profile shows 47, but 6 is realistic max for family size)
parch_outliers = (df['Parch'] > 6)
df.loc[parch_outliers, 'Parch'] = np.nan
df['Parch'] = df['Parch'].fillna(df['Parch'].median())

# Age: min=0, max=80 is plausible (no action beyond imputation)
# Fare: min=0, max=512.3292 (profile max, no action beyond imputation)

# --- 4. Categorical columns ---
# Sex: low-cardinality (2 valid values: 'male', 'female')
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
# Impute remaining missing Sex with mode
sex_mode = df['Sex'].mode(dropna=True)
fill_value = sex_mode.iloc[0] if not sex_mode.empty else 'male'
df['Sex'] = df['Sex'].fillna(fill_value)

# Embarked: low-cardinality (3 valid values: 'S', 'C', 'Q')
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
# Impute remaining missing Embarked with mode
embarked_mode = df['Embarked'].mode(dropna=True)
fill_value = embarked_mode.iloc[0] if not embarked_mode.empty else 'S'
df['Embarked'] = df['Embarked'].fillna(fill_value)

# Cabin: very high missing rate (79.46%), leave as NaN unless group-imputed
# Group-aware imputation: Ticket -> Cabin (strong signal, NOT exact)
cabin_by_ticket = df.groupby('Ticket')['Cabin'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['Cabin'] = df['Cabin'].fillna(cabin_by_ticket)
# Fallback to global mode for remaining missing Cabin
cabin_mode = df['Cabin'].mode(dropna=True)
if not cabin_mode.empty:
    df['Cabin'] = df['Cabin'].fillna(cabin_mode.iloc[0])

# --- 5. Group-aware imputation for strong-signal keys ---
# Ticket -> Fare (already numeric, but impute missing with group median)
fare_by_ticket = df.groupby('Ticket')['Fare'].transform('median')
df['Fare'] = df['Fare'].fillna(fare_by_ticket)
# Fallback to global median
df['Fare'] = df['Fare'].fillna(fare_median)

# Ticket -> Embarked (already cleaned, but impute missing with group mode)
embarked_by_ticket = df.groupby('Ticket')['Embarked'].transform(
    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
)
df['Embarked'] = df['Embarked'].fillna(embarked_by_ticket)
# Fallback to global mode
df['Embarked'] = df['Embarked'].fillna(fill_value)

# Pclass -> Age (impute missing Age with group median)
age_by_pclass = df.groupby('Pclass')['Age'].transform('median')
df['Age'] = df['Age'].fillna(age_by_pclass)
# Fallback to global median
df['Age'] = df['Age'].fillna(age_median)

# Pclass -> Fare (impute missing Fare with group median)
fare_by_pclass = df.groupby('Pclass')['Fare'].transform('median')
df['Fare'] = df['Fare'].fillna(fare_by_pclass)
# Fallback to global median
df['Fare'] = df['Fare'].fillna(fare_median)

# --- 6. Name: no cleaning needed (unique identifier) ---
# --- 7. Ticket: no cleaning needed (unique identifier) ---

# Final DataFrame
df