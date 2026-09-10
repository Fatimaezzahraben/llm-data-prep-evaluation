import pandas as pd
import numpy as np
import difflib
import re

# Clean PassengerId (no action needed, already numeric and no missing)
# Clean Survived (no action needed, already numeric and no missing)
# Clean Pclass (no action needed, already numeric and no missing)

# Clean Name (no action needed, already text and no missing)

# Clean Sex - handle missing and typos
def correct_sex(val):
    if pd.isna(val):
        return val
    val_norm = str(val).strip().lower()
    valid_values = ['male', 'female']
    if val_norm in [v.lower() for v in valid_values]:
        return next(v for v in valid_values if v.lower() == val_norm)
    match = difflib.get_close_matches(val_norm, [v.lower() for v in valid_values], n=1, cutoff=0.6)
    if match:
        return next(v for v in valid_values if v.lower() == match[0])
    return val  # leave unmatched values as-is

df['Sex'] = df['Sex'].apply(correct_sex)
df['Sex'] = df['Sex'].replace(['mle', '  male ', 'FEMAle', 'ale'], np.nan)
_mode = df['Sex'].mode(dropna=True)
fill_value = _mode.iloc[0] if not _mode.empty else df['Sex']
df['Sex'] = df['Sex'].fillna(fill_value)

# Clean Age - handle missing and outliers first
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.loc[(df['Age'] < 0) | (df['Age'] > 100), 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df.groupby('Pclass')['Age'].transform(lambda s: s.median()))

# Clean SibSp and Parch (no action needed, already numeric and no missing)

# Clean Ticket (no action needed, already text and no missing)

# Clean Fare - handle missing and outliers first
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
df.loc[(df['Fare'] < 0) | (df['Fare'] > 520), 'Fare'] = np.nan
df['Fare'] = df['Fare'].fillna(df.groupby('Ticket')['Fare'].transform(lambda s: s.median()))

# Clean Cabin - handle missing only
df['Cabin'] = df['Cabin'].replace(['unknown', ' '], np.nan)
group_val = df.groupby('Ticket')['Cabin'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['Cabin'] = df['Cabin'].fillna(group_val)

# Clean Embarked - handle missing and typos
df['Embarked'] = df['Embarked'].replace([' ', 'unknown'], np.nan)
_mode = df['Embarked'].mode(dropna=True)
fill_value = _mode.iloc[0] if not _mode.empty else df['Embarked']
df['Embarked'] = df['Embarked'].fillna(fill_value)

# Clean Age - additional cleaning for currency symbols and letters
df['Age'] = df['Age'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.loc[(df['Age'] < 0) | (df['Age'] > 100), 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df.groupby('Pclass')['Age'].transform(lambda s: s.median()))

# Clean Fare - additional cleaning for non-numeric characters
df['Fare'] = df['Fare'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
df.loc[(df['Fare'] < 0) | (df['Fare'] > 520), 'Fare'] = np.nan
df['Fare'] = df['Fare'].fillna(df.groupby('Ticket')['Fare'].transform(lambda s: s.median()))

# Clean Age - handle specific known problematic values
df.loc[df['Age'].astype(str).str.contains(r'[^0-9.\-]', regex=True), 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df.groupby('Pclass')['Age'].transform(lambda s: s.median()))

# Clean Fare - handle specific known problematic values
df.loc[df['Fare'].astype(str).str.contains(r'[^0-9.\-]', regex=True), 'Fare'] = np.nan
df['Fare'] = df['Fare'].fillna(df.groupby('Ticket')['Fare'].transform(lambda s: s.median()))

# Clean Age - ensure no remaining non-numeric values
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.loc[(df['Age'] < 0) | (df['Age'] > 100), 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df.groupby('Pclass')['Age'].transform(lambda s: s.median()))

# Clean Fare - ensure no remaining non-numeric values
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
df.loc[(df['Fare'] < 0) | (df['Fare'] > 520), 'Fare'] = np.nan
df['Fare'] = df['Fare'].fillna(df.groupby('Ticket')['Fare'].transform(lambda s: s.median()))