import pandas as pd
import numpy as np
import difflib
import re

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
df['Sex'] = df['Sex'].fillna(_mode.iloc[0] if not _mode.empty else df['Sex'])

# Clean Age - handle missing and outliers first
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.loc[(df['Age'] < 0) | (df['Age'] > 100), 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df.groupby('Pclass')['Age'].transform(lambda s: s.median()))

# Clean Fare - handle missing and outliers first
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
df.loc[(df['Fare'] < 0) | (df['Fare'] > 520), 'Fare'] = np.nan
df['Fare'] = df['Fare'].fillna(df.groupby('Ticket')['Fare'].transform(lambda s: s.median()))

# Clean Embarked - handle missing and typos
df['Embarked'] = df['Embarked'].replace([' ', 'unknown'], np.nan)
_mode = df['Embarked'].mode(dropna=True)
df['Embarked'] = df['Embarked'].fillna(_mode.iloc[0] if not _mode.empty else df['Embarked'])

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

# Clean Cabin - handle missing only
df['Cabin'] = df['Cabin'].replace(['unknown', ' '], np.nan)
group_val = df.groupby('Ticket')['Cabin'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
df['Cabin'] = df['Cabin'].fillna(group_val)

# Clean Age - final pass for any remaining issues
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.loc[(df['Age'] < 0) | (df['Age'] > 100), 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df.groupby('Pclass')['Age'].transform(lambda s: s.median()))

# Clean Fare - final pass for any remaining issues
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
df.loc[(df['Fare'] < 0) | (df['Fare'] > 520), 'Fare'] = np.nan
df['Fare'] = df['Fare'].fillna(df.groupby('Ticket')['Fare'].transform(lambda s: s.median()))

# Clean Fare - handle specific cases like '13l.0' and '49.5O'
df['Fare'] = df['Fare'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
df.loc[(df['Fare'] < 0) | (df['Fare'] > 520), 'Fare'] = np.nan
df['Fare'] = df['Fare'].fillna(df.groupby('Ticket')['Fare'].transform(lambda s: s.median()))

# Clean Age - handle specific cases like '52€.0'
df['Age'] = df['Age'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.loc[(df['Age'] < 0) | (df['Age'] > 100), 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df.groupby('Pclass')['Age'].transform(lambda s: s.median()))

# Clean Embarked - handle specific cases
df['Embarked'] = df['Embarked'].replace([' ', 'unknown'], np.nan)
_mode = df['Embarked'].mode(dropna=True)
df['Embarked'] = df['Embarked'].fillna(_mode.iloc[0] if not _mode.empty else df['Embarked'])