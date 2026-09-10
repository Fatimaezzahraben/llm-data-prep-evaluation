import pandas as pd
import numpy as np
import difflib

# Clean PassengerId (no action needed, already numeric and no missing)
# Clean Survived (no action needed, already numeric and no missing)
# Clean Pclass (no action needed, already numeric and no missing)

# Clean Name (no action needed, already text and no missing)

# Clean Sex - handle missing and typos
df['Sex'] = df['Sex'].replace(['mle', '  male ', 'FEMAle', 'ale'], np.nan)
df['Sex'] = df['Sex'].fillna(df['Sex'].mode(dropna=True).iloc[0] if not df['Sex'].mode(dropna=True).empty else df['Sex'])

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
df['Cabin'] = df['Cabin'].fillna(df.groupby('Ticket')['Cabin'].transform(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan))

# Clean Embarked - handle missing and typos
df['Embarked'] = df['Embarked'].replace([' ', 'unknown'], np.nan)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode(dropna=True).iloc[0] if not df['Embarked'].mode(dropna=True).empty else df['Embarked'])