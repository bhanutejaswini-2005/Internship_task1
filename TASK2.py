import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 1. Load Data & Initial Inspection
df = pd.read_csv("titanic.csv")
print(f"Data Shape: {df.shape}")
print("\nMissing Values:\n", df.isnull().sum())

# 2. Summary Statistics
print("\nSummary Stats:\n", df.describe(include='all'))

# 3. Handle Missing Values (Simple Imputation for EDA)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 4. Distribution Analysis
# Numerical Features: Age and Fare
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['Age'], bins=30, kde=True, ax=ax[0]).set_title('Age Distribution')
sns.boxplot(x='Pclass', y='Fare', data=df, ax=ax[1]).set_title('Fare by Class')
plt.show()

# Categorical Features: Survival by Gender
plt.figure(figsize=(8, 4))
sns.countplot(x='Sex', hue='Survived', data=df).set_title('Survival by Gender')
plt.show()

# 5. Correlation & Relationships
# Correlation Matrix Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Cross-Tabulation: Class vs Survival
class_survival = pd.crosstab(df['Pclass'], df['Survived'], normalize='index') * 100
print("\nSurvival Rate by Class (%):\n", class_survival)

# 6. Outlier Detection
# Z-Score Analysis for Fare
z_scores = np.abs(stats.zscore(df['Fare']))
outliers = df[z_scores > 3]
print(f"Found {len(outliers)} fare outliers")

# 7. Advanced Visualizations
# Faceted Analysis
g = sns.FacetGrid(df, col='Survived', row='Pclass', height=3)
g.map(sns.histplot, 'Age', bins=20)
plt.show()

# Pairplot for Multivariate Analysis
sns.pairplot(df[['Age', 'Fare', 'Parch', 'Survived']], hue='Survived')
plt.show()