import pandas as pd

# Dataset load
df = pd.read_csv("../data/creditcard.csv")

print("===================================")
print("FRAUD DETECTION DATASET ANALYSIS")
print("===================================")

# Dataset size
print("\nDataset Shape:")
print(df.shape)

# First 5 records
print("\nFirst 5 Records:")
print(df.head())

# Columns
print("\nColumns:")
print(df.columns.tolist())

# Data information
print("\nDataset Information:")
df.info()

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Fraud distribution
print("\nFraud Distribution:")
print(df["Class"].value_counts())

# Fraud percentage
print("\nFraud Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)
# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# Fraud vs Genuine
print("\nTransaction Distribution:")
print(df["Class"].value_counts())

# Fraud percentage
print("\nTransaction Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)
# ==========================================
# DATA CLEANING
# ==========================================

print("\n========== DATA CLEANING ==========")

# Remove duplicate rows
df_clean = df.drop_duplicates().copy()

print("Original rows:", len(df))
print("Rows after removing duplicates:", len(df_clean))
print("Duplicates removed:", len(df) - len(df_clean))

# Check missing values
print("\nTotal missing values:")
print(df_clean.isnull().sum().sum())

# Check fraud distribution after cleaning
print("\nFraud distribution after cleaning:")
print(df_clean["Class"].value_counts())

# Save cleaned dataset
df_clean.to_csv("../data/cleaned_creditcard.csv", index=False)

print("\nCleaned dataset saved successfully!")
# ==========================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# Use cleaned data
df = df_clean

# 1. Fraud vs Genuine
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df)

plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.xticks([0, 1], ["Genuine", "Fraud"])

plt.show()


# 2. Transaction Amount Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Amount"], bins=50)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.show()


# 3. Fraud Transaction Amount
fraud_df = df[df["Class"] == 1]

plt.figure(figsize=(8, 5))
plt.hist(fraud_df["Amount"], bins=30)

plt.title("Fraud Transaction Amount Distribution")
plt.xlabel("Fraud Transaction Amount")
plt.ylabel("Frequency")

plt.show()


print("\nEDA completed successfully!")
# ==========================================
# FEATURE ENGINEERING
# ==========================================

print("\n========== FEATURE ENGINEERING ==========")

# Work on cleaned dataset
df = df_clean.copy()

# 1. Convert transaction time into hours
df["Hour"] = ((df["Time"] / 3600) % 24).astype(int)

# 2. Create amount categories
def amount_category(amount):
    if amount < 50:
        return "Low"
    elif amount < 500:
        return "Medium"
    elif amount < 2000:
        return "High"
    else:
        return "Very High"

df["Amount_Category"] = df["Amount"].apply(amount_category)

# 3. Log transformation of amount
import numpy as np

df["Log_Amount"] = np.log1p(df["Amount"])

# 4. Display new features
print("\nNew Features:")
print(df[[
    "Time",
    "Amount",
    "Hour",
    "Amount_Category",
    "Log_Amount",
    "Class"
]].head())

# 5. Fraud by hour
print("\nFraud Transactions by Hour:")
print(
    df[df["Class"] == 1]["Hour"]
    .value_counts()
    .sort_index()
)

# 6. Fraud by amount category
print("\nFraud Transactions by Amount Category:")
print(
    df[df["Class"] == 1]["Amount_Category"]
    .value_counts()
)

# Save feature-engineered dataset
df.to_csv(
    "../data/fraud_features.csv",
    index=False
)

print("\nFeature engineering completed successfully!")
print("File saved: data/fraud_features.csv")