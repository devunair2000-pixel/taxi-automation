# ============================================================
# AUTOMATIDATA - NYC TLC HYPOTHESIS TESTING PROJECT
# FULL PROJECT CODE (UPDATED FOR YOUR DATASET NAME)
# ============================================================

# ------------------------------------------------------------
# STEP 1 — IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



# Optional display settings
pd.set_option('display.max_columns', None)

# ------------------------------------------------------------
# STEP 2 — LOAD DATASET
# ------------------------------------------------------------

# Your dataset name:
# autodata.csv

df = pd.read_csv('2017_Yellow_Taxi_Trip_Data.csv')

# ------------------------------------------------------------
# STEP 3 — INITIAL DATA EXPLORATION
# ------------------------------------------------------------

print("===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns)

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe())

# ------------------------------------------------------------
# STEP 4 — CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ------------------------------------------------------------
# STEP 5 — REMOVE MISSING VALUES
# ------------------------------------------------------------

df = df.dropna()

print("\nDataset shape after removing missing values:")
print(df.shape)

# ------------------------------------------------------------
# STEP 6 — DATA CLEANING
# ------------------------------------------------------------

# Remove negative fares
df = df[df['fare_amount'] > 0]

# Remove negative trip distances
df = df[df['trip_distance'] > 0]

# Remove unrealistic passenger counts
df = df[
    (df['passenger_count'] > 0) &
    (df['passenger_count'] <= 6)
]

# ------------------------------------------------------------
# STEP 7 — REMOVE OUTLIERS USING IQR
# ------------------------------------------------------------

Q1 = df['fare_amount'].quantile(0.25)
Q3 = df['fare_amount'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

df = df[
    (df['fare_amount'] >= lower_limit) &
    (df['fare_amount'] <= upper_limit)
]

print("\nDataset shape after outlier removal:")
print(df.shape)

# ------------------------------------------------------------
# STEP 8 — EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Histogram of Fare Amount
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.histplot(df['fare_amount'], bins=30)

plt.title('Distribution of Fare Amount')

plt.xlabel('Fare Amount')

plt.ylabel('Frequency')

plt.show()

# ------------------------------------------------------------
# Boxplot of Fare Amount
# ------------------------------------------------------------

plt.figure(figsize=(8,5))

sns.boxplot(x=df['fare_amount'])

plt.title('Boxplot of Fare Amount')

plt.show()

# ------------------------------------------------------------
# Scatterplot
# Trip Distance vs Fare Amount
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(
    x=df['trip_distance'],
    y=df['fare_amount']
)

plt.title('Trip Distance vs Fare Amount')

plt.xlabel('Trip Distance')

plt.ylabel('Fare Amount')

plt.show()

# ------------------------------------------------------------
# STEP 9 — PAYMENT TYPE ANALYSIS
# ------------------------------------------------------------

print("\n===== PAYMENT TYPE COUNTS =====")
print(df['payment_type'].value_counts())

# ------------------------------------------------------------
# STEP 10 — CREATE DATA SUBSETS
# ------------------------------------------------------------

# Usually:
# 1 = Credit Card
# 2 = Cash

credit_card = df[df['payment_type'] == 1]['fare_amount']

cash = df[df['payment_type'] == 2]['fare_amount']

# ------------------------------------------------------------
# STEP 11 — DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n===== CREDIT CARD AVERAGE FARE =====")
print(credit_card.mean())

print("\n===== CASH AVERAGE FARE =====")
print(cash.mean())

# ------------------------------------------------------------
# STEP 12 — HYPOTHESIS TESTING
# ------------------------------------------------------------

# Null Hypothesis:
# Mean fare of credit card users
# equals mean fare of cash users

# Alternative Hypothesis:
# Mean fares are different

t_stat, p_value = stats.ttest_ind(
    credit_card,
    cash,
    equal_var=False
)

print("\n===== T-STATISTIC =====")
print(t_stat)

print("\n===== P-VALUE =====")
print(p_value)

# ------------------------------------------------------------
# STEP 13 — INTERPRET RESULTS
# ------------------------------------------------------------

alpha = 0.05

print("\n===== HYPOTHESIS TEST RESULT =====")

if p_value < alpha:

    print("Reject the null hypothesis.")

    print("There is a statistically significant")

    print("difference between credit card")

    print("and cash fare amounts.")

else:

    print("Fail to reject the null hypothesis.")

    print("There is NO statistically significant")

    print("difference between payment types.")

# ------------------------------------------------------------
# STEP 14 — VISUALIZATION BY PAYMENT TYPE
# ------------------------------------------------------------

plt.figure(figsize=(8,6))

sns.boxplot(
    x='payment_type',
    y='fare_amount',
    data=df
)

plt.title('Fare Amount by Payment Type')

plt.xlabel('Payment Type')

plt.ylabel('Fare Amount')

plt.show()

# ------------------------------------------------------------
# STEP 15 — CORRELATION MATRIX
# ------------------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title('Correlation Matrix')

plt.show()

# ------------------------------------------------------------
# STEP 16 — EXECUTIVE SUMMARY OUTPUT
# ------------------------------------------------------------

print("\n===== EXECUTIVE SUMMARY =====")

print("""

1. The dataset was cleaned by removing:
   - Missing values
   - Negative fares
   - Invalid trip distances
   - Outliers

2. Exploratory analysis showed that:
   - Fare amount increases with trip distance
   - Payment methods vary among passengers

3. A two-sample independent t-test was conducted
   to compare fare amounts between:
   - Credit card customers
   - Cash customers

4. Statistical results indicate whether payment
   type has a significant effect on fare amount.

5. These findings can help Automatidata and the
   NYC TLC improve predictive fare estimation
   models.

""")

# ------------------------------------------------------------
# STEP 17 — SAVE CLEANED DATASET
# ------------------------------------------------------------

df.to_csv('cleaned_autodata.csv', index=False)

print("Cleaned dataset saved successfully.")