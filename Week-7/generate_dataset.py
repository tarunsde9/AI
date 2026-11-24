import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

age = np.random.randint(18, 65, n)
income = np.random.randint(20000, 120000, n)
spending = np.random.randint(100, 2000, n)
tenure = np.random.randint(1, 72, n)
support_calls = np.random.randint(0, 10, n)

# Generate churn labels
churn = (
    (support_calls >= 6) |
    ((tenure <= 6) & (spending < 500))
).astype(int)

df = pd.DataFrame({
    "Age": age,
    "AnnualIncome": income,
    "MonthlySpending": spending,
    "Tenure": tenure,
    "SupportCalls": support_calls,
    "Churn": churn
})

df.to_csv("dataset.csv", index=False)

print("Dataset generated successfully!")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nChurn distribution:")
print(df["Churn"].value_counts())
