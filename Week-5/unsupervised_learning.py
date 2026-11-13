import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

np.random.seed(42)

data = {
    "CustomerID": range(1, 101),
    "Age": np.random.randint(18, 65, 100),
    "AnnualIncome": np.random.randint(15000, 100000, 100),
    "SpendingScore": np.random.randint(1, 100, 100)
}

df = pd.DataFrame(data)

print(df.head())
print("\nDataset Shape:", df.shape)

# Select features for clustering
X = df[["Age", "AnnualIncome", "SpendingScore"]]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# Display cluster assignments
print("\nCluster Assignments:")
print(df[["CustomerID", "Age", "AnnualIncome",
          "SpendingScore", "Cluster"]].head(10))

# Count customers in each cluster
print("\nCustomers per Cluster:")
print(df["Cluster"].value_counts().sort_index())

# Profile each cluster
cluster_profile = df.groupby("Cluster")[
    ["Age", "AnnualIncome", "SpendingScore"]
].mean().round(2)

print("\nCluster Profiles:")
print(cluster_profile)

print("\nCluster Sizes:")
print(df.groupby("Cluster").size())

# Apply Isolation Forest
iso_forest = IsolationForest(
    contamination=0.05,
    random_state=42
)

df["Anomaly"] = iso_forest.fit_predict(X_scaled)

# Extract anomalies
anomalies = df[df["Anomaly"] == -1]

print("\nDetected Anomalies:")
print(anomalies)

print("\nTotal Anomalies:", len(anomalies))

# Save clustered dataset
df.to_csv("datasets/clustered_customers.csv", index=False)

# Save anomaly records
anomalies.to_csv("datasets/customer_anomalies.csv", index=False)

# Save cluster profiles
cluster_profile.to_csv("datasets/cluster_profiles.csv")

print("\nResults saved successfully!")