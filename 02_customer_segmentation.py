
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("data/orders.csv", parse_dates=["Date"])

SNAPSHOT_DATE = df["Date"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer_ID").agg(
    recency=("Date", lambda x: (SNAPSHOT_DATE - x.max()).days),
    frequency=("Order_ID", "count"),
    monetary=("Total_Amount", "sum"),
).reset_index()

print("RFM summary stats:")
print(rfm[["recency", "frequency", "monetary"]].describe().round(2))


# Log-transform monetary since it's right-skewed (frequency here is capped
# at 10, so it doesn't need it as much, but monetary has a long tail)
rfm["log_monetary"] = np.log1p(rfm["monetary"])

features = rfm[["recency", "frequency", "log_monetary"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Elbow method to choose k
inertias = []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(list(k_range), inertias, marker="o", color="#2563eb")
ax.set_title("Elbow Method for Choosing k", fontweight="bold")
ax.set_xlabel("Number of clusters (k)")
ax.set_ylabel("Inertia")
plt.tight_layout()
plt.savefig("outputs/06_elbow_method.png")
plt.close()

K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
rfm["cluster"] = kmeans.fit_predict(X_scaled)

cluster_profile = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean().round(1)
print("\nCluster profiles (raw averages):")
print(cluster_profile)

cluster_profile["value_score"] = (
    cluster_profile["monetary"].rank() + cluster_profile["frequency"].rank() - cluster_profile["recency"].rank()
)
ranked = cluster_profile.sort_values("value_score", ascending=False).index.tolist()

label_map = {
    ranked[0]: "Champions",
    ranked[1]: "Loyal Customers",
    ranked[2]: "At Risk",
    ranked[3]: "New / Low-Engagement",
}
rfm["segment"] = rfm["cluster"].map(label_map)

print("\nSegment sizes:")
print(rfm["segment"].value_counts())
print("\nSegment sizes (% of customers):")
print((rfm["segment"].value_counts(normalize=True) * 100).round(1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

segment_order = ["Champions", "Loyal Customers", "At Risk", "New / Low-Engagement"]
palette = {"Champions": "#16a34a", "Loyal Customers": "#2563eb",
           "At Risk": "#f97316", "New / Low-Engagement": "#94a3b8"}

sns.scatterplot(
    data=rfm, x="recency", y="monetary", hue="segment", hue_order=segment_order,
    palette=palette, alpha=0.6, ax=axes[0]
)
axes[0].set_title("Customer Segments: Recency vs. Monetary Value", fontweight="bold")
axes[0].set_xlabel("Recency (days since last order)")
axes[0].set_ylabel("Total Spend")

seg_counts = rfm["segment"].value_counts().reindex(segment_order)
sns.barplot(x=seg_counts.index, y=seg_counts.values, hue=seg_counts.index,
            palette=palette, ax=axes[1], legend=False)
axes[1].set_title("Customers per Segment", fontweight="bold")
axes[1].set_ylabel("Number of Customers")
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("outputs/07_customer_segments.png")
plt.close()

rfm.to_csv("data/customer_segments.csv", index=False)
print("\nSaved data/customer_segments.csv and segment visualizations to outputs/")
