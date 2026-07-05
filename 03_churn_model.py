
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("data/orders.csv", parse_dates=["Date"])

SNAPSHOT_DATE = df["Date"].max() + pd.Timedelta(days=1)
CHURN_WINDOW_DAYS = 90
FEATURE_CUTOFF = SNAPSHOT_DATE - pd.Timedelta(days=CHURN_WINDOW_DAYS)

history = df[df["Date"] < FEATURE_CUTOFF].copy()


first_seen = history.sort_values("Date").groupby("Customer_ID").first()

features = history.groupby("Customer_ID").agg(
    frequency=("Order_ID", "count"),
    monetary=("Total_Amount", "sum"),
    avg_order_value=("Total_Amount", "mean"),
    recency_at_cutoff=("Date", lambda x: (FEATURE_CUTOFF - x.max()).days),
    category_diversity=("Product_Category", "nunique"),
    avg_session_minutes=("Session_Duration_Minutes", "mean"),
    avg_pages_viewed=("Pages_Viewed", "mean"),
    avg_delivery_days=("Delivery_Time_Days", "mean"),
    avg_rating=("Customer_Rating", "mean"),
    discount_share=("Discount_Amount", lambda x: (x > 0).mean()),
).reset_index()

demo = first_seen[["Age", "Gender", "City"]].reset_index()
features = features.merge(demo, on="Customer_ID")

tenure = history.groupby("Customer_ID")["Date"].min().reset_index()
tenure["tenure_days"] = (FEATURE_CUTOFF - tenure["Date"]).dt.days
features = features.merge(tenure[["Customer_ID", "tenure_days"]], on="Customer_ID")

future_active = df[df["Date"] >= FEATURE_CUTOFF]["Customer_ID"].unique()
features["churned"] = (~features["Customer_ID"].isin(future_active)).astype(int)

print(f"Customers in training set: {len(features):,}")
print(f"Churn rate: {features['churned'].mean():.1%}")

top_cities = features["City"].value_counts().nlargest(4).index
features["City"] = features["City"].where(features["City"].isin(top_cities), "Other")

features_encoded = pd.get_dummies(features, columns=["Gender", "City"], drop_first=True)
X = features_encoded.drop(columns=["Customer_ID", "churned"])
y = features_encoded["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
axes[0].set_title("Confusion Matrix", fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"ROC-AUC = {auc:.3f}")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[1].set_title("ROC Curve", fontweight="bold")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/08_confusion_matrix_roc.png")
plt.close()

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances:")
print(importances.round(3))

fig, ax = plt.subplots(figsize=(9, 6.5))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
            palette="Reds_r", ax=ax, legend=False)
ax.set_title("What Predicts Customer Churn?", fontweight="bold")
ax.set_xlabel("Feature Importance")
plt.tight_layout()
plt.savefig("outputs/09_feature_importance.png")
plt.close()

print("\nSaved model evaluation charts to outputs/")
