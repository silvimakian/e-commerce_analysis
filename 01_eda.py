
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("data/orders.csv", parse_dates=["Date"])

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print(f"\nTotal revenue: ${df['Total_Amount'].sum():,.2f}")
print(f"Total orders: {len(df):,}")
print(f"Unique customers: {df['Customer_ID'].nunique():,}")
print(f"Average order value: ${df['Total_Amount'].mean():.2f}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

#Q1: How is revenue trending over time?
monthly = df.set_index("Date").resample("ME")["Total_Amount"].sum()

fig, ax = plt.subplots(figsize=(11, 5))
monthly.plot(ax=ax, marker="o", color="#2563eb", linewidth=2)
ax.set_title("Monthly Revenue Trend", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")
ax.yaxis.set_major_formatter(lambda x, _: f"${x/1000:.0f}K")
plt.tight_layout()
plt.savefig("outputs/01_monthly_revenue_trend.png")
plt.close()


#Q2: Which product categories and cities drive the most revenue?
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

cat_rev = df.groupby("Product_Category")["Total_Amount"].sum().sort_values(ascending=False)
sns.barplot(x=cat_rev.values, y=cat_rev.index, hue=cat_rev.index, ax=axes[0],
            palette="Blues_d", legend=False)
axes[0].set_title("Revenue by Product Category", fontweight="bold")
axes[0].set_xlabel("Revenue")
axes[0].xaxis.set_major_formatter(lambda x, _: f"${x/1000:.0f}K")

city_rev = df.groupby("City")["Total_Amount"].sum().sort_values(ascending=False)
sns.barplot(x=city_rev.values, y=city_rev.index, hue=city_rev.index, ax=axes[1],
            palette="Greens_d", legend=False)
axes[1].set_title("Revenue by City", fontweight="bold")
axes[1].set_xlabel("Revenue")
axes[1].xaxis.set_major_formatter(lambda x, _: f"${x/1000:.0f}K")

plt.tight_layout()
plt.savefig("outputs/02_category_city_revenue.png")
plt.close()

#Q3: Do discounts actually change order size / basket behavior?
df["had_discount"] = df["Discount_Amount"] > 0
promo_compare = df.groupby("had_discount")["Total_Amount"].agg(["mean", "count"])
promo_compare.index = ["No Discount", "Discount Applied"]
print("\n" + "=" * 60)
print("DISCOUNT IMPACT")
print("=" * 60)
print(promo_compare)

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(x=promo_compare.index, y=promo_compare["mean"], hue=promo_compare.index,
            ax=ax, palette=["#94a3b8", "#f97316"], legend=False)
ax.set_title("Average Order Value: Discount vs. No Discount", fontweight="bold")
ax.set_ylabel("Average Order Value")
for i, v in enumerate(promo_compare["mean"]):
    ax.text(i, v + 5, f"${v:.2f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/03_discount_impact.png")
plt.close()


#Q4: Payment method & device -- how do people actually shop?
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

payment_counts = df["Payment_Method"].value_counts()
sns.barplot(x=payment_counts.values, y=payment_counts.index, hue=payment_counts.index,
            ax=axes[0], palette="Purples_d", legend=False)
axes[0].set_title("Orders by Payment Method", fontweight="bold")
axes[0].set_xlabel("Number of Orders")

device_rating = df.groupby("Device_Type")["Customer_Rating"].mean().sort_values(ascending=False)
sns.barplot(x=device_rating.index, y=device_rating.values, hue=device_rating.index,
            ax=axes[1], palette="Oranges_d", legend=False)
axes[1].set_title("Average Customer Rating by Device", fontweight="bold")
axes[1].set_ylabel("Average Rating (1-5)")
axes[1].set_ylim(0, 5)

plt.tight_layout()
plt.savefig("outputs/04_payment_device.png")
plt.close()

# Q5: Does delivery time affect customer satisfaction?
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.boxplot(data=df, x="Customer_Rating", y="Delivery_Time_Days",
            hue="Customer_Rating", palette="RdYlGn", legend=False, ax=ax)
ax.set_title("Delivery Time vs. Customer Rating", fontweight="bold")
ax.set_xlabel("Customer Rating (1-5)")
ax.set_ylabel("Delivery Time (days)")
plt.tight_layout()
plt.savefig("outputs/05_delivery_vs_rating.png")
plt.close()

corr = df["Delivery_Time_Days"].corr(df["Customer_Rating"])
print(f"\nCorrelation between delivery time and rating: {corr:.3f}")

#Q6: New vs returning customers -- who spends more?
returning_compare = df.groupby("Is_Returning_Customer")["Total_Amount"].agg(["mean", "count"])
returning_compare.index = ["New Customer", "Returning Customer"]
print("\n" + "=" * 60)
print("NEW VS RETURNING CUSTOMER SPEND")
print("=" * 60)
print(returning_compare)

print("\nAll EDA charts saved to outputs/")
