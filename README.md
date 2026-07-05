# E-Commerce Customer Behavior Analytics

Analysis of a real e-commerce transactions dataset: 17,049 orders from
5,000 customers across 10 Turkish cities, January 2023 – March 2024.
Covers the full pipeline a data/business analyst intern is actually asked
to do: explore the data, segment customers, and build a predictive model
— then translate all of it into recommendations a manager could act on.

## The business questions

1. How is revenue trending, and which categories/cities drive it?
2. Do discounts actually increase order value?
3. Does delivery speed affect customer satisfaction?
4. Who are our most valuable customers, and who's at risk of leaving?
5. Can we predict churn before it happens?

## Project structure

```
ecommerce-analytics/
├── 01_eda.py                    # exploratory analysis + business-question charts
├── 02_customer_segmentation.py  # RFM analysis + KMeans customer segments
├── 03_churn_model.py            # RandomForest churn prediction + evaluation
├── data/
│   ├── orders.csv               # the raw dataset
│   └── customer_segments.csv    # generated: RFM scores + segment labels
└── outputs/                     # all charts (PNG)
```

Run in order:
```bash
pip install -r requirements.txt
python 01_eda.py
python 02_customer_segmentation.py
python 03_churn_model.py
```

## Key findings

**Revenue is concentrated in a few categories and cities**
Electronics alone brought in **$10.48M** of the **$21.78M** total revenue —
nearly half — with Home & Garden ($4.02M) and Sports ($3.21M) a distant
second and third. Istanbul leads all cities at **$5.65M**, roughly double
Ankara, the next closest. Any inventory or marketing prioritization should
follow this concentration, not treat all categories/cities equally.

**Discounts didn't increase order size — a genuinely useful negative result**
Orders with a discount applied averaged **$1,178**, actually *lower* than
the **$1,338** average for orders with no discount. This is a case where
digging into the "why" matters more than the headline number: it likely
reflects which categories get discounted rather than discounts suppressing
spend directly — worth flagging to a marketing team rather than assuming
"discounts = more revenue" by default.

**Delivery time has no measurable effect on customer rating**
Correlation between delivery time and rating: **-0.008** — essentially
zero. That's a useful (if less exciting) finding: it suggests rating in
this dataset is driven by something other than logistics performance, so
a business shouldn't assume faster shipping alone will move satisfaction
scores.

**Customer value is concentrated (RFM segmentation)**
- **Loyal Customers** (~40%) — steady frequency and spend, the backbone
  of the business.
- **At Risk** (~23%) — used to buy, activity has cooled off. The clearest
  actionable win-back target.
- **New / Low-Engagement** (~20%) — early-stage, unproven.
- **Champions** (~17%) — highest frequency and spend, most recent activity.
  Worth protecting with loyalty perks.

**Churn is only weakly predictable from this data — and that's a real finding too**
A RandomForest churn model (customer inactive in the final 90 days) scored
**ROC-AUC ≈ 0.57** — only marginally better than a coin flip. I checked
whether this was a modeling mistake rather than a real property of the
data: I tested inter-purchase-interval-based features, and confirmed that
even *recency of last order* barely correlates with future churn
(correlation ≈ 0.06–0.08) in this dataset. **Order frequency** was the
strongest signal available, but even that was modest.

This is worth stating plainly rather than dressing up: the behavioral
columns available (session duration, pages viewed, rating, delivery time)
simply don't carry strong churn signal here. In a real analyst role, the
right move isn't to force a falsely impressive model — it's to report this
honestly and recommend what additional data would help (e.g. marketing
engagement/email opens, customer service contacts, cart abandonment
events), which are exactly the kinds of signals that tend to predict churn
in practice but aren't present in this dataset.

## Methodology notes 

- **No data leakage**: churn features are built only from activity *before*
  a 90-day cutoff; the label comes from activity *after* it.
- **Log-transforming monetary value** before clustering, since a handful
  of big spenders would otherwise dominate the distance calculation.
- **Elbow method** used to justify k=4 clusters rather than picking a
  number arbitrarily.
- **class_weight="balanced"** in the classifier (churn rate here is ~54%,
  close to balanced, but this is good practice regardless).
- Reported a weak result (AUC 0.57) honestly instead of cherry-picking
  a rosier metric or overfitting to inflate it — a real analyst has to be
  able to tell a stakeholder "this doesn't work well, here's why."

## What I'd do next 
- Get access to marketing/engagement data (email opens, ad clicks, support
  tickets) to see if churn becomes more predictable with richer features
- A/B test framework to properly isolate *why* discounted orders skew lower
  (category mix vs. discount depth vs. customer type)
- Build a small dashboard (Streamlit) so segments update as new orders come in

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
# e-commerce_analysis
