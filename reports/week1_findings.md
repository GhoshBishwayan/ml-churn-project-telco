# Week 1 – Data Understanding & EDA

## Dataset
- Path: `/Users/rahuldas/Documents/ml-churn-project-telco/data/telco_churn.csv`
- Rows: **7043**
- Columns: **20**
- Target: **Churn**
- Churn Rate: **26.54%**

## Data Quality
### Missing Values (after cleaning)
- `TotalCharges`: 11

## Key Observations (auto-generated)
- Highest churn by **Contract**: `Month-to-month` (~42.7%).
- Highest churn by **Payment Method**: `Electronic check` (~45.3%).
- Highest churn by **Internet Service**: `Fiber optic` (~41.9%).

## Numeric Signals (quick comparison)
- Mean **tenure**: churn=**17.98**, non-churn=**37.57**
- Mean **MonthlyCharges**: churn=**74.44**, non-churn=**61.27**
- Mean **TotalCharges**: churn=**1531.80**, non-churn=**2555.34**

## Saved Figures
- 01_churn_distribution.png
- cat_churn_rate_Contract.png
- cat_churn_rate_Dependents.png
- cat_churn_rate_InternetService.png
- cat_churn_rate_OnlineSecurity.png
- cat_churn_rate_PaperlessBilling.png
- cat_churn_rate_Partner.png
- cat_churn_rate_PaymentMethod.png
- cat_churn_rate_SeniorCitizen.png
- cat_churn_rate_TechSupport.png
- cat_churn_rate_gender.png
- cat_counts_Contract.png
- cat_counts_Dependents.png
- cat_counts_InternetService.png
- cat_counts_OnlineSecurity.png
- cat_counts_PaperlessBilling.png
- cat_counts_Partner.png
- cat_counts_PaymentMethod.png
- cat_counts_SeniorCitizen.png
- cat_counts_TechSupport.png
- cat_counts_gender.png
- corr_heatmap_numeric.png
- num_box_MonthlyCharges.png
- num_box_TotalCharges.png
- num_box_tenure.png
- num_hist_MonthlyCharges.png
- num_hist_TotalCharges.png
- num_hist_tenure.png

## Suggested Next Steps (Week 2)
- Encode categorical variables (one-hot)
- Handle missing numeric values (impute)
- Create train/test split with stratification
- Start feature engineering (tenure buckets, charge ratios, contract flags)
