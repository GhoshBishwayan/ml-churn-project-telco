from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CFG
from src.load_data import load_data
from src.clean_data import clean_data, data_quality_summary

sns.set_context("talk")


def _ensure_dirs():
    CFG.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CFG.FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(filename: str):
    out_path = CFG.FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def churn_rate(df: pd.DataFrame) -> float:
    vc = df[CFG.TARGET_COL].value_counts(normalize=True)
    return float(vc.get("Yes", 0.0))


def _plot_target_distribution(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    sns.countplot(x=CFG.TARGET_COL, data=df)
    plt.title("Churn Distribution")
    _save_fig("01_churn_distribution.png")


def _plot_churn_rate_by_category(df: pd.DataFrame, col: str, top_n: int | None = None):
    if col not in df.columns:
        return

    # Compute churn rate by category
    tmp = df.groupby(col)[CFG.TARGET_COL].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)
    if top_n:
        tmp = tmp.head(top_n)

    plt.figure(figsize=(10, 6))
    tmp.plot(kind="bar")
    plt.ylabel("Churn Rate")
    plt.title(f"Churn Rate by {col}")
    _save_fig(f"cat_churn_rate_{col}.png")


def _plot_count_by_category(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return
    plt.figure(figsize=(11, 6))
    sns.countplot(x=col, hue=CFG.TARGET_COL, data=df)
    plt.xticks(rotation=30, ha="right")
    plt.title(f"{col} vs Churn (Counts)")
    _save_fig(f"cat_counts_{col}.png")


def _plot_numeric_box(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=CFG.TARGET_COL, y=col, data=df)
    plt.title(f"{col} by Churn")
    _save_fig(f"num_box_{col}.png")


def _plot_numeric_hist(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return
    plt.figure(figsize=(9, 6))
    sns.histplot(data=df, x=col, hue=CFG.TARGET_COL, kde=True, element="step")
    plt.title(f"{col} Distribution by Churn")
    _save_fig(f"num_hist_{col}.png")


def _plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] < 2:
        return

    corr = num_df.corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numeric Features)")
    _save_fig("corr_heatmap_numeric.png")


def _build_week1_findings(df: pd.DataFrame, quality: dict) -> str:
    rate = churn_rate(df) * 100.0

    # Simple, safe “top churn” indicators from common columns if present
    insights = []

    def add_top_churn(col: str, label: str):
        if col in df.columns:
            tmp = df.groupby(col)[CFG.TARGET_COL].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)
            if tmp.shape[0] > 0:
                top_cat = tmp.index[0]
                top_rate = float(tmp.iloc[0]) * 100.0
                insights.append(f"- Highest churn by **{label}**: `{top_cat}` (~{top_rate:.1f}%).")

    add_top_churn(CFG.CONTRACT_COL, "Contract")
    add_top_churn(CFG.PAYMENT_METHOD_COL, "Payment Method")
    add_top_churn(CFG.INTERNET_SERVICE_COL, "Internet Service")

    # Numeric quick insight
    numeric_ins = []
    for col in [CFG.TENURE_COL, CFG.MONTHLY_CHARGES_COL, CFG.TOTAL_CHARGES_COL]:
        if col in df.columns and df[col].notna().any():
            churn_mean = df.loc[df[CFG.TARGET_COL] == "Yes", col].mean()
            non_mean = df.loc[df[CFG.TARGET_COL] == "No", col].mean()
            numeric_ins.append(f"- Mean **{col}**: churn=**{churn_mean:.2f}**, non-churn=**{non_mean:.2f}**")

    missing_cols = quality["missing_columns"]
    missing_text = "None" if not missing_cols else "\n".join([f"- `{k}`: {v}" for k, v in missing_cols.items()])

    plots_list = sorted([p.name for p in CFG.FIGURES_DIR.glob("*.png")])
    plots_md = "\n".join([f"- {p}" for p in plots_list]) if plots_list else "- (No plots saved)"

    md = f"""# Week 1 – Data Understanding & EDA

## Dataset
- Path: `{CFG.DATA_PATH}`
- Rows: **{quality["num_rows"]}**
- Columns: **{quality["num_columns"]}**
- Target: **{CFG.TARGET_COL}**
- Churn Rate: **{rate:.2f}%**

## Data Quality
### Missing Values (after cleaning)
{missing_text}

## Key Observations (auto-generated)
{os.linesep.join(insights) if insights else "- (Not enough categorical columns found for auto-insights.)"}

## Numeric Signals (quick comparison)
{os.linesep.join(numeric_ins) if numeric_ins else "- (Not enough numeric columns found.)"}

## Saved Figures
{plots_md}

## Suggested Next Steps (Week 2)
- Encode categorical variables (one-hot)
- Handle missing numeric values (impute)
- Create train/test split with stratification
- Start feature engineering (tenure buckets, charge ratios, contract flags)
"""
    return md


def run_eda():
    _ensure_dirs()

    print(f"[INFO] Loading data from: {CFG.DATA_PATH}")
    df_raw = load_data(CFG.DATA_PATH)

    print("[INFO] Cleaning data...")
    df = clean_data(
        df_raw,
        target_col=CFG.TARGET_COL,
        id_cols=CFG.ID_COLS,
        total_charges_col=CFG.TOTAL_CHARGES_COL,
    )
    quality = data_quality_summary(df)

    print("[INFO] Basic info:")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Churn distribution:\n", df[CFG.TARGET_COL].value_counts())
    print("Churn rate:", f"{churn_rate(df)*100:.2f}%")

    # Plots: target distribution
    print("[INFO] Generating plots...")
    _plot_target_distribution(df)

    # Categorical plots (common telco fields, but safe if missing)
    for col in [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        CFG.CONTRACT_COL,
        CFG.PAYMENT_METHOD_COL,
        CFG.INTERNET_SERVICE_COL,
        "OnlineSecurity",
        "TechSupport",
        "PaperlessBilling",
    ]:
        if col in df.columns:
            _plot_count_by_category(df, col)
            _plot_churn_rate_by_category(df, col, top_n=20)

    # Numeric plots
    for col in [CFG.TENURE_COL, CFG.MONTHLY_CHARGES_COL, CFG.TOTAL_CHARGES_COL]:
        if col in df.columns:
            _plot_numeric_box(df, col)
            _plot_numeric_hist(df, col)

    # Correlation for numeric columns
    _plot_correlation_heatmap(df)

    # Write week1 report
    print(f"[INFO] Writing report to: {CFG.WEEK1_REPORT_PATH}")
    report_md = _build_week1_findings(df, quality)
    CFG.WEEK1_REPORT_PATH.write_text(report_md, encoding="utf-8")

    print("[DONE] Week 1 EDA complete.")
    print(f"Figures saved to: {CFG.FIGURES_DIR}")
    print(f"Report saved to: {CFG.WEEK1_REPORT_PATH}")
