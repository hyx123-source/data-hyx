"""
Module 2: Data Preprocessing
Missing value handling, outlier detection, feature engineering, RFM computation.
Source: Student + AI collaboration.
"""
import pandas as pd
import numpy as np
from datetime import datetime


def clean_missing(df: pd.DataFrame, strategy: dict = None) -> pd.DataFrame:
    """Handle missing values based on per-column strategies.

    Args:
        df: Input DataFrame.
        strategy: Dict mapping column name -> strategy.
                  Strategies: 'drop', 'mean', 'median', 'mode', 'zero', 'ffill', or a constant value.

    Returns:
        Cleaned DataFrame (copy).
    """
    if strategy is None:
        strategy = {}

    df = df.copy()

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        s = strategy.get(col, "drop")  # default: drop rows with missing

        if s == "drop":
            df = df.dropna(subset=[col])
        elif s == "mean" and df[col].dtype in ("float64", "int64"):
            df[col].fillna(df[col].mean(), inplace=True)
        elif s == "median" and df[col].dtype in ("float64", "int64"):
            df[col].fillna(df[col].median(), inplace=True)
        elif s == "mode":
            df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 0, inplace=True)
        elif s == "zero":
            df[col].fillna(0, inplace=True)
        elif s == "ffill":
            df[col].fillna(method="ffill", inplace=True)
        else:
            df[col].fillna(s, inplace=True)

    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Remove rows with outliers in specified columns (IQR method).

    Outliers are defined as values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        if df[col].dtype not in ("float64", "int64"):
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df


def remove_outliers_zscore(df: pd.DataFrame, columns: list, threshold: float = 3.0) -> pd.DataFrame:
    """Remove rows with outliers in specified columns (Z-score method)."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        if df[col].dtype not in ("float64", "int64"):
            continue
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z <= threshold]
    return df


def preprocess_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline for the Online Retail dataset.

    Steps:
    1. Drop rows with missing CustomerID (can't do RFM without it)
    2. Drop rows with missing Description
    3. Filter out negative/zero Quantity and UnitPrice
    4. Mark cancelled orders (InvoiceNo starts with 'C')
    5. Create TotalPrice = Quantity * UnitPrice
    6. Parse InvoiceDate to datetime, extract year/month/day/hour/weekday
    7. Compute RFM (Recency, Frequency, Monetary) per customer
    """
    df = df.copy()

    # 1-2. Drop critical missing
    df = df.dropna(subset=["CustomerID", "Description"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # 3. Filter valid transactions
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # 4. Mark cancelled orders
    df["IsCancelled"] = df["InvoiceNo"].astype(str).str.startswith("C")

    # 5. Total price
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # 6. Date features
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["Weekday"] = df["InvoiceDate"].dt.weekday
    df["WeekdayName"] = df["InvoiceDate"].dt.day_name()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # 7. Compute RFM
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    # RFM scoring (1-4 quartile based)
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=4, labels=[4, 3, 2, 1])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4])
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(int) + rfm["F_Score"].astype(int) + rfm["M_Score"].astype(int)
    )

    def segment_rfm(score):
        if score >= 10:
            return "Champions"
        elif score >= 8:
            return "Loyal Customers"
        elif score >= 6:
            return "Potential Loyalists"
        elif score >= 4:
            return "At Risk"
        else:
            return "Lost"

    rfm["Segment"] = rfm["RFM_Score"].apply(segment_rfm)

    # Merge RFM back to main df (keep per-customer info)
    df = df.merge(rfm, on="CustomerID", how="left")

    return df


def get_clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Return only non-cancelled transactions."""
    return df[~df["IsCancelled"]].copy()


def get_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to get one row per customer with RFM info."""
    return df[["CustomerID", "Recency", "Frequency", "Monetary",
               "R_Score", "F_Score", "M_Score", "RFM_Score", "Segment"]] \
        .drop_duplicates(subset="CustomerID").reset_index(drop=True)
