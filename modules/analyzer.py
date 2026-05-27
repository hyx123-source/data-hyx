"""
Module 3: Data Analysis Methods
RFM analysis, market basket analysis, time series, top-N, clustering, churn.
Source: Student + AI collaboration.
"""
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations


def rfm_summary(rfm_df: pd.DataFrame) -> dict:
    """Summarize RFM segmentation results.

    Args:
        rfm_df: Output of get_rfm_table() — one row per customer.

    Returns:
        dict with segment_counts, segment_stats, overall_stats
    """
    segments = rfm_df["Segment"].value_counts().to_dict()
    segment_stats = {}
    for seg in rfm_df["Segment"].unique():
        seg_data = rfm_df[rfm_df["Segment"] == seg]
        segment_stats[seg] = {
            "count": int(len(seg_data)),
            "avg_recency": round(float(seg_data["Recency"].mean()), 1),
            "avg_frequency": round(float(seg_data["Frequency"].mean()), 1),
            "avg_monetary": round(float(seg_data["Monetary"].mean()), 2),
            "total_revenue": round(float(seg_data["Monetary"].sum()), 2),
            "revenue_pct": round(
                float(seg_data["Monetary"].sum() / rfm_df["Monetary"].sum() * 100), 1
            ),
        }

    overall = {
        "total_customers": int(len(rfm_df)),
        "avg_recency": round(float(rfm_df["Recency"].mean()), 1),
        "avg_frequency": round(float(rfm_df["Frequency"].mean()), 1),
        "avg_monetary": round(float(rfm_df["Monetary"].mean()), 2),
        "total_revenue": round(float(rfm_df["Monetary"].sum()), 2),
    }
    return {
        "segment_counts": segments,
        "segment_stats": segment_stats,
        "overall_stats": overall,
    }


def top_n_analysis(df: pd.DataFrame, by: str = "TotalPrice", n: int = 10,
                   group_col: str = "Description") -> pd.DataFrame:
    """Top-N analysis: group by a column and rank by a metric.

    Args:
        df: Cleaned transaction DataFrame.
        by: Metric column name.
        n: Number of top items.
        group_col: Column to group by.

    Returns:
        DataFrame with columns: group_col, total_{by}, avg_{by}, count
    """
    agg_dict = {"total": ("TotalPrice", "sum")}
    if "Quantity" in df.columns:
        agg_dict["quantity"] = ("Quantity", "sum")
    if "UnitPrice" in df.columns:
        agg_dict["avg_unit_price"] = ("UnitPrice", "mean")
    if "InvoiceNo" in df.columns:
        agg_dict["transactions"] = ("InvoiceNo", "nunique")
    if "CustomerID" in df.columns:
        agg_dict["customers"] = ("CustomerID", "nunique")

    result = (
        df.groupby(group_col)
        .agg(**agg_dict)
        .sort_values("total", ascending=False)
        .head(n)
        .reset_index()
    )
    col_map = {"total": "TotalRevenue"}
    if "quantity" in agg_dict:
        col_map["quantity"] = "TotalQuantity"
    if "avg_unit_price" in agg_dict:
        col_map["avg_unit_price"] = "AvgUnitPrice"
    if "transactions" in agg_dict:
        col_map["transactions"] = "TransactionCount"
    if "customers" in agg_dict:
        col_map["customers"] = "CustomerCount"
    result.rename(columns=col_map, inplace=True)
    return result.round(2)


def country_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Sales analysis by country."""
    agg_dict = {"TotalRevenue": ("TotalPrice", "sum"), "AvgOrderValue": ("TotalPrice", "mean")}
    if "InvoiceNo" in df.columns:
        agg_dict["TransactionCount"] = ("InvoiceNo", "nunique")
    if "CustomerID" in df.columns:
        agg_dict["CustomerCount"] = ("CustomerID", "nunique")
    return (
        df.groupby("Country")
        .agg(**agg_dict)
        .sort_values("TotalRevenue", ascending=False)
        .reset_index()
        .round(2)
    )


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly sales trend analysis."""
    agg_dict = {"TotalRevenue": ("TotalPrice", "sum"), "AvgOrderValue": ("TotalPrice", "mean")}
    if "InvoiceNo" in df.columns:
        agg_dict["TransactionCount"] = ("InvoiceNo", "nunique")
    if "CustomerID" in df.columns:
        agg_dict["CustomerCount"] = ("CustomerID", "nunique")
    if "StockCode" in df.columns:
        agg_dict["UniqueProducts"] = ("StockCode", "nunique")
    trend = (
        df.groupby("YearMonth")
        .agg(**agg_dict)
        .sort_index()
        .reset_index()
    )
    return trend.round(2)


def hourly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly sales pattern."""
    agg_dict = {"TotalRevenue": ("TotalPrice", "sum")}
    if "InvoiceNo" in df.columns:
        agg_dict["TransactionCount"] = ("InvoiceNo", "nunique")
    return (
        df.groupby("Hour")
        .agg(**agg_dict)
        .reset_index()
    )


def weekday_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Sales by day of week."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    agg_dict = {"TotalRevenue": ("TotalPrice", "sum")}
    if "InvoiceNo" in df.columns:
        agg_dict["TransactionCount"] = ("InvoiceNo", "nunique")
    present = [d for d in order if d in df["WeekdayName"].unique()]
    trend = (
        df.groupby("WeekdayName")
        .agg(**agg_dict)
        .reindex(present)
        .reset_index()
    )
    return trend


def market_basket_analysis(df: pd.DataFrame, min_support: int = 20) -> list:
    """Simple market basket analysis: find frequently co-occurring products.

    Uses a simplified approach (pairwise co-occurrence counting)
    suitable for demonstration with the Online Retail dataset.

    Args:
        df: Cleaned transaction DataFrame.
        min_support: Minimum number of co-occurrences to include.

    Returns:
        List of dicts with keys: item_a, item_b, count, support
    """
    # Group items by invoice
    baskets = df.groupby("InvoiceNo")["Description"].apply(list)

    pair_counts = Counter()
    for items in baskets:
        # Deduplicate within same invoice
        unique_items = list(set(items))
        if len(unique_items) < 2:
            continue
        # Count all pairs
        for pair in combinations(sorted(unique_items), 2):
            pair_counts[pair] += 1

    total_baskets = len(baskets)
    results = []
    for (item_a, item_b), count in pair_counts.most_common(50):
        if count < min_support:
            continue
        results.append({
            "item_a": item_a,
            "item_b": item_b,
            "count": count,
            "support": round(count / total_baskets * 100, 2),
        })

    return results


def product_clustering(rfm_df: pd.DataFrame, n_clusters: int = 4) -> dict:
    """KMeans customer clustering based on RFM.

    Returns dict with cluster_labels (list), cluster_stats, and inertia.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    features = rfm_df[["Recency", "Frequency", "Monetary"]].copy()
    # Log transform to handle skewed distributions
    features_log = np.log1p(features)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_log)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    rfm_df = rfm_df.copy()
    rfm_df["Cluster"] = labels

    cluster_stats = {}
    for c in range(n_clusters):
        cluster_data = rfm_df[rfm_df["Cluster"] == c]
        cluster_stats[int(c)] = {
            "count": int(len(cluster_data)),
            "avg_recency": round(float(cluster_data["Recency"].mean()), 1),
            "avg_frequency": round(float(cluster_data["Frequency"].mean()), 1),
            "avg_monetary": round(float(cluster_data["Monetary"].mean()), 2),
            "total_revenue": round(float(cluster_data["Monetary"].sum()), 2),
        }

    return {
        "cluster_labels": labels.tolist(),
        "cluster_stats": cluster_stats,
        "inertia": float(kmeans.inertia_),
    }


def search_products(df: pd.DataFrame, keyword: str, n: int = 10) -> pd.DataFrame:
    """Search products by keyword in Description."""
    mask = df["Description"].str.contains(keyword, case=False, na=False)
    agg_dict = {"TotalRevenue": ("TotalPrice", "sum")}
    if "Quantity" in df.columns:
        agg_dict["TotalSold"] = ("Quantity", "sum")
    result = (
        df[mask]
        .groupby("Description")
        .agg(**agg_dict)
        .sort_values("TotalRevenue", ascending=False)
        .head(n)
        .reset_index()
    )
    return result
