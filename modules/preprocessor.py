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


# Column name mappings for auto-detection (canonical -> common alternatives)
COLUMN_ALIASES = {
    "CustomerID": ["customerid", "customer_id", "customer", "client_id", "clientid", "user_id", "userid"],
    "Description": ["description", "product", "product_name", "productname", "item", "name"],
    "Quantity": ["quantity", "qty", "amount", "units", "volume"],
    "UnitPrice": ["unitprice", "unit_price", "price", "cost", "单价", "price_per_unit"],
    "InvoiceNo": ["invoiceno", "invoice_no", "invoice", "order_id", "orderid", "transaction_id", "tid"],
    "InvoiceDate": ["invoicedate", "invoice_date", "date", "datetime", "timestamp", "order_date", "time"],
    "StockCode": ["stockcode", "stock_code", "sku", "product_id", "productid", "item_code"],
    "Country": ["country", "nation", "region", "area", "国家", "地区"],
    "TotalPrice": ["totalprice", "total_price", "revenue", "sales", "amount"],
}

REQUIRED_ONLINE_RETAIL = ["CustomerID", "Quantity", "UnitPrice", "InvoiceNo", "InvoiceDate"]


def _normalize_column_name(col: str) -> str:
    """Map a column name to its canonical form, or return original."""
    col_lower = col.lower().strip()
    for canonical, aliases in COLUMN_ALIASES.items():
        if col_lower in aliases or col_lower == canonical.lower():
            return canonical
    return col


def auto_detect_columns(df: pd.DataFrame) -> dict:
    """Detect and map DataFrame columns to canonical Online Retail names.

    Returns a dict: {canonical_name: actual_column_name_in_df} for matched columns.
    """
    mapping = {}
    for actual_col in df.columns:
        canonical = _normalize_column_name(actual_col)
        if canonical != actual_col:
            mapping[canonical] = actual_col
        else:
            mapping[actual_col] = actual_col
    return mapping


def is_online_retail_dataset(df: pd.DataFrame) -> bool:
    """Check if the dataset looks like Online Retail format."""
    col_map = auto_detect_columns(df)
    detected = set(col_map.keys())
    required = {"Quantity", "UnitPrice", "InvoiceDate"}
    return required.issubset(detected)


def preprocess_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline for the Online Retail dataset.

    Auto-detects column names and falls back to generic preprocessing
    if the dataset doesn't match the expected format.
    """
    df = df.copy()

    # Auto-detect columns
    col_map = auto_detect_columns(df)

    # Check if we have the minimum required columns for Online Retail pipeline
    has_customer = "CustomerID" in col_map
    has_quantity = "Quantity" in col_map
    has_price = "UnitPrice" in col_map
    has_date = "InvoiceDate" in col_map
    has_invoice = "InvoiceNo" in col_map

    # If missing core numeric columns, fall back to generic
    if not (has_quantity and has_price and has_date):
        return preprocess_generic(df)

    c_quantity = col_map["Quantity"]
    c_price = col_map["UnitPrice"]
    c_date = col_map["InvoiceDate"]

    # 1. Drop critical missing (CustomerID and Description if present)
    if has_customer:
        df = df.dropna(subset=[col_map["CustomerID"]])
        df[col_map["CustomerID"]] = df[col_map["CustomerID"]].astype(int)

    desc_col = col_map.get("Description")
    if desc_col and desc_col in df.columns:
        df = df.dropna(subset=[desc_col])

    # 2. Filter valid transactions
    df = df[(df[c_quantity] > 0) & (df[c_price] > 0)]

    # 3. Mark cancelled orders (if InvoiceNo present)
    if has_invoice:
        df["IsCancelled"] = df[col_map["InvoiceNo"]].astype(str).str.startswith("C")
    else:
        df["IsCancelled"] = False

    # 4. Create TotalPrice
    df["TotalPrice"] = df[c_quantity] * df[c_price]

    # 5. Date features
    df["InvoiceDate"] = pd.to_datetime(df[c_date], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["Weekday"] = df["InvoiceDate"].dt.weekday
    df["WeekdayName"] = df["InvoiceDate"].dt.day_name()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # Rename key columns to canonical names for downstream compatibility
    if has_customer and col_map["CustomerID"] != "CustomerID":
        df["CustomerID"] = df[col_map["CustomerID"]]
    if has_invoice and col_map["InvoiceNo"] != "InvoiceNo":
        df["InvoiceNo"] = df[col_map["InvoiceNo"]]
    country_col = col_map.get("Country")
    if country_col and country_col != "Country":
        df["Country"] = df[country_col]
    elif "Country" not in df.columns:
        df["Country"] = "Unknown"
    desc_col = col_map.get("Description")
    if desc_col and desc_col != "Description":
        df["Description"] = df[desc_col]
    elif "Description" not in df.columns:
        stock_col = col_map.get("StockCode")
        if stock_col:
            df["Description"] = df[stock_col].astype(str)
        else:
            df["Description"] = "Unknown"
    stock_col = col_map.get("StockCode")
    if stock_col and stock_col != "StockCode":
        df["StockCode"] = df[stock_col]
    elif "StockCode" not in df.columns:
        if "Description" in df.columns:
            df["StockCode"] = df["Description"]
        else:
            df["StockCode"] = "SKU-UNKNOWN"

    # Ensure Quantity and UnitPrice canonical columns exist (for analyzer compatibility)
    if c_quantity != "Quantity":
        df["Quantity"] = df[c_quantity]
    elif "Quantity" not in df.columns:
        df["Quantity"] = 1
    if c_price != "UnitPrice":
        df["UnitPrice"] = df[c_price]
    elif "UnitPrice" not in df.columns:
        df["UnitPrice"] = df["TotalPrice"] / df["Quantity"].replace(0, 1)

    # 6. Compute RFM (only if CustomerID exists)
    if "CustomerID" in df.columns:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        rfm = df.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique") if "InvoiceNo" in df.columns else ("TotalPrice", "count"),
            Monetary=("TotalPrice", "sum"),
        ).reset_index()

        try:
            rfm["R_Score"] = pd.qcut(rfm["Recency"], q=4, labels=[4, 3, 2, 1])
            rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4])
            rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4])
        except ValueError:
            rfm["R_Score"] = 1
            rfm["F_Score"] = 1
            rfm["M_Score"] = 1

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
        df = df.merge(rfm, on="CustomerID", how="left")
    else:
        df["Recency"] = 0
        df["Frequency"] = 1
        df["Monetary"] = df["TotalPrice"]
        df["R_Score"] = 1
        df["F_Score"] = 1
        df["M_Score"] = 1
        df["RFM_Score"] = 3
        df["Segment"] = "Unknown"

    return df


def preprocess_generic(df: pd.DataFrame) -> pd.DataFrame:
    """Generic preprocessing for any tabular dataset.

    Handles: missing values, date parsing, numeric column detection,
    categorical encoding prep. Does NOT assume Online Retail columns.
    """
    df = df.copy()

    # Drop fully empty rows and columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Parse datetime columns
    for col in df.columns:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:
                    df[col] = parsed
            except Exception:
                pass

    # Fill numeric missing with median
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical missing with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                df[col].fillna(mode_vals[0], inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)

    # Add fake Online Retail compatible columns for downstream compatibility
    if "TotalPrice" not in df.columns:
        # Use the first numeric column as TotalPrice
        if len(num_cols) > 0:
            df["TotalPrice"] = df[num_cols[0]]
        else:
            df["TotalPrice"] = 1

    if "InvoiceDate" not in df.columns:
        date_cols = df.select_dtypes(include=["datetime64"]).columns
        if len(date_cols) > 0:
            df["InvoiceDate"] = df[date_cols[0]]
        else:
            df["InvoiceDate"] = pd.Timestamp.now()

    if "IsCancelled" not in df.columns:
        df["IsCancelled"] = False

    if "CustomerID" not in df.columns:
        # Use first column as fake customer ID
        df["CustomerID"] = range(1, len(df) + 1)

    if "InvoiceNo" not in df.columns:
        df["InvoiceNo"] = range(1, len(df) + 1)

    if "Country" not in df.columns:
        df["Country"] = "Unknown"

    if "Description" not in df.columns:
        df["Description"] = "Item-" + df.index.astype(str)

    if "StockCode" not in df.columns:
        df["StockCode"] = "SKU-" + df.index.astype(str)

    if "Quantity" not in df.columns:
        df["Quantity"] = 1

    if "UnitPrice" not in df.columns:
        df["UnitPrice"] = df["TotalPrice"]

    # Date features
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["Weekday"] = df["InvoiceDate"].dt.weekday
    df["WeekdayName"] = df["InvoiceDate"].dt.day_name()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # RFM
    df["Recency"] = 0
    df["Frequency"] = 1
    df["Monetary"] = df["TotalPrice"]
    df["R_Score"] = 1
    df["F_Score"] = 1
    df["M_Score"] = 1
    df["RFM_Score"] = 3
    df["Segment"] = "通用数据"

    return df


def get_clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Return only non-cancelled transactions."""
    return df[~df["IsCancelled"]].copy()


def get_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to get one row per customer with RFM info."""
    return df[["CustomerID", "Recency", "Frequency", "Monetary",
               "R_Score", "F_Score", "M_Score", "RFM_Score", "Segment"]] \
        .drop_duplicates(subset="CustomerID").reset_index(drop=True)
