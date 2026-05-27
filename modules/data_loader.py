"""
Module 1: Data File Loading
Supports CSV, Excel, JSON with auto encoding/charset detection.
Source: Student + AI collaboration.
"""
import pandas as pd
import os
from io import BytesIO, StringIO

try:
    import chardet
    _HAS_CHARDET = True
except ImportError:
    _HAS_CHARDET = False


def detect_encoding(file_bytes: bytes) -> str:
    """Auto-detect file encoding using chardet."""
    if _HAS_CHARDET:
        result = chardet.detect(file_bytes[:100000])
        return result.get("encoding", "utf-8") or "utf-8"
    return "utf-8"


def load_csv(file_bytes: bytes, encoding: str = None, **kwargs) -> pd.DataFrame:
    """Load CSV with auto encoding detection, delimiter sniffing, and error recovery."""
    if encoding is None:
        encoding = detect_encoding(file_bytes)

    # Try multiple encodings
    encodings_to_try = [encoding]
    for fallback_enc in ["utf-8", "gbk", "gb2312", "latin-1", "iso-8859-1"]:
        if fallback_enc not in encodings_to_try:
            encodings_to_try.append(fallback_enc)

    text = None
    used_encoding = encoding
    for enc in encodings_to_try:
        try:
            text = file_bytes.decode(enc, errors="replace")
            # Check if decoding produced reasonable Chinese content or minimal replacement chars
            if text.count('�') < len(text) * 0.1:
                used_encoding = enc
                break
        except Exception:
            continue

    if text is None:
        text = file_bytes.decode("utf-8", errors="replace")

    # Detect delimiter from first few lines
    delimiters = [",", ";", "\t", "|"]
    lines_sample = "\n".join(text.split("\n")[:10])
    best_sep = ","
    for sep in delimiters:
        if sep in lines_sample:
            best_sep = sep
            break

    # Try multiple parsing strategies
    strategies = [
        {"sep": best_sep, "on_bad_lines": "skip", "encoding": used_encoding},
        {"sep": best_sep, "on_bad_lines": "warn", "encoding": used_encoding},
        {"sep": None, "encoding": used_encoding, "engine": "python"},  # auto-detect with python engine
        {"sep": best_sep, "encoding": used_encoding},
    ]

    last_error = None
    for strat in strategies:
        try:
            kwargs_copy = {k: v for k, v in kwargs.items() if k not in strat}
            return pd.read_csv(StringIO(text), **strat, **kwargs_copy)
        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"CSV 解析失败: {last_error}")


def load_excel(file_bytes: bytes, **kwargs) -> pd.DataFrame:
    """Load Excel file with auto header detection and numeric string cleaning."""
    # First pass: load without header to detect structure
    raw = pd.read_excel(BytesIO(file_bytes), header=None)

    # Auto-detect best header row: the row with the most non-null, non-numeric, non-empty cells
    best_row = 0
    best_score = 1
    for i in range(min(5, len(raw))):
        row_vals = raw.iloc[i].dropna().astype(str).tolist()
        # Score: count of reasonably named cells (not pure numbers, not too long)
        score = sum(1 for v in row_vals
                    if len(v) > 1 and len(v) < 30 and not v.replace(".", "").replace("-", "").isdigit())
        if score > best_score:
            best_score = score
            best_row = i

    # Reload with detected header
    df = pd.read_excel(BytesIO(file_bytes), header=best_row)

    # Drop columns that are fully unnamed or empty
    unnamed_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Unnamed:")]
    df = df.drop(columns=[c for c in unnamed_cols if df[c].isnull().all()])

    # Drop rows that are completely empty
    df = df.dropna(how="all").reset_index(drop=True)

    # Drop rows where ALL columns except 1-2 are NaN (metadata rows like titles)
    df = df[df.notna().sum(axis=1) >= max(2, len(df.columns) // 2)].reset_index(drop=True)

    # Clean numeric strings: "1.2元" → 1.2, "¥100" → 100, "1,234.56" → 1234.56
    df = _clean_numeric_strings(df)

    # Re-evaluate column types after cleaning
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == 'str':
            df[col] = _infer_column_type(df[col])

    return df


def _clean_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip common units/symbols from numeric-looking strings.

    Two-pass strategy: first try matching known unit patterns, then fall back
    to stripping all non-numeric characters.
    """
    import re
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object and str(df[col].dtype) != 'str':
            continue
        sample = df[col].dropna().head(20).astype(str)
        # Pass 1: check if values look like numbers with optional unit suffix
        has_digit = sample.str.contains(r'\d')
        if has_digit.mean() < 0.5:
            continue

        # Pass 2: try stripping known unit chars first, then fall back to brute force
        cleaned = df[col].astype(str).str.replace(r'[¥$€£元角分个只件台套次人天月年万千百十亿‰%\s]', '', regex=True)
        cleaned = cleaned.str.replace(',', '')
        numeric = pd.to_numeric(cleaned, errors='coerce')

        # If the known-unit approach mostly failed, try stripping all non-[0-9.] chars
        if numeric.notna().sum() < len(cleaned) * 0.7:
            cleaned2 = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            numeric2 = pd.to_numeric(cleaned2, errors='coerce')
            if numeric2.notna().sum() >= len(cleaned2) * 0.7:
                df[col] = numeric2
                continue

        if numeric.notna().sum() >= len(cleaned) * 0.7:
            df[col] = numeric

    return df


def _infer_column_type(series: pd.Series) -> pd.Series:
    """Try to convert a series to numeric or datetime."""
    # Try numeric
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().sum() > len(series) * 0.8:
        return numeric
    # Try datetime (suppress dateutil fallback warning for non-date strings)
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            dt = pd.to_datetime(series, errors='coerce')
        if dt.notna().sum() > len(series) * 0.5:
            return dt
    except Exception:
        pass
    return series


def load_json(file_bytes: bytes, encoding: str = None, **kwargs) -> pd.DataFrame:
    """Load JSON file (record-oriented or column-oriented)."""
    if encoding is None:
        encoding = detect_encoding(file_bytes)
    text = file_bytes.decode(encoding, errors="replace")
    return pd.read_json(StringIO(text), **kwargs)


def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Auto-detect file format and load into a DataFrame.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used to guess format).

    Returns:
        pd.DataFrame
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".csv", ".txt", ".tsv", ""):
        return load_csv(file_bytes)
    elif ext in (".xls", ".xlsx"):
        return load_excel(file_bytes)
    elif ext == ".json":
        return load_json(file_bytes)
    else:
        # Fallback: try CSV first, then Excel
        try:
            return load_csv(file_bytes)
        except Exception:
            try:
                return load_excel(file_bytes)
            except Exception:
                raise ValueError(f"Cannot parse file: {filename}")


def get_data_summary(df: pd.DataFrame) -> dict:
    """Generate a comprehensive data overview.

    Returns a dict with keys:
        shape, columns, dtypes, missing, missing_pct, duplicated, memory_usage,
        numeric_stats, categorical_stats, sample_rows
    """
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicated": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "sample_rows": df.head(5).to_dict("records"),
    }

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        summary["numeric_stats"] = df[num_cols].describe().to_dict()

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        summary["categorical_stats"] = {
            col: df[col].value_counts().head(10).to_dict() for col in cat_cols[:5]
        }
        summary["n_unique_categorical"] = {
            col: int(df[col].nunique()) for col in cat_cols
        }

    return summary
