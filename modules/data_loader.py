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
    """Load CSV with auto encoding detection and common delimiter sniffing."""
    if encoding is None:
        encoding = detect_encoding(file_bytes)
    text = file_bytes.decode(encoding, errors="replace")
    # Try common delimiters
    for sep in [",", ";", "\t", "|"]:
        try:
            first_line = text.split("\n")[0]
            if sep in first_line:
                return pd.read_csv(StringIO(text), sep=sep, **kwargs)
        except Exception:
            continue
    return pd.read_csv(StringIO(text), **kwargs)


def load_excel(file_bytes: bytes, **kwargs) -> pd.DataFrame:
    """Load Excel file (.xls / .xlsx)."""
    return pd.read_excel(BytesIO(file_bytes), **kwargs)


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
