from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def validate_dataset(df: pd.DataFrame, target_column: Optional[str]) -> Dict:
    """
    Hard gate before running heavy analysis.
    Returns:
      {"valid": bool, "reason": str, "details": {...}}
    """
    if df is None or df.empty:
        return {
            "valid": False,
            "reason": "Dataset is empty.",
            "details": {},
        }

    rows, cols = df.shape
    if rows < 20:
        return {
            "valid": False,
            "reason": f"Dataset has only {rows} rows; at least 20 rows are required.",
            "details": {"rows": rows, "columns": cols},
        }

    if cols < 2:
        return {
            "valid": False,
            "reason": f"Dataset has only {cols} column(s); at least 2 columns are required.",
            "details": {"rows": rows, "columns": cols},
        }

    if target_column and target_column not in df.columns:
        return {
            "valid": False,
            "reason": f"Target column '{target_column}' is not present in dataset.",
            "details": {"target_column": target_column},
        }

    # Optional: require at least one numeric feature
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) == 0:
        return {
            "valid": False,
            "reason": "No numeric columns found; regression analysis is not reliable.",
            "details": {},
        }

    return {"valid": True, "reason": "", "details": {"rows": rows, "columns": cols}}