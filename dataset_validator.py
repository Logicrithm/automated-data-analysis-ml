from __future__ import annotations

import pandas as pd


MIN_ROWS = 10
MIN_COLUMNS = 2


def validate_dataset(df: pd.DataFrame, target_column: str | None = None) -> dict:
    """Validate minimum dataset requirements for reliable analysis."""
    if df is None or df.empty:
        return {"valid": False, "reason": "Dataset is empty"}

    if len(df) < MIN_ROWS:
        return {"valid": False, "reason": f"Dataset has fewer than {MIN_ROWS} rows"}

    if df.shape[1] < MIN_COLUMNS:
        return {"valid": False, "reason": f"Dataset has fewer than {MIN_COLUMNS} columns"}

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_column and target_column in df.columns and target_column not in numeric_cols:
        return {"valid": False, "reason": "Target column is not numeric"}

    if len(numeric_cols) < 1:
        return {"valid": False, "reason": "No numerical columns found"}

    return {"valid": True, "reason": "Dataset is valid"}
