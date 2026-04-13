from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from data_quality_scorer import calculate_data_quality_score

MAX_UNIQUE_FOR_CLASSIFICATION = 20
MAX_UNIQUE_RATIO_FOR_CLASSIFICATION = 0.05


def _target_type(series: pd.Series) -> str:
    if series.empty:
        return "unknown"
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    unique_count = int(series.nunique(dropna=True))
    unique_ratio = unique_count / max(len(series), 1)
    if unique_count <= MAX_UNIQUE_FOR_CLASSIFICATION or unique_ratio <= MAX_UNIQUE_RATIO_FOR_CLASSIFICATION:
        return "classification"
    return "regression"


def extract_signals(df: pd.DataFrame, target_col: str | None) -> Dict:
    n_rows, n_cols = df.shape
    n_features = max(n_cols - (1 if target_col in df.columns else 0), 0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_feature_cols = [col for col in numeric_cols if col != target_col]
    numeric_ratio = float(len(numeric_feature_cols) / max(n_features, 1))

    target_series = df[target_col] if target_col and target_col in df.columns else pd.Series(dtype=float)
    target_type = _target_type(target_series)

    avg_correlation = 0.0
    if target_col and target_col in numeric_cols and numeric_feature_cols:
        corr_with_target = df[numeric_feature_cols + [target_col]].corr(numeric_only=True)[target_col].drop(labels=[target_col])
        if not corr_with_target.empty:
            avg_correlation = float(corr_with_target.abs().mean())

    max_correlation = 0.0
    if len(numeric_feature_cols) > 1:
        corr_matrix = df[numeric_feature_cols].corr(numeric_only=True).abs()
        matrix = corr_matrix.to_numpy(copy=True)
        np.fill_diagonal(matrix, 0.0)
        max_correlation = float(matrix.max())

    quality_result = calculate_data_quality_score(df)
    data_quality_score = float((quality_result.get("data_quality") or {}).get("overall_score", 0.0))

    return {
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "numeric_ratio": round(numeric_ratio, 3),
        "target_type": target_type,
        "avg_correlation": round(avg_correlation, 3),
        "max_correlation": round(max_correlation, 3),
        "data_quality_score": round(data_quality_score, 1),
    }
