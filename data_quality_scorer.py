from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

TYPE_DIVERSITY_THRESHOLD = 2
CONSISTENCY_PENALTY = 10.0


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def calculate_data_quality_score(df: pd.DataFrame) -> Dict:
    total_rows = max(len(df), 1)
    total_features = max(df.shape[1], 1)

    completeness = float((df.notna().all(axis=1).sum() / total_rows) * 100)
    uniqueness = float((1 - (df.duplicated().sum() / total_rows)) * 100)

    consistency_issue_columns = 0
    for column in df.columns:
        non_null = df[column].dropna()
        if non_null.empty:
            continue
        python_types = non_null.map(type).nunique()
        if python_types > TYPE_DIVERSITY_THRESHOLD:
            consistency_issue_columns += 1
    consistency = max(
        0.0,
        100.0 - ((consistency_issue_columns / total_features) * 100.0 * (CONSISTENCY_PENALTY / 100.0)),
    )

    numeric_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    plausibility = 100.0
    if not numeric_df.empty:
        outlier_total = 0
        value_total = 0
        for column in numeric_df.columns:
            series = numeric_df[column].dropna()
            if len(series) < 4:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_total += int(((series < lower) | (series > upper)).sum())
            value_total += int(series.shape[0])
        if value_total > 0:
            plausibility = float((1 - (outlier_total / value_total)) * 100)

    feature_richness = float((numeric_df.shape[1] / total_features) * 100)
    overall_score = (
        (0.25 * completeness)
        + (0.20 * uniqueness)
        + (0.20 * consistency)
        + (0.20 * plausibility)
        + (0.15 * feature_richness)
    )
    rounded = round(overall_score, 1)
    grade = _grade(rounded)
    assessment = (
        "Good quality data with minor outlier concerns"
        if grade in {"A", "B"}
        else "Data quality is moderate and needs improvement before reliable modeling"
    )
    return {
        "data_quality": {
            "completeness": round(completeness, 1),
            "uniqueness": round(uniqueness, 1),
            "consistency": round(consistency, 1),
            "plausibility": round(plausibility, 1),
            "feature_richness": round(feature_richness, 1),
            "overall_score": rounded,
        },
        "grade": grade,
        "assessment": assessment,
    }
