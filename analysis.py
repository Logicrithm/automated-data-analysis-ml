from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


TARGET_CANDIDATES = ("price", "target", "label", "y")


@dataclass
class ModelInterpretation:
    severity: str
    message: str


def interpret_r2_score(r2_value: float, target_label: str = "target") -> ModelInterpretation:
    if r2_value < 0.2:
        return ModelInterpretation(
            severity="critical",
            message=f"🚨 Model is unreliable. CRITICAL: {target_label} prediction model failed with R² = {r2_value:.1%}, indicating current features are insufficient.",
        )
    if r2_value < 0.5:
        return ModelInterpretation(
            severity="warning",
            message=f"⚠️ Model performance is weak (R² = {r2_value:.1%}).",
        )
    return ModelInterpretation(
        severity="ok",
        message=f"✓ Model shows reasonable predictive power (R² = {r2_value:.1%}).",
    )


def choose_target_column(df: pd.DataFrame, requested_target: Optional[str] = None) -> Optional[str]:
    if requested_target and requested_target in df.columns:
        return requested_target

    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lower_to_original and pd.api.types.is_numeric_dtype(df[lower_to_original[candidate]]):
            return lower_to_original[candidate]

    numeric_columns = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 10
    ]
    if not numeric_columns:
        return None

    for preferred in numeric_columns:
        if "price" in preferred.lower():
            return preferred

    return max(numeric_columns, key=lambda col: df[col].var(skipna=True))


def run_regression_analysis(df: pd.DataFrame, target_column: str) -> Dict:
    working_df = df.dropna(subset=[target_column]).copy()
    if target_column not in working_df.columns:
        return {"problem_type": "none", "target_column": None}

    feature_df = working_df.drop(columns=[target_column])
    feature_df = pd.get_dummies(feature_df, drop_first=True)
    feature_df = feature_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)

    if feature_df.empty or len(working_df) < 20:
        interpretation = interpret_r2_score(0.0, target_column)
        return {
            "problem_type": "regression",
            "target_column": target_column,
            "r2_score": 0.0,
            "severity": interpretation.severity,
            "interpretation": interpretation.message,
        }

    y = working_df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(feature_df, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r2_value = float(r2_score(y_test, predictions))

    strongest_idx = int(np.argmax(np.abs(model.coef_))) if len(model.coef_) else 0
    strongest_feature = feature_df.columns[strongest_idx] if len(feature_df.columns) else "N/A"

    interpretation = interpret_r2_score(r2_value, target_column)
    return {
        "problem_type": "regression",
        "target_column": target_column,
        "r2_score": r2_value,
        "severity": interpretation.severity,
        "interpretation": interpretation.message,
        "strongest_feature": strongest_feature,
    }
