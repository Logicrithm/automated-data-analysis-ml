from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


TARGET_CANDIDATES = ("price", "target", "label", "y")
RANDOM_STATE = 42


@dataclass
class ModelInterpretation:
    severity: str
    message: str


def interpret_r2_score(r2_value: float, target_label: str = "target") -> ModelInterpretation:
    unexplained_variance = max(0.0, 1.0 - r2_value)
    if r2_value < 0.2:
        return ModelInterpretation(
            severity="critical",
            message=(
                f"🚨 Model is unreliable. CRITICAL: {target_label} prediction model failed with R² = {r2_value:.1%}. "
                f"{unexplained_variance:.1%} of variance remains unexplained, which suggests missing features, non-linear relationships, or multicollinearity."
            ),
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
    np.random.seed(RANDOM_STATE)
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
    x_train, x_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r2_value = float(r2_score(y_test, predictions))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feature_df)
    scaled_model = LinearRegression()
    scaled_model.fit(x_scaled, y)

    scaled_importance = np.abs(scaled_model.coef_)
    sorted_scaled_idx = np.argsort(scaled_importance)[::-1]
    strongest_feature = feature_df.columns[int(sorted_scaled_idx[0])] if len(feature_df.columns) else "N/A"
    standardized_importance = [
        {"feature": feature_df.columns[int(idx)], "importance": float(scaled_importance[int(idx)])}
        for idx in sorted_scaled_idx
    ]

    perm_result = permutation_importance(
        model, x_test, y_test, n_repeats=10, random_state=RANDOM_STATE
    )
    perm_idx = np.argsort(np.abs(perm_result.importances_mean))[::-1]
    permutation_scores = [
        {
            "feature": feature_df.columns[int(idx)],
            "importance": float(perm_result.importances_mean[int(idx)]),
        }
        for idx in perm_idx
    ]

    vif_scores = []
    for index, feature_name in enumerate(feature_df.columns):
        try:
            score = float(variance_inflation_factor(x_scaled, index))
        except (ValueError, ZeroDivisionError):
            score = float("inf")
        vif_scores.append({"feature": feature_name, "vif": score})
    vif_scores = sorted(vif_scores, key=lambda item: item["vif"], reverse=True)
    max_vif = vif_scores[0]["vif"] if vif_scores else 0.0
    multicollinearity_warning = (
        "⚠️ High multicollinearity detected (VIF > 10). Feature effects may appear unstable."
        if max_vif > 10
        else ""
    )

    interpretation = interpret_r2_score(r2_value, target_column)
    return {
        "problem_type": "regression",
        "target_column": target_column,
        "r2_score": r2_value,
        "severity": interpretation.severity,
        "interpretation": interpretation.message,
        "strongest_feature": strongest_feature,
        "standardized_importance": standardized_importance,
        "permutation_importance": permutation_scores,
        "vif": vif_scores,
        "multicollinearity_warning": multicollinearity_warning,
    }
