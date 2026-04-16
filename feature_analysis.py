from __future__ import annotations

import numpy as np
import pandas as pd

WEAK_CORR_THRESHOLD = 0.15
REDUNDANCY_THRESHOLD = 0.85
MAX_REDUNDANT_PAIRS = 10
LOW_VARIANCE_THRESHOLD = 0.01


def analyze_features(df: pd.DataFrame, target_col: str) -> dict:
    """
    Feature analysis for decision-intelligence evidence.
    Returns deterministic counts based on numeric predictor columns only.
    """
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_col not in numeric_cols:
            return {
                "weak_features": 0,
                "predictor_count": 0,
                "redundant_pairs": [],
                "feature_quality": 0.0,
                "diversity_score": 0.0,
            }

        # Predictor columns = numeric columns excluding target
        predictor_cols = [c for c in numeric_cols if c != target_col]
        predictor_count = len(predictor_cols)

        if predictor_count == 0:
            return {
                "weak_features": 0,
                "predictor_count": 0,
                "redundant_pairs": [],
                "feature_quality": 0.0,
                "diversity_score": 0.0,
            }

        # Correlation of predictors with target
        corr_with_target = df[numeric_cols].corr()[target_col].fillna(0.0)
        weak_count = sum(
            1 for col in predictor_cols if abs(float(corr_with_target[col])) < WEAK_CORR_THRESHOLD
        )

        # Redundant predictor pairs (exclude target from pair matrix)
        corr_matrix = df[predictor_cols].corr().abs().fillna(0.0)
        redundant = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = float(corr_matrix.iloc[i, j])
                if corr_val > REDUNDANCY_THRESHOLD:
                    redundant.append(
                        {
                            "feature_a": corr_matrix.columns[i],
                            "feature_b": corr_matrix.columns[j],
                            "correlation": corr_val,
                        }
                    )
                    if len(redundant) >= MAX_REDUNDANT_PAIRS:
                        break
            if len(redundant) >= MAX_REDUNDANT_PAIRS:
                break

        # Quality score based on low variance among predictors
        low_variance = sum(
            1 for col in predictor_cols if float(df[col].var()) < LOW_VARIANCE_THRESHOLD
        )
        quality_score = (1 - (low_variance / max(predictor_count, 1))) * 100

        # Diversity = numeric predictors / total non-target columns
        total_non_target = max(len(df.columns) - 1, 1)
        diversity = predictor_count / total_non_target

        return {
            "weak_features": int(weak_count),
            "predictor_count": int(predictor_count),
            "redundant_pairs": redundant,
            "feature_quality": float(quality_score),
            "diversity_score": float(diversity),
        }

    except Exception:
        return {
            "weak_features": 0,
            "predictor_count": 0,
            "redundant_pairs": [],
            "feature_quality": 0.0,
            "diversity_score": 0.0,
        }