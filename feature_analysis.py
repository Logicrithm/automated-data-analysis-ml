from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_features(df: pd.DataFrame, target_col: str) -> dict:
    """
    Simple feature analysis based on signals.
    FIX 2: Limit redundant pairs to 10
    FIX 3: NaN safety with fillna(0)

    Returns:
    {
        'weak_features': count,
        'redundant_pairs': list,
        'feature_quality': score,
        'diversity_score': score,
    }
    """

    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if target_col not in numeric_cols:
            return {"weak_features": 0, "redundant_pairs": [], "feature_quality": 0, "diversity_score": 0}

        # Weak features: correlation < 0.15 with target
        # FIX 3: NaN safety
        target_corr = df[numeric_cols].corr()[target_col].fillna(0)
        weak_count = sum(1 for col in numeric_cols if col != target_col and abs(target_corr[col]) < 0.15)

        # Redundant pairs: correlation > 0.85 between features
        # FIX 2: limit to 10 pairs
        corr_matrix = df[numeric_cols].corr().abs()
        redundant = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.85 and len(redundant) < 10:
                    redundant.append(
                        {
                            "feature_a": corr_matrix.columns[i],
                            "feature_b": corr_matrix.columns[j],
                            "correlation": float(corr_matrix.iloc[i, j]),
                        }
                    )

        # Feature quality: % of features with variance
        low_variance = sum(1 for col in numeric_cols if df[col].var() < 0.01)
        quality_score = (1 - (low_variance / max(len(numeric_cols), 1))) * 100

        # Diversity: ratio of numeric to total features
        diversity = len(numeric_cols) / max(len(df.columns), 1)

        return {
            "weak_features": int(weak_count),
            "redundant_pairs": redundant,
            "feature_quality": float(quality_score),
            "diversity_score": float(diversity),
        }

    except Exception:
        return {"weak_features": 0, "redundant_pairs": [], "feature_quality": 0, "diversity_score": 0}
