from __future__ import annotations

from typing import Dict


def _clamp_level(value: float) -> str:
    if value >= 0.75:
        return "HIGH"
    if value >= 0.45:
        return "MEDIUM"
    return "LOW"


def calculate_confidence_levels(
    results: Dict,
    quality_summary: Dict,
    multicollinearity_summary: Dict,
) -> Dict[str, str]:
    ml_results = results.get("ml_results") or {}
    r2_value = float(ml_results.get("r2_score", 0.0))
    missing = int(quality_summary.get("total_missing", 0))
    duplicates = int(quality_summary.get("duplicate_rows", 0))

    data_quality_score = 1.0
    if missing > 0:
        data_quality_score -= 0.4
    if duplicates > 0:
        data_quality_score -= 0.3

    model_score = 0.5
    if r2_value <= 0.2 or r2_value >= 0.7:
        model_score += 0.35
    elif r2_value <= 0.4:
        model_score += 0.15

    strongest_importance = 0.0
    standardized = ml_results.get("standardized_importance") or []
    if standardized:
        strongest_importance = float(standardized[0].get("importance", 0.0))
    feature_relevance_score = 0.3 if strongest_importance <= 0 else 0.65
    if strongest_importance > 0.5:
        feature_relevance_score += 0.2

    target_name = str(ml_results.get("target_column", "")).lower()
    domain_score = 0.6 if any(token in target_name for token in ("price", "value", "cost")) else 0.45

    multicollinearity_score = 0.5
    if multicollinearity_summary.get("high_vif_pairs"):
        multicollinearity_score = 0.85
    elif multicollinearity_summary.get("high_vif_features"):
        multicollinearity_score = 0.7

    return {
        "MODEL_PERFORMANCE": _clamp_level(model_score),
        "MULTICOLLINEARITY": _clamp_level(multicollinearity_score),
        "FEATURES": _clamp_level((feature_relevance_score + domain_score) / 2),
        "DATA_QUALITY": _clamp_level(data_quality_score),
        "RECOMMENDATIONS": _clamp_level((model_score + domain_score) / 2),
    }
