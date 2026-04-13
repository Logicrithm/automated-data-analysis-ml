from __future__ import annotations

from typing import Dict

R2_THRESHOLD_LOW = 0.2
R2_THRESHOLD_HIGH = 0.7
R2_THRESHOLD_MID = 0.4
DOMAIN_SCORE_HIGH = 0.6
DOMAIN_SCORE_DEFAULT = 0.45
FEATURE_SCORE_WEAK = 0.3
FEATURE_SCORE_MODERATE = 0.65
FEATURE_SCORE_STRONG_BONUS = 0.2


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
    if r2_value >= R2_THRESHOLD_HIGH:
        model_score += 0.35
    elif r2_value <= R2_THRESHOLD_LOW:
        model_score -= 0.2
    elif r2_value <= R2_THRESHOLD_MID:
        model_score += 0.05

    strongest_importance = 0.0
    standardized = ml_results.get("standardized_importance") or []
    if standardized:
        strongest_importance = float(standardized[0].get("importance", 0.0))
    feature_relevance_score = FEATURE_SCORE_WEAK if strongest_importance <= 0 else FEATURE_SCORE_MODERATE
    if strongest_importance > 0.5:
        feature_relevance_score += FEATURE_SCORE_STRONG_BONUS

    target_name = str(ml_results.get("target_column", "")).lower()
    domain_score = (
        DOMAIN_SCORE_HIGH if any(token in target_name for token in ("price", "value", "cost")) else DOMAIN_SCORE_DEFAULT
    )

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
