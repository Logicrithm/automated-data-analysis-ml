from __future__ import annotations

from typing import Dict


def diagnose(signals: Dict, ml_results: Dict, evidence: Dict | None = None) -> Dict:
    if not evidence:
        return {
            "model_perf": "unknown",
            "feature_strength": "unknown",
            "multicollinearity": "unknown",
            "data_quality": "unknown",
        }

    if evidence["r2_score"] < 0.1:
        model_perf = "critical"
    elif evidence["r2_score"] < 0.3:
        model_perf = "weak"
    elif evidence["r2_score"] < 0.5:
        model_perf = "moderate"
    else:
        model_perf = "good"

    if evidence["weak_feature_pct"] > 70:
        feature_strength = "critical"
    elif evidence["weak_feature_pct"] > 50:
        feature_strength = "weak"
    elif evidence["weak_feature_pct"] > 30:
        feature_strength = "moderate"
    else:
        feature_strength = "strong"

    if evidence["redundant_pairs_count"] > 5:
        multicollinearity = "critical"
    elif evidence["redundant_pairs_count"] > 2:
        multicollinearity = "high"
    elif evidence["redundant_pairs_count"] > 0:
        multicollinearity = "moderate"
    else:
        multicollinearity = "low"

    if evidence["data_quality_score"] < 60:
        data_quality = "poor"
    elif evidence["data_quality_score"] < 75:
        data_quality = "fair"
    elif evidence["data_quality_score"] < 85:
        data_quality = "good"
    else:
        data_quality = "excellent"

    return {
        "model_perf": model_perf,
        "feature_strength": feature_strength,
        "multicollinearity": multicollinearity,
        "data_quality": data_quality,
        "evidence_used": True,
    }
