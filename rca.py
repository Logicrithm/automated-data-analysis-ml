from __future__ import annotations

from typing import Dict


def diagnose(
    signals: Dict | None = None, ml_results: Dict | None = None, evidence: Dict | None = None
) -> Dict:
    _ = (signals, ml_results)  # Retained for compatibility and potential fallback diagnostics.
    if not evidence:
        return {
            "model_perf": "unknown",
            "feature_strength": "unknown",
            "multicollinearity": "unknown",
            "data_quality": "unknown",
        }

    r2_score = float(evidence.get("r2_score", 0.0))
    weak_feature_pct = int(evidence.get("weak_feature_pct", 0))
    redundant_pairs_count = int(evidence.get("redundant_pairs_count", 0))
    data_quality_score = float(evidence.get("data_quality_score", 0.0))

    if r2_score < 0.1:
        model_perf = "critical"
    elif r2_score < 0.3:
        model_perf = "weak"
    elif r2_score < 0.5:
        model_perf = "moderate"
    else:
        model_perf = "good"

    if weak_feature_pct > 70:
        feature_strength = "critical"
    elif weak_feature_pct > 50:
        feature_strength = "weak"
    elif weak_feature_pct > 30:
        feature_strength = "moderate"
    else:
        feature_strength = "strong"

    if redundant_pairs_count > 5:
        multicollinearity = "critical"
    elif redundant_pairs_count > 2:
        multicollinearity = "high"
    elif redundant_pairs_count > 0:
        multicollinearity = "moderate"
    else:
        multicollinearity = "low"

    if data_quality_score < 60:
        data_quality = "poor"
    elif data_quality_score < 75:
        data_quality = "fair"
    elif data_quality_score < 85:
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
