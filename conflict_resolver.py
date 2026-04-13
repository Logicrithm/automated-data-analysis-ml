from __future__ import annotations

from typing import Dict, List


def resolve_conflicts(signals: Dict, domain: str, diagnosis: Dict, recommendations: List[Dict] | None) -> Dict:
    if float(signals.get("data_quality_score", 0.0)) < 70:
        return {
            "primary_issue": "data_quality",
            "is_data_problem": True,
            "confidence": 0.95,
            "domain": domain,
        }

    if diagnosis.get("feature_strength") == "weak":
        return {
            "primary_issue": "feature_gap",
            "is_feature_problem": True,
            "confidence": 0.9,
            "domain": domain,
        }

    if diagnosis.get("multicollinearity") in {"high", "critical"}:
        return {
            "primary_issue": "multicollinearity",
            "is_feature_problem": True,
            "confidence": 0.88,
            "domain": domain,
        }

    if diagnosis.get("feature_strength") in {"moderate", "strong"} and diagnosis.get("model_perf") == "critical":
        return {
            "primary_issue": "model_mismatch",
            "is_model_problem": True,
            "confidence": 0.8,
            "domain": domain,
        }

    return {
        "primary_issue": "unknown",
        "confidence": 0.5,
        "domain": domain,
    }
