from __future__ import annotations

from typing import Dict

FINDING_BASE = 0.55
FINDING_DATA_WEIGHT = 0.25
FINDING_FEATURE_WEIGHT = 0.20
RELIABILITY_MODEL_WEIGHT = 0.60
RELIABILITY_DATA_WEIGHT = 0.40
OVERALL_FINDING_WEIGHT = 0.30
OVERALL_RELIABILITY_WEIGHT = 0.30
OVERALL_DOMAIN_WEIGHT = 0.20
OVERALL_ACTIONABLE_WEIGHT = 0.20


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def calculate_weighted_confidence(
    data_quality: float,
    model_performance: float,
    feature_relevance: float,
    domain_confidence: float,
    actionability: float,
) -> Dict[str, float]:
    finding = _clamp(FINDING_BASE + (FINDING_DATA_WEIGHT * data_quality) + (FINDING_FEATURE_WEIGHT * feature_relevance))
    reliability = _clamp((RELIABILITY_MODEL_WEIGHT * model_performance) + (RELIABILITY_DATA_WEIGHT * data_quality))
    domain = _clamp(domain_confidence)
    actionable = _clamp(actionability)
    overall = _clamp(
        (OVERALL_FINDING_WEIGHT * finding)
        + (OVERALL_RELIABILITY_WEIGHT * reliability)
        + (OVERALL_DOMAIN_WEIGHT * domain)
        + (OVERALL_ACTIONABLE_WEIGHT * actionable)
    )
    return {
        "finding": round(finding, 2),
        "reliability": round(reliability, 2),
        "domain": round(domain, 2),
        "actionable": round(actionable, 2),
        "overall": round(overall, 2),
    }
