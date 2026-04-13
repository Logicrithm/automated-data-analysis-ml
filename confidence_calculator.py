from __future__ import annotations

from typing import Dict


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def calculate_weighted_confidence(
    data_quality: float,
    model_performance: float,
    feature_relevance: float,
    domain_confidence: float,
    actionability: float,
) -> Dict[str, float]:
    finding = _clamp(0.55 + (0.25 * data_quality) + (0.20 * feature_relevance))
    reliability = _clamp((0.60 * model_performance) + (0.40 * data_quality))
    domain = _clamp(domain_confidence)
    actionable = _clamp(actionability)
    overall = _clamp(
        (0.30 * finding)
        + (0.30 * reliability)
        + (0.20 * domain)
        + (0.20 * actionable)
    )
    return {
        "finding": round(finding, 2),
        "reliability": round(reliability, 2),
        "domain": round(domain, 2),
        "actionable": round(actionable, 2),
        "overall": round(overall, 2),
    }
