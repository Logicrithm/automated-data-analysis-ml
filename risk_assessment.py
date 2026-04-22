from __future__ import annotations

from typing import Any, Dict, List


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _impact_level(severity: str) -> int:
    mapping = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    return mapping.get(str(severity).upper(), 2)


def _likelihood_from_value(value: float, thresholds: List[float]) -> int:
    if value >= thresholds[2]:
        return 4
    if value >= thresholds[1]:
        return 3
    if value >= thresholds[0]:
        return 2
    return 1


def generate_risk_assessment(results: Dict[str, Any]) -> Dict[str, Any]:
    """Build a 4x4 likelihood-vs-impact matrix and categorized risk items."""
    evidence = results.get("evidence", {}) or {}
    diagnosis = results.get("diagnosis", {}) or {}

    severity = str(diagnosis.get("severity", "MEDIUM")).upper()
    impact_base = _impact_level(severity)

    missing_percentage = _to_float(evidence.get("missing_percentage"), 0.0)
    weak_feature_pct = _to_float(evidence.get("weak_feature_pct"), 0.0)
    r2_score = _to_float(evidence.get("r2_score"), 0.0)
    redundancy = _to_float(evidence.get("redundant_pairs_count"), 0.0)

    model_risk_likelihood = _likelihood_from_value(max(0.0, 1.0 - r2_score), [0.25, 0.5, 0.75])
    data_risk_likelihood = _likelihood_from_value(missing_percentage, [2.0, 8.0, 15.0])
    feature_risk_likelihood = _likelihood_from_value(weak_feature_pct, [30.0, 60.0, 80.0])
    redundancy_risk_likelihood = _likelihood_from_value(redundancy, [1.0, 3.0, 6.0])

    risks = [
        {
            "name": "Model Reliability Risk",
            "likelihood": model_risk_likelihood,
            "impact": max(impact_base, 2),
            "description": f"Primary risk: model is not predictive (R² = {r2_score:.2f}), leading to unreliable outputs.",
        },
        {
            "name": "Data Quality Risk",
            "likelihood": data_risk_likelihood,
            "impact": max(impact_base - 1, 1),
            "description": f"Missing data at {missing_percentage:.1f}% can degrade consistency.",
        },
        {
            "name": "Feature Signal Risk",
            "likelihood": feature_risk_likelihood,
            "impact": impact_base,
            "description": f"Weak features ({weak_feature_pct:.0f}%) can undermine prediction quality.",
        },
        {
            "name": "Redundancy Risk",
            "likelihood": redundancy_risk_likelihood,
            "impact": max(impact_base - 1, 1),
            "description": f"Redundant feature pairs ({int(redundancy)}) may cause unstable attribution.",
        },
    ]

    matrix = [[0 for _ in range(4)] for _ in range(4)]
    for risk in risks:
        matrix[risk["impact"] - 1][risk["likelihood"] - 1] += 1

    highest = max(risks, key=lambda x: (x["impact"] * x["likelihood"], x["impact"], x["likelihood"]))
    overall_score = max(item["impact"] * item["likelihood"] for item in risks)

    return {
        "matrix": matrix,
        "risks": risks,
        "overall_risk_score": overall_score,
        "overall_risk_band": "CRITICAL" if overall_score >= 12 else "HIGH" if overall_score >= 8 else "MEDIUM" if overall_score >= 4 else "LOW",
        "highest_risk": highest,
    }
