from __future__ import annotations

from typing import Dict, List


def recommend(
    domain: str = "", diagnosis: Dict | None = None, evidence: Dict | None = None
) -> List[Dict]:
    _ = (domain, diagnosis)  # Domain/diagnosis kept for compatibility with current pipeline API.
    if not evidence:
        return [
            {
                "priority": "LOW",
                "action": "Collect required evidence signals",
                "reason": "Evidence was unavailable, so targeted recommendations could not be generated reliably.",
                "impact": "MEDIUM",
                "effort": "LOW",
                "evidence": {},
            }
        ]

    weak_feature_pct = int(evidence.get("weak_feature_pct", 0))
    strongest_correlation = float(evidence.get("strongest_correlation", 0.0))
    weak_features = int(evidence.get("weak_features", 0))
    redundant_pairs_count = int(evidence.get("redundant_pairs_count", 0))
    max_redundancy_correlation = float(evidence.get("max_redundancy_correlation", 0.0))
    r2_score = float(evidence.get("r2_score", 0.0))
    r2_percentage = float(evidence.get("r2_percentage", 0.0))
    data_quality_score = float(evidence.get("data_quality_score", 0.0))
    missing_percentage = float(evidence.get("missing_percentage", 0.0))

    recommendations: List[Dict] = []

    if weak_feature_pct > 50:
        recommendations.append(
            {
                "priority": "CRITICAL",
                "action": "Add domain-specific features",
                "reason": f"{weak_feature_pct}% of features are weak (correlation < 0.15). Strongest predictor is only {strongest_correlation:.2f}. Insufficient signal in current feature set.",
                "impact": "HIGH",
                "effort": "HIGH",
                "evidence": {
                    "weak_features": weak_features,
                    "strongest_correlation": strongest_correlation,
                },
            }
        )

    if redundant_pairs_count > 2:
        recommendations.append(
            {
                "priority": "HIGH",
                "action": "Remove multicollinear features",
                "reason": f"{redundant_pairs_count} pairs of features show high correlation (max: {max_redundancy_correlation:.2f}). Redundant features distort model coefficients.",
                "impact": "HIGH",
                "effort": "LOW",
                "evidence": {
                    "redundant_pairs": redundant_pairs_count,
                    "max_correlation": max_redundancy_correlation,
                },
            }
        )

    if r2_score < 0.3 and weak_feature_pct <= 50:
        recommendations.append(
            {
                "priority": "HIGH",
                "action": "Try non-linear models (Random Forest, Gradient Boosting)",
                "reason": f"Model R² = {r2_percentage:.1f}% is poor despite reasonable features. Linear relationships may not capture data patterns.",
                "impact": "MEDIUM",
                "effort": "MEDIUM",
                "evidence": {
                    "r2_score": r2_score,
                    "r2_percentage": r2_percentage,
                },
            }
        )

    if data_quality_score < 70:
        recommendations.append(
            {
                "priority": "CRITICAL",
                "action": "Improve data quality - handle missing values",
                "reason": f"Data quality score is only {data_quality_score}/100 with {missing_percentage:.1f}% missing values. Poor data quality undermines all analysis.",
                "impact": "HIGH",
                "effort": "HIGH",
                "evidence": {
                    "quality_score": data_quality_score,
                    "missing_percentage": missing_percentage,
                },
            }
        )

    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))
    return recommendations