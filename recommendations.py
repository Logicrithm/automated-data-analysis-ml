from __future__ import annotations

from typing import Dict, List


def recommend(domain: str, diagnosis: Dict, evidence: Dict | None = None) -> List[Dict]:
    if not evidence:
        return []

    recommendations: List[Dict] = []

    if evidence["weak_feature_pct"] > 50:
        recommendations.append(
            {
                "priority": "CRITICAL",
                "action": "Add domain-specific features",
                "reason": f"{evidence['weak_feature_pct']}% of features are weak (correlation < 0.15). Strongest predictor is only {evidence['strongest_correlation']:.2f}. Insufficient signal in current feature set.",
                "impact": "HIGH",
                "effort": "HIGH",
                "evidence": {
                    "weak_features": evidence["weak_features"],
                    "strongest_correlation": evidence["strongest_correlation"],
                },
            }
        )

    if evidence["redundant_pairs_count"] > 2:
        recommendations.append(
            {
                "priority": "HIGH",
                "action": "Remove multicollinear features",
                "reason": f"{evidence['redundant_pairs_count']} pairs of features show high correlation (max: {evidence['max_redundancy_correlation']:.2f}). Redundant features distort model coefficients.",
                "impact": "HIGH",
                "effort": "LOW",
                "evidence": {
                    "redundant_pairs": evidence["redundant_pairs_count"],
                    "max_correlation": evidence["max_redundancy_correlation"],
                },
            }
        )

    if evidence["r2_score"] < 0.3 and evidence["weak_feature_pct"] <= 50:
        recommendations.append(
            {
                "priority": "HIGH",
                "action": "Try non-linear models (Random Forest, Gradient Boosting)",
                "reason": f"Model R² = {evidence['r2_percentage']:.1f}% is poor despite reasonable features. Linear relationships may not capture data patterns.",
                "impact": "MEDIUM",
                "effort": "MEDIUM",
                "evidence": {
                    "r2_score": evidence["r2_score"],
                    "r2_percentage": evidence["r2_percentage"],
                },
            }
        )

    if evidence["data_quality_score"] < 70:
        recommendations.append(
            {
                "priority": "CRITICAL",
                "action": "Improve data quality - handle missing values",
                "reason": f"Data quality score is only {evidence['data_quality_score']}/100 with {evidence['missing_percentage']:.1f}% missing values. Poor data quality undermines all analysis.",
                "impact": "HIGH",
                "effort": "HIGH",
                "evidence": {
                    "quality_score": evidence["data_quality_score"],
                    "missing_percentage": evidence["missing_percentage"],
                },
            }
        )

    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))
    return recommendations
