from __future__ import annotations

from typing import Dict, List


def _action(priority: str, action: str, details: str, impact: str, effort: str) -> Dict:
    return {
        "priority": priority,
        "action": action,
        "details": details,
        "impact": impact,
        "effort": effort,
    }


def generate_rule_chained_recommendations(
    r2: float,
    multicollinearity: Dict,
    feature_count: int,
    vif_data: List[Dict],
    residuals_have_pattern: bool = False,
) -> Dict:
    high_vif_pairs = multicollinearity.get("high_vif_pairs") or []
    high_vif = [item for item in (vif_data or []) if float(item.get("vif", 0.0)) > 10]
    max_vif = max([float(item.get("vif", 0.0)) for item in (vif_data or [])], default=0.0)
    multicollinearity_high = bool(high_vif_pairs or high_vif)

    recommendations: List[Dict] = []
    if r2 < 0.3 and multicollinearity_high and max_vif > 10:
        pair_text = ""
        if high_vif_pairs:
            top = high_vif_pairs[0]
            pair_text = (
                f"{top['feature_a']} and {top['feature_b']} correlation is {top['correlation']:+.2f}."
            )
        recommendations.extend(
            [
                _action(
                    "CRITICAL",
                    "Remove redundant features",
                    pair_text or "High-VIF features are introducing redundant signal.",
                    "HIGH",
                    "LOW",
                ),
                _action(
                    "HIGH",
                    "Apply dimensionality reduction",
                    "Use PCA or feature engineering to reduce multicollinearity.",
                    "HIGH",
                    "MEDIUM",
                ),
                _action(
                    "MEDIUM",
                    "Then retry with non-linear model",
                    "Random Forest or Gradient Boosting can handle complex interactions better.",
                    "MEDIUM",
                    "MEDIUM",
                ),
            ]
        )
    elif r2 < 0.3 and feature_count < 5:
        recommendations.extend(
            [
                _action("CRITICAL", "Collect additional features", "Current feature set is too small.", "HIGH", "MEDIUM"),
                _action("HIGH", "Perform domain research", "Gather known predictive signals from the business domain.", "HIGH", "MEDIUM"),
                _action("MEDIUM", "Add feature interactions", "Introduce interaction and polynomial terms.", "MEDIUM", "LOW"),
            ]
        )
    elif r2 < 0.3 and feature_count > 10:
        recommendations.extend(
            [
                _action("CRITICAL", "Try non-linear models", "Random Forest/Gradient Boosting likely fit better.", "HIGH", "LOW"),
                _action("HIGH", "Apply feature selection", "Reduce noise and keep high-signal predictors.", "HIGH", "MEDIUM"),
                _action("MEDIUM", "Use ensemble methods", "Blend models for robustness.", "MEDIUM", "MEDIUM"),
            ]
        )
    elif r2 > 0.7 and residuals_have_pattern:
        recommendations.extend(
            [
                _action("HIGH", "Add interaction features", "Residual pattern implies missing interaction effects.", "MEDIUM", "LOW"),
                _action("MEDIUM", "Check non-linear patterns", "Inspect residual plots and non-linear transforms.", "MEDIUM", "LOW"),
                _action("LOW", "Verify assumptions", "Confirm remaining assumptions for production deployment.", "LOW", "LOW"),
            ]
        )
    else:
        recommendations.extend(
            [
                _action("HIGH", "Continue feature engineering", "Add domain-specific predictors.", "MEDIUM", "MEDIUM"),
                _action("MEDIUM", "Benchmark non-linear models", "Compare with tree-based methods.", "MEDIUM", "LOW"),
            ]
        )

    return {"recommendations": recommendations}
