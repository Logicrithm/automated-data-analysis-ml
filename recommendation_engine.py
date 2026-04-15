from __future__ import annotations

from typing import Dict, List, Optional


def _action(priority: str, action: str, reason: str, evidence: Dict, impact: str = "HIGH", effort: str = "MEDIUM") -> Dict:
    return {
        "priority": priority,
        "action": action,
        "reason": reason,
        "impact": impact,
        "effort": effort,
        "evidence": evidence,
    }


def generate_recommendations_v2(
    decision: Optional[Dict],
    evidence: Optional[Dict],
    causal_layer: Optional[Dict],
    context: Optional[Dict] = None,
) -> List[Dict]:
    decision = decision or {}
    evidence = evidence or {}
    causal_layer = causal_layer or {}
    context = context or {}

    decision_name = str(decision.get("decision", "UNKNOWN"))
    severity = str(decision.get("severity", "MEDIUM"))
    root_cause = str(causal_layer.get("root_cause", "Primary issue identified from evidence."))
    dominant_signal = str(decision.get("dominant_signal", "evidence trigger not available"))

    common_evidence = {
        "dominant_signal": dominant_signal,
        "secondary_signals": decision.get("secondary_signals", []),
        "r2_percentage": evidence.get("r2_percentage", 0.0),
        "weak_feature_pct": evidence.get("weak_feature_pct", 0),
    }

    if decision_name == "DATA_ISSUE":
        return [
            _action(
                severity,
                "Fix missing values and schema consistency before retraining",
                f"{root_cause} Trigger: {dominant_signal}",
                {**common_evidence, "missing_percentage": evidence.get("missing_percentage", 0.0)},
                impact="HIGH",
                effort="HIGH",
            ),
            _action(
                "HIGH",
                "Run data validation checks (nulls, duplicates, outliers)",
                "Data defects must be controlled before any model-level change.",
                common_evidence,
                impact="HIGH",
                effort="MEDIUM",
            ),
        ]

    if decision_name == "MULTICOLLINEARITY":
        return [
            _action(
                severity,
                "Remove or merge highly correlated feature pairs",
                f"{root_cause} Trigger: {dominant_signal}",
                {**common_evidence, "redundant_pairs_count": evidence.get("redundant_pairs_count", 0)},
                impact="HIGH",
                effort="LOW",
            ),
            _action(
                "HIGH",
                "Apply PCA or regularization to stabilize feature effects",
                "Dimensionality reduction reduces redundant signal overlap.",
                common_evidence,
                impact="HIGH",
                effort="MEDIUM",
            ),
        ]

    if decision_name == "NON_LINEARITY":
        return [
            _action(
                severity,
                "Benchmark tree-based models (Random Forest / Gradient Boosting)",
                f"{root_cause} Trigger: {dominant_signal}",
                common_evidence,
                impact="HIGH",
                effort="MEDIUM",
            ),
            _action(
                "MEDIUM",
                "Add interaction and non-linear transformed features",
                "Feature transforms can expose structure missed by linear assumptions.",
                common_evidence,
                impact="MEDIUM",
                effort="MEDIUM",
            ),
        ]

    if decision_name == "FEATURE_GAP":
        return [
            _action(
                severity,
                "Collect additional domain variables with direct target relevance",
                f"{root_cause} Trigger: {dominant_signal}",
                {**common_evidence, "total_features": evidence.get("total_features", 0)},
                impact="HIGH",
                effort="HIGH",
            ),
            _action(
                "HIGH",
                "Prioritize data acquisition plan with subject-matter experts",
                "Missing explanatory variables limit modeling gains from algorithm tuning.",
                common_evidence,
                impact="HIGH",
                effort="HIGH",
            ),
        ]

    return [
        _action(
            severity,
            "Perform targeted feature engineering on low-signal predictors",
            f"{root_cause} Trigger: {dominant_signal}",
            common_evidence,
            impact="HIGH",
            effort="MEDIUM",
        ),
        _action(
            "MEDIUM",
            "Re-assess domain feature coverage and interactions",
            "Weak predictive signal indicates current features are not sufficiently informative.",
            common_evidence,
            impact="MEDIUM",
            effort="MEDIUM",
        ),
    ]


def generate_rule_chained_recommendations(
    r2: float,
    multicollinearity: Dict,
    feature_count: int,
    vif_data: List[Dict],
    residuals_have_pattern: bool = False,
) -> Dict:
    """Backward-compatible wrapper for legacy call sites."""
    recommendations = generate_recommendations_v2(
        {"decision": "WEAK_FEATURES", "severity": "MEDIUM", "dominant_signal": "legacy_fallback", "secondary_signals": []},
        {},
        {"root_cause": "Legacy recommendation flow invoked."},
        {},
    )
    return {"recommendations": recommendations}
