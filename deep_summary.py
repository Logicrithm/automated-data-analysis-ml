from __future__ import annotations


def generate_deep_summary(evidence: dict, diagnosis: dict | None = None) -> dict:
    if not evidence:
        return {
            "executive_insight": "Analysis complete.",
            "key_finding": "Review results and recommendations.",
            "action_priority": "Follow recommended actions.",
        }

    if evidence["poor_data_quality"]:
        executive = f"Data quality is critical (score: {evidence['data_quality_score']}/100). Missing values at {evidence['missing_percentage']:.1f}% undermine all conclusions. Address data issues first."
    elif evidence["weak_feature_pct"] > 60:
        executive = f"{evidence['weak_feature_pct']}% of features show weak correlation (<0.15). Strongest predictor is {evidence['strongest_correlation']:.2f}. Dataset lacks sufficient predictive signal."
    elif evidence["poor_model_fit"]:
        executive = f"Model R² = {evidence['r2_percentage']:.1f}% indicates poor fit. Non-linear relationships may be present."
    else:
        executive = f"Model explains {evidence['r2_percentage']:.1f}% of variance. Feature set is reasonable with {evidence['weak_feature_pct']}% weak features."

    if evidence["redundant_pairs_count"] > 2:
        key = f"{evidence['redundant_pairs_count']} feature pairs are redundant (max correlation: {evidence['max_redundancy_correlation']:.2f}). Remove correlated features."
    elif evidence["weak_feature_pct"] > 60:
        key = "Insufficient predictive signal. Current features cannot drive reliable predictions."
    elif evidence["poor_model_fit"]:
        key = f"Poor model fit (R² = {evidence['r2_percentage']:.1f}%). Try non-linear models or add interaction features."
    else:
        key = f"Reasonable model fit with {evidence['weak_feature_pct']}% weak features and {evidence['redundant_pairs_count']} redundant pairs."

    if evidence["poor_data_quality"]:
        action = f"CRITICAL: Fix data quality (currently {evidence['data_quality_score']}/100). Rerun analysis after cleaning."
    elif evidence["weak_feature_pct"] > 70:
        action = f"CRITICAL: {evidence['weak_feature_pct']}% of features are weak. Collect domain-specific features."
    elif evidence["redundant_pairs_count"] > 2:
        action = f"HIGH: Remove {evidence['redundant_pairs_count']} redundant feature pairs. Then retry modeling."
    else:
        action = "MEDIUM: Try advanced models and feature engineering for marginal improvements."

    return {
        "executive_insight": executive,
        "key_finding": key,
        "action_priority": action,
        "evidence_summary": {
            "model_performance": f"R² = {evidence['r2_percentage']:.1f}%",
            "feature_quality": f"{evidence['weak_feature_pct']}% weak",
            "redundancy": f"{evidence['redundant_pairs_count']} correlated pairs",
            "data_quality": f"{evidence['data_quality_score']}/100",
        },
    }
