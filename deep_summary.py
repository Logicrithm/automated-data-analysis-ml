from __future__ import annotations


def generate_deep_summary(evidence: dict, diagnosis: dict | None = None) -> dict:
    _ = diagnosis  # Kept for backward-compatible call sites.
    if not evidence:
        return {
            "executive_insight": "Analysis complete.",
            "key_finding": "Review results and recommendations.",
            "action_priority": "Follow recommended actions.",
        }

    poor_data_quality = bool(evidence.get("poor_data_quality", False))
    weak_feature_pct = int(evidence.get("weak_feature_pct", 0))
    strongest_correlation = float(evidence.get("strongest_correlation", 0.0))
    poor_model_fit = bool(evidence.get("poor_model_fit", False))
    r2_percentage = float(evidence.get("r2_percentage", 0.0))
    redundant_pairs_count = int(evidence.get("redundant_pairs_count", 0))
    max_redundancy_correlation = float(evidence.get("max_redundancy_correlation", 0.0))
    data_quality_score = float(evidence.get("data_quality_score", 0.0))
    missing_percentage = float(evidence.get("missing_percentage", 0.0))

    if poor_data_quality:
        executive = f"Data quality is critical (score: {data_quality_score}/100). Missing values at {missing_percentage:.1f}% undermine all conclusions. Address data issues first."
    elif weak_feature_pct > 60:
        executive = f"{weak_feature_pct}% of features show weak correlation (<0.15). Strongest predictor is {strongest_correlation:.2f}. Dataset lacks sufficient predictive signal."
    elif poor_model_fit:
        executive = f"Model R² = {r2_percentage:.1f}% indicates poor fit. Non-linear relationships may be present."
    else:
        executive = f"Model explains {r2_percentage:.1f}% of variance. Feature set is reasonable with {weak_feature_pct}% weak features."

    if redundant_pairs_count > 2:
        key = f"{redundant_pairs_count} feature pairs are redundant (max correlation: {max_redundancy_correlation:.2f}). Remove correlated features."
    elif weak_feature_pct > 60:
        key = "Insufficient predictive signal. Current features cannot drive reliable predictions."
    elif poor_model_fit:
        key = f"Poor model fit (R² = {r2_percentage:.1f}%). Try non-linear models or add interaction features."
    else:
        key = f"Reasonable model fit with {weak_feature_pct}% weak features and {redundant_pairs_count} redundant pairs."

    if poor_data_quality:
        action = f"CRITICAL: Fix data quality (currently {data_quality_score}/100). Rerun analysis after cleaning."
    elif weak_feature_pct > 70:
        action = f"CRITICAL: {weak_feature_pct}% of features are weak. Collect domain-specific features."
    elif redundant_pairs_count > 2:
        action = f"HIGH: Remove {redundant_pairs_count} redundant feature pairs. Then retry modeling."
    else:
        action = "MEDIUM: Try advanced models and feature engineering for marginal improvements."

    return {
        "executive_insight": executive,
        "key_finding": key,
        "action_priority": action,
        "evidence_summary": {
            "model_performance": f"R² = {r2_percentage:.1f}%",
            "feature_quality": f"{weak_feature_pct}% weak",
            "redundancy": f"{redundant_pairs_count} correlated pairs",
            "data_quality": f"{data_quality_score}/100",
        },
    }
