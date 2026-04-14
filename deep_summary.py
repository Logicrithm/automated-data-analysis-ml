from __future__ import annotations


def generate_deep_summary(
    signals: dict, diagnosis: dict, verdict: dict, features: dict, interpretation: dict
) -> dict:
    """
    Simple narrative summary based on signals.
    FIX 1: Safe n_features access with fallback to 10

    Returns:
    {
        'executive_insight': str,
        'key_finding': str,
        'action_priority': str,
    }
    """

    try:
        primary_issue = verdict.get("primary_issue", "unknown")
        data_quality = diagnosis.get("data_quality", "unknown")
        feature_strength = diagnosis.get("feature_strength", "unknown")

        # EXECUTIVE INSIGHT
        if data_quality in ["poor", "fair"]:
            executive = "Data quality is the primary bottleneck. Improve data collection and cleaning first."
        elif feature_strength == "weak":
            executive = "The dataset lacks predictive features. Domain-specific variables are needed."
        elif interpretation.get("is_model_limited"):
            executive = "Features are adequate but model cannot capture complex relationships."
        else:
            executive = "Analysis indicates reasonable data and model fit. Refinement opportunities exist."

        # KEY FINDING
        if primary_issue == "feature_gap":
            key = "Missing important predictors; model improvements alone won't solve this."
        elif primary_issue == "multicollinearity":
            key = "Redundant features distort model. Remove correlated predictors."
        elif primary_issue == "non_linearity":
            key = "Non-linear relationships detected. Advanced models (GB, RF) recommended."
        else:
            key = f"Primary issue: {primary_issue}. Address with recommended actions."

        # ACTION PRIORITY
        # FIX 1: Safe n_features access with fallback to 10
        total_features = signals.get("n_features", 10) or 10
        weak_features_count = features.get("weak_features", 0)

        if data_quality in ["poor", "fair"]:
            action = "CRITICAL: Fix data quality issues first. Re-evaluate after cleaning."
        elif weak_features_count > total_features * 0.5:
            action = "CRITICAL: Collect domain-specific features. Current set insufficient."
        elif len(features.get("redundant_pairs", [])) > 2:
            action = "HIGH: Remove multicollinear features. Then retry modeling."
        else:
            action = "MEDIUM: Try advanced models and feature interactions."

        return {
            "executive_insight": executive,
            "key_finding": key,
            "action_priority": action,
        }

    except Exception:
        return {
            "executive_insight": "Analysis complete.",
            "key_finding": "Review results and recommendations.",
            "action_priority": "Follow recommended actions.",
        }
