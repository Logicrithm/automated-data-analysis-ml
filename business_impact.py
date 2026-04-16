from __future__ import annotations

def generate_business_impact(decision: dict | None, evidence: dict | None) -> str:
    """
    Generate business-facing impact statement from decision + evidence.
    
    Args:
        decision: Root cause decision from decision_engine
        evidence: Quantified evidence from evidence.py
        
    Returns:
        Plain-text business impact statement (no jargon)
    """
    decision = decision or {}
    evidence = evidence or {}
    
    decision_name = str(decision.get("decision", "UNKNOWN"))
    r2_pct = float(evidence.get("r2_percentage", 0.0))
    weak_pct = int(evidence.get("weak_feature_pct", 0))
    data_quality = float(evidence.get("data_quality_score", 0.0))
    missing_pct = float(evidence.get("missing_percentage", 0.0))
    redundant_pairs = int(evidence.get("redundant_pairs_count", 0))
    
    # Decision-specific impact narratives
    if decision_name == "DATA_ISSUE":
        return (
            f"Data quality at {data_quality:.1f}/100 with {missing_pct:.1f}% missing values "
            f"makes predictive models unreliable. Current model fit (R²={r2_pct:.1f}%) cannot be trusted. "
            f"Impact: Business decisions based on this data carry unquantified risk. "
            f"Action priority: Data remediation before model deployment."
        )
    
    elif decision_name == "MULTICOLLINEARITY":
        return (
            f"Detected {redundant_pairs} redundant feature pairs (overlapping signal). "
            f"This causes model coefficients to become unstable and feature importance rankings unreliable. "
            f"Impact: Model predictions are fragile—small data changes produce large prediction swings. "
            f"Action priority: Feature reduction to stabilize interpretability."
        )
    
    elif decision_name == "NON_LINEARITY":
        return (
            f"Linear models achieve only R²={r2_pct:.1f}%. The true relationship is non-linear. "
            f"This means {100-r2_pct:.1f}% of variance is left unexplained by simple models. "
            f"Impact: Predictive power is capped; business forecasts will be inaccurate. "
            f"Action priority: Switch to tree-based or neural models to capture non-linear patterns."
        )
    
    elif decision_name == "FEATURE_GAP":
        total_features = int(evidence.get("total_features", 0))
        return (
            f"Only {total_features} features available with {weak_pct}% showing weak correlation. "
            f"Current model fit plateaus at R²={r2_pct:.1f}%. "
            f"Impact: Missing key business variables limits prediction capability; further model tuning will not help. "
            f"Action priority: Collect additional domain variables (sales pipeline stage, competitor activity, etc.)."
        )
    
    else:  # WEAK_FEATURES or fallback
        return (
            f"{weak_pct}% of features show weak correlation with target; "
            f"strongest single feature correlation is only {evidence.get('strongest_correlation', 0.0):.2f}. "
            f"Current fit (R²={r2_pct:.1f}%) is below useful threshold. "
            f"Impact: Predictions will have high uncertainty margin; not suitable for high-stakes decisions. "
            f"Action priority: Feature engineering to create more predictive variables."
        )
