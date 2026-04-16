from __future__ import annotations

from typing import Dict


def generate_business_impact(decision: Dict, evidence: Dict) -> str:
    decision_name = str(decision.get("decision", "UNKNOWN"))
    r2 = float(evidence.get("r2_score", 0.0))
    r2_pct = float(evidence.get("r2_percentage", r2 * 100))
    missing_pct = float(evidence.get("missing_percentage", 0.0))
    weak_pct = int(evidence.get("weak_feature_pct", 0))
    redundant_pairs = int(evidence.get("redundant_pairs_count", 0))
    data_quality = float(evidence.get("data_quality_score", 0.0))

    if decision_name == "DATA_ISSUE":
        return (
            f"Data quality is {data_quality:.1f}/100 with {missing_pct:.1f}% missing values. "
            "Operational decisions based on incomplete records can lead to planning errors and avoidable cost."
        )

    if decision_name == "MULTICOLLINEARITY":
        return (
            f"{redundant_pairs} redundant feature pairs are distorting attribution. "
            "Teams may optimize the wrong levers, reducing ROI from pricing, marketing, or process changes."
        )

    if decision_name == "NON_LINEARITY":
        return (
            f"Current linear fit explains only {r2_pct:.1f}% variance. "
            "If non-linear effects are ignored, forecasts and risk estimates will be systematically off."
        )

    if decision_name == "FEATURE_GAP":
        return (
            f"Model explains only {r2_pct:.1f}% variance with limited informative predictors. "
            "Critical business drivers are missing, so decisions may remain guess-driven."
        )

    # WEAK_FEATURES / fallback
    return (
        f"{weak_pct}% of features are weak and model fit is {r2_pct:.1f}% R². "
        "Prediction-led decisions are likely low-confidence and may miss performance targets."
    )