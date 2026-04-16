from __future__ import annotations

def generate_business_impact(decision: dict | None, evidence: dict | None) -> str:
    decision = decision or {}
    evidence = evidence or {}

    decision_name = str(decision.get("decision", "UNKNOWN"))
    r2_percentage = evidence.get("r2_percentage")
    r2_pct = r2_percentage if r2_percentage is not None else evidence.get("r2_score", 0.0) * 100.0
    r2_pct = max(0.0, min(100.0, float(r2_pct)))
    weak_pct = int(evidence.get("weak_feature_pct", 0))
    strongest_corr = float(evidence.get("strongest_correlation", evidence.get("strongest_corr", 0.0)))
    data_quality = float(evidence.get("data_quality_score", 0.0))
    missing_pct = float(evidence.get("missing_percentage", 0.0))
    redundant_pairs = int(evidence.get("redundant_pairs_count", evidence.get("redundant_pairs", 0)))
    nonlinear_gain = float(evidence.get("nonlinear_gain", 0.0))
    total_features = int(evidence.get("total_features", 0))
    unexplained_pct = max(0.0, 100.0 - r2_pct)

    if decision_name == "DATA_ISSUE":
        return (
            f"Data quality is {data_quality:.1f}/100 with {missing_pct:.1f}% missing values, "
            f"so current predictions (R²={r2_pct:.1f}%) are not reliable for operational decisions."
        )

    if decision_name == "MULTICOLLINEARITY":
        return (
            f"{redundant_pairs} redundant feature pairs indicate overlapping signals; this can destabilize feature "
            f"importance and create volatile decisions even when R² is {r2_pct:.1f}%."
        )

    if decision_name == "NON_LINEARITY":
        return (
            f"Linear fit explains only {r2_pct:.1f}% variance while non-linear models improve by {nonlinear_gain:.2f} R², "
            f"leaving {unexplained_pct:.1f}% variance under-captured by simple models."
        )

    if decision_name == "FEATURE_GAP":
        return (
            f"Only {total_features} predictive features were available and {weak_pct}% are weak, "
            f"which caps model quality at R²={r2_pct:.1f}% and limits forecast usefulness."
        )

    return (
        f"{weak_pct}% of features are weak and strongest correlation is {strongest_corr:.2f}; "
        f"with R²={r2_pct:.1f}% ({unexplained_pct:.1f}% unexplained variance), predictions carry elevated uncertainty."
    )