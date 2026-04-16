from __future__ import annotations


def generate_deep_summary(
    evidence: dict,
    decision: dict | None = None,
    causal_layer: dict | None = None,
) -> dict:
    decision = decision or {}
    causal_layer = causal_layer or {}

    if not evidence:
        return {
            "decision": decision.get("decision", "UNKNOWN"),
            "executive_insight": "Analysis complete but evidence is missing.",
            "key_finding": "UNKNOWN: Evidence was unavailable for decision-backed summary.",
            "action_priority": "LOW: Re-run analysis with complete evidence.",
        }

    decision_name = str(decision.get("decision", "UNKNOWN"))
    severity = str(decision.get("severity", "LOW"))
    r2_percentage = float(evidence.get("r2_percentage", 0.0))
    weak_feature_pct = int(evidence.get("weak_feature_pct", 0))
    redundant_pairs_count = int(evidence.get("redundant_pairs_count", 0))
    data_quality_score = float(evidence.get("data_quality_score", 0.0))

    root_cause = str(causal_layer.get("root_cause", "Root cause derived from evidence."))
    evidence_chain = causal_layer.get("evidence_chain") or []
    chain_text = " ".join([str(item) for item in evidence_chain[:2]])

    executive = (
        f"Primary decision: {decision_name} ({severity}). "
        f"{root_cause} {chain_text}".strip()
    )

    key_finding = (
        f"{decision_name}: R²={r2_percentage:.1f}%, weak features={weak_feature_pct}%, "
        f"redundant pairs={redundant_pairs_count}, data quality={data_quality_score:.1f}/100."
    )

    if severity == "CRITICAL":
        action = f"CRITICAL: Execute immediate remediation for {decision_name} before further modeling."
    elif severity == "HIGH":
        action = f"HIGH: Prioritize {decision_name} correction in the next iteration."
    elif severity == "MEDIUM":
        action = f"MEDIUM: Address {decision_name} with targeted experiments and validate gains."
    else:
        action = f"LOW: Monitor {decision_name} and optimize opportunistically."

    return {
        "decision": decision_name,
        "executive_insight": executive,
        "key_finding": key_finding,
        "action_priority": action,
        "evidence_summary": {
            "model_performance": f"R² = {r2_percentage:.1f}%",
            "feature_quality": f"{weak_feature_pct}% weak",
            "redundancy": f"{redundant_pairs_count} correlated pairs",
            "data_quality": f"{data_quality_score:.1f}/100",
        },
    }