from __future__ import annotations

from typing import Dict, List


def _confidence_label(decision: Dict, evidence: Dict) -> str:
    c = float(decision.get("confidence", 0.0))
    if c >= 0.8:
        return "HIGH"
    if c >= 0.6:
        return "MEDIUM"
    return "LOW"


def _top_actions(recommendations: List[Dict], n: int = 3) -> List[Dict]:
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    recs = recommendations or []
    recs = sorted(recs, key=lambda r: priority_order.get(str(r.get("priority", "LOW")), 99))
    return recs[:n]


def generate_final_output(
    decision: Dict,
    evidence: Dict,
    causal_layer: Dict,
    recommendations: List[Dict],
    business_impact: str,
) -> str:
    decision_name = str(decision.get("decision", "UNKNOWN"))
    severity = str(decision.get("severity", "LOW"))
    confidence = _confidence_label(decision, evidence)

    # Evidence numbers (explicit)
    r2_pct = float(evidence.get("r2_percentage", float(evidence.get("r2_score", 0.0)) * 100))
    weak_pct = int(evidence.get("weak_feature_pct", 0))
    strongest = float(evidence.get("strongest_correlation", 0.0))
    redundant = int(evidence.get("redundant_pairs_count", 0))
    max_corr = float(evidence.get("max_redundancy_correlation", 0.0))
    dqs = float(evidence.get("data_quality_score", 0.0))
    miss = float(evidence.get("missing_percentage", 0.0))

    root_cause = str(causal_layer.get("root_cause", "Root cause derived from evidence."))
    chain = causal_layer.get("evidence_chain") or []
    chain_txt = " ".join(str(x) for x in chain[:2])

    actions = _top_actions(recommendations, 3)
    action_lines = []
    for i, a in enumerate(actions, start=1):
        action_lines.append(
            f"{i}. {a.get('action','N/A')} "
            f"(Reason: {a.get('reason','N/A')}; Impact: {a.get('impact','N/A')}; Effort: {a.get('effort','N/A')})"
        )

    return (
        f"Problem:\n"
        f"{decision_name} (Severity: {severity})\n\n"
        f"Why:\n"
        f"{root_cause} {chain_txt}\n"
        f"Evidence: R²={r2_pct:.1f}%, weak_feature_pct={weak_pct}%, strongest_corr={strongest:.2f}, "        f"redundant_pairs={redundant}, max_corr={max_corr:.2f}, data_quality={dqs:.1f}/100, missing={miss:.1f}%.\n\n"
        f"Business Impact:\n"
        f"{business_impact}\n\n"
        f"Action Plan:\n"
        f"{chr(10).join(action_lines) if action_lines else '1. No valid actions available.'}\n\n"
        f"Confidence:\n"
        f"{confidence}"
    )
