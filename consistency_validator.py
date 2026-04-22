from __future__ import annotations

import re
from typing import Dict, List


_ALLOWED_ACTION_KEYWORDS = {
    "DATA_ISSUE": ["data", "missing", "quality", "clean"],
    "MULTICOLLINEARITY": ["multicollinearity", "pca", "redundant", "vif", "correlated"],
    "NON_LINEARITY": ["non-linear", "tree", "boost", "interaction", "random forest", "gradient boosting"],
    "WEAK_FEATURES": ["feature", "engineering", "domain", "signal"],
    "WEAK_SIGNAL": ["feature", "engineering", "domain", "signal", "predictive", "target", "collect", "data"],
    "FEATURE_GAP": ["collect", "feature", "domain", "variables"],
}
WEAK_SIGNAL_R2_THRESHOLD = 0.10
NEGLIGIBLE_IMPROVEMENT_THRESHOLD = 0.05


def _is_multicollinearity_over_weak_signal(decision_name: str, r2_score: float) -> bool:
    """Return True when decision is MULTICOLLINEARITY but R² is too low – should be WEAK_SIGNAL."""
    return decision_name == "MULTICOLLINEARITY" and r2_score < WEAK_SIGNAL_R2_THRESHOLD


def _is_weak_signal_misclassified(decision_name: str, r2_score: float, best_improvement: float) -> bool:
    """Return True when WEAK_SIGNAL is selected but neither criterion (low R² or negligible gain) holds."""
    return (
        decision_name == "WEAK_SIGNAL"
        and r2_score >= WEAK_SIGNAL_R2_THRESHOLD
        and best_improvement >= NEGLIGIBLE_IMPROVEMENT_THRESHOLD
    )


def _has_keyword_match(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    patterns = [re.compile(rf"(?<!\w){re.escape(keyword.lower())}(?!\w)") for keyword in keywords]
    for pattern in patterns:
        if pattern.search(lowered):
            return True
    return False


def _build_verdict(decision: Dict) -> Dict:
    decision_name = str(decision.get("decision", "UNKNOWN"))
    confidence = float(decision.get("confidence", 0.5))
    return {
        "primary_issue": decision_name.lower(),
        "decision": decision_name,
        "severity": decision.get("severity", "LOW"),
        "confidence": confidence,
        "is_data_problem": decision_name == "DATA_ISSUE",
        "is_feature_problem": decision_name in {"MULTICOLLINEARITY", "WEAK_FEATURES", "FEATURE_GAP"},
        "is_model_problem": decision_name == "NON_LINEARITY",
    }


def validate_consistency(
    decision: Dict | None,
    recommendations: List[Dict] | None,
    deep_summary: Dict | None,
    verdict: Dict | None = None,
) -> Dict:
    decision = decision or {}
    deep_summary = deep_summary or {}
    verdict = verdict or _build_verdict(decision)
    recommendations = recommendations or []

    decision_name = str(decision.get("decision", "UNKNOWN"))
    severity = str(decision.get("severity", "LOW"))
    r2_score = float(decision.get("r2_score", 0.0))
    best_improvement = float(decision.get("best_improvement", 0.0))
    allowed_keywords = _ALLOWED_ACTION_KEYWORDS.get(decision_name, [])

    corrections: List[str] = []
    flags: List[str] = []
    checks = {
        "decision_priority_consistent": True,
        "improvement_is_meaningful": best_improvement >= NEGLIGIBLE_IMPROVEMENT_THRESHOLD,
        "recommendations_match_decision": True,
        "no_conflicting_priorities": True,
        "summary_aligned_with_decision": True,
        "all_recommendations_have_evidence": True,
    }

    if _is_multicollinearity_over_weak_signal(decision_name, r2_score):
        checks["decision_priority_consistent"] = False
        flags.append(
            f"Conflict detected: R²={r2_score:.3f} is below {WEAK_SIGNAL_R2_THRESHOLD:.2f}, "
            "but primary decision is MULTICOLLINEARITY. Expected WEAK_SIGNAL."
        )
    if _is_weak_signal_misclassified(decision_name, r2_score, best_improvement):
        checks["decision_priority_consistent"] = False
        flags.append(
            f"Conflict detected: WEAK_SIGNAL selected with R²={r2_score:.3f} and "
            f"best improvement {best_improvement * 100:.1f}% (not negligible)."
        )

    if best_improvement < NEGLIGIBLE_IMPROVEMENT_THRESHOLD:
        flags.append(
            f"Best observed improvement {best_improvement * 100:.1f}% is below "
            f"{NEGLIGIBLE_IMPROVEMENT_THRESHOLD * 100:.0f}% and should be treated as negligible."
        )

    validated_recommendations: List[Dict] = []
    seen_actions = set()
    for rec in recommendations:
        action = str(rec.get("action", "")).strip()
        reason = str(rec.get("reason", "")).strip()
        if not action:
            corrections.append("Removed recommendation with empty action.")
            continue

        action_key = action.lower()
        if action_key in seen_actions:
            corrections.append(f"Removed duplicate recommendation: {action}.")
            continue

        if allowed_keywords and not _has_keyword_match(action_key, allowed_keywords):
            corrections.append(f"Removed decision-mismatched recommendation: {action}.")
            checks["recommendations_match_decision"] = False
            continue

        fixed = dict(rec)
        fixed["priority"] = fixed.get("priority") or severity
        fixed["reason"] = reason or f"Aligned with decision {decision_name}: {decision.get('dominant_signal', 'evidence trigger')}"
        fixed["evidence"] = fixed.get("evidence") or {
            "dominant_signal": decision.get("dominant_signal", "not provided"),
            "secondary_signals": decision.get("secondary_signals", []),
        }
        seen_actions.add(action_key)
        validated_recommendations.append(fixed)

    if not validated_recommendations and decision_name != "UNKNOWN":
        validated_recommendations.append(
            {
                "priority": severity,
                "action": f"Execute primary remediation for {decision_name}",
                "reason": f"No aligned recommendations remained after validation. Triggered by: {decision.get('dominant_signal', 'N/A')}",
                "impact": "HIGH",
                "effort": "MEDIUM",
                "evidence": {
                    "dominant_signal": decision.get("dominant_signal", "N/A"),
                    "secondary_signals": decision.get("secondary_signals", []),
                },
            }
        )
        corrections.append("Inserted fallback recommendation to preserve decision alignment.")

    key_finding = str(deep_summary.get("key_finding", "")).strip()
    if decision_name and decision_name not in key_finding.upper():
        deep_summary["key_finding"] = f"{decision_name}: {key_finding or 'Primary decision established from evidence.'}"
        corrections.append("Updated deep summary key finding to include primary decision.")

    if flags:
        existing_flags = deep_summary.get("validation_flags")
        if isinstance(existing_flags, list):
            deep_summary["validation_flags"] = existing_flags + flags
        else:
            deep_summary["validation_flags"] = flags

    verdict = _build_verdict(decision)

    return {
        "recommendations": validated_recommendations,
        "deep_summary": deep_summary,
        "verdict": verdict,
        "validation_report": {
            "decision": decision_name,
            "metrics": {
                "r2_score": r2_score,
                "best_improvement": best_improvement,
                "negligible_improvement_threshold": NEGLIGIBLE_IMPROVEMENT_THRESHOLD,
            },
            "checks": checks,
            "flags": flags,
            "corrections": corrections,
            "correction_count": len(corrections),
        },
    }
