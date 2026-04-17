# insights_generator.py (COMPLETE REPLACEMENT)
from __future__ import annotations

from typing import Dict, List

# Treat R² below 0.30 as weak predictive utility for action prioritization.
LOW_MODEL_THRESHOLD = 0.3
IMPROVEMENT_THRESHOLD = 0.05  # 5% - below this is negligible


def _insight(
    rank: int,
    severity: str,
    category: str,
    content: str,
    root_cause: str,
    confidence: Dict[str, float],
    actions: List[Dict],
) -> Dict:
    return {
        "rank": rank,
        "severity": severity,
        "category": category,
        "content": content,
        "root_cause": root_cause,
        "confidence": confidence,
        "actions": actions,
    }


def _content_fingerprint(content: str) -> str:
    return " ".join(str(content).lower().split())


def _deduplicate(items: List[Dict]) -> List[Dict]:
    seen = set()
    deduped: List[Dict] = []
    for item in items:
        key = _content_fingerprint(str(item.get("content", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def generate_ranked_insights(results: Dict, quality_summary: Dict, confidence_scores: Dict[str, float]) -> List[Dict]:
    ml_results = results.get("ml_results") or {}
    multicollinearity = results.get("multicollinearity") or {}
    recommendations_obj = results.get("recommendations") or []
    if isinstance(recommendations_obj, dict):
        recommendations = recommendations_obj.get("recommendations") or []
    elif isinstance(recommendations_obj, list):
        recommendations = recommendations_obj
    else:
        recommendations = []
    rca = results.get("root_cause_analysis") or {}
    context = results.get("context") or {}
    data_quality = (results.get("data_quality") or {}).get("data_quality") or {}
    evidence = results.get("evidence") or {}

    r2_value = float(ml_results.get("r2_score", 0.0))
    target_column = ml_results.get("target_column", "target")
    strongest_feature = ml_results.get("strongest_feature", "top feature")
    explained_pct = r2_value * 100
    unexplained_pct = 100.0 - explained_pct
    
    # ✅ FIX #3: ADD EXPERIMENT INTERPRETATION
    # Check if best_improvement indicates negligible gains
    best_improvement = float(evidence.get("best_improvement", 0.0))
    improvement_meaningful = best_improvement >= IMPROVEMENT_THRESHOLD
    
    improvement_note = (
        f"Best experiment improvement: +{best_improvement*100:.1f}% (meaningful)"
        if improvement_meaningful
        else f"Best experiment improvement: +{best_improvement*100:.1f}% (below {IMPROVEMENT_THRESHOLD*100:.0f}% threshold → negligible)"
    )

    insights: List[Dict] = []
    model_actions = recommendations[:3]
    performance_prefix = (
        f"Model performs worse than baseline (R²={r2_value:.3f}). "
        if r2_value < 0
        else ""
    )
    insights.append(
        _insight(
            rank=1,
            severity="CRITICAL" if r2_value < LOW_MODEL_THRESHOLD else "MEDIUM",
            category="MODEL_PERFORMANCE",
            content=(
                f"{performance_prefix}Model explains only {explained_pct:.1f}% of {target_column} variance "
                f"({strongest_feature} strongest), leaving {unexplained_pct:.1f}% unexplained. "
                f"{improvement_note}."  # NEW: Quantified experiment results
            ),
            root_cause=rca.get("root_cause", "Feature set insufficient for robust prediction."),
            confidence=confidence_scores,
            actions=model_actions,
        )
    )

    high_vif_pairs = multicollinearity.get("high_vif_pairs") or []
    if high_vif_pairs:
        pair = high_vif_pairs[0]
        vif_map = {item["feature"]: item["vif"] for item in (multicollinearity.get("vif") or [])}
        insights.append(
            _insight(
                rank=2,
                severity="HIGH",
                category="MULTICOLLINEARITY",
                content=(
                    f"Detected redundant feature pair: {pair.get('feature_a', 'feature1')} ↔ {pair.get('feature_b', 'feature2')} "
                    f"(r={pair.get('correlation', 0.0):.2f}). May cause unstable coefficient attribution."
                ),
                root_cause="Feature redundancy detected; not primary cause of low R².",
                confidence={"multicollinearity_likelihood": 0.85},
                actions=recommendations[1:3] if len(recommendations) > 1 else [],
            )
        )

    issue_count = len(quality_summary.get("issues", []))
    if issue_count > 0:
        insights.append(
            _insight(
                rank=3,
                severity="MEDIUM",
                category="DATA_QUALITY",
                content=f"Detected {issue_count} data quality issue(s). Address before advanced modeling.",
                root_cause="Data defects can limit model performance.",
                confidence={"data_quality_concern": 0.7},
                actions=recommendations[2:4] if len(recommendations) > 2 else [],
            )
        )

    return _deduplicate(insights)
