from __future__ import annotations

from typing import Dict, List

# Treat R² below 0.30 as weak predictive utility for action prioritization.
LOW_MODEL_THRESHOLD = 0.3


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


def _deduplicate(items: List[Dict]) -> List[Dict]:
    seen = set()
    deduped: List[Dict] = []
    for item in items:
        key = " ".join(str(item.get("content", "")).lower().split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def generate_ranked_insights(results: Dict, quality_summary: Dict, confidence_scores: Dict[str, float]) -> List[Dict]:
    ml_results = results.get("ml_results") or {}
    multicollinearity = results.get("multicollinearity") or {}
    recommendations = (results.get("recommendations") or {}).get("recommendations") or []
    rca = results.get("root_cause_analysis") or {}
    context = results.get("context") or {}
    data_quality = (results.get("data_quality") or {}).get("data_quality") or {}

    r2_value = float(ml_results.get("r2_score", 0.0))
    target_column = ml_results.get("target_column", "target")
    strongest_feature = ml_results.get("strongest_feature", "top feature")
    explained_pct = max(0.0, r2_value * 100)
    unexplained_pct = max(0.0, 100.0 - explained_pct)

    insights: List[Dict] = []
    model_actions = recommendations[:3]
    insights.append(
        _insight(
            rank=1,
            severity="CRITICAL" if r2_value < LOW_MODEL_THRESHOLD else "MEDIUM",
            category="MODEL_PERFORMANCE",
            content=(
                f"Model explains only {explained_pct:.1f}% of {target_column} variance "
                f"({strongest_feature} strongest), leaving {unexplained_pct:.1f}% unexplained."
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
                    f"Strong multicollinearity: {pair['feature_a']} (VIF={vif_map.get(pair['feature_a'], 0.0):.1f}) "
                    f"correlates {pair['correlation']:.2f} with {pair['feature_b']} "
                    f"(VIF={vif_map.get(pair['feature_b'], 0.0):.1f})."
                ),
                root_cause="Redundant features are measuring similar signal.",
                confidence={**confidence_scores, "finding": 0.99},
                actions=[item for item in recommendations if item.get("priority") in {"CRITICAL", "HIGH"}][:3],
            )
        )

    total_missing = int(quality_summary.get("total_missing", 0))
    duplicate_rows = int(quality_summary.get("duplicate_rows", 0))
    quality_text = (
        f"Data quality score is {data_quality.get('overall_score', 0):.1f}/100 with grade "
        f"{(results.get('data_quality') or {}).get('grade', 'N/A')}."
    )
    if total_missing or duplicate_rows:
        quality_text += (
            f" Found {total_missing} missing values and {duplicate_rows} duplicate rows needing treatment."
        )
    else:
        quality_text += " Completeness and uniqueness are strong."
    insights.append(
        _insight(
            rank=3,
            severity="MEDIUM",
            category="DATA_QUALITY",
            content=quality_text,
            root_cause="Outliers and schema consistency influence downstream reliability.",
            confidence=confidence_scores,
            actions=[{"priority": "HIGH", "action": "Apply data cleaning checks"}],
        )
    )

    insights.append(
        _insight(
            rank=4,
            severity="MEDIUM",
            category="CONTEXT",
            content=(
                f"Domain inferred as {context.get('domain', 'generic')} "
                f"({context.get('confidence', 'LOW')} confidence): {context.get('context', 'Generic dataset')}."
            ),
            root_cause="Domain context inferred from column keywords and structure.",
            confidence=confidence_scores,
            actions=[{"priority": "MEDIUM", "action": "Validate inferred domain with stakeholder input"}],
        )
    )

    deduped = _deduplicate(insights)
    severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    deduped.sort(key=lambda item: (severity_rank.get(item.get("severity"), 99), item.get("rank", 99)))
    for idx, item in enumerate(deduped, start=1):
        item["rank"] = idx
    return deduped
