from __future__ import annotations

from typing import Dict, List


def _deduplicate(items: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def generate_ranked_insights(results: Dict, quality_summary: Dict) -> List[str]:
    insights: List[str] = []

    ml_results = results.get("ml_results") or {}
    r2_value = float(ml_results.get("r2_score", 0.0))
    if ml_results.get("problem_type") == "regression":
        target_column = ml_results.get("target_column", "target")
        insights.append(f"Model explains {r2_value:.1%} of {target_column} variance; {max(0.0, 1.0 - r2_value):.1%} remains unexplained.")
        if r2_value < 0.5:
            insights.append(
                f"The model with {ml_results.get('strongest_feature', 'top feature')} as the strongest predictor explains only {r2_value:.1%} of {target_column} variance."
            )
        else:
            insights.append(
                f"{ml_results.get('strongest_feature', 'Top feature')} is the strongest predictor, and total explained variance is {r2_value:.1%}."
            )

        if r2_value < 0.2:
            insights.append("Missing key predictive features are likely driving most of the unexplained variance.")
            insights.append("Action: Collect additional features or use non-linear models with interaction effects.")

    total_missing = int(quality_summary.get("total_missing", 0))
    if total_missing == 0:
        if r2_value < 0.5:
            insights.append("Data Quality: Clean data (no missing values), but feature selection is weak.")
        else:
            insights.append("Data Quality: Clean data (no missing values).")
    else:
        insights.append(f"Data Quality: Found {total_missing} missing values that should be handled before modeling.")

    return _deduplicate(insights)[:5]
