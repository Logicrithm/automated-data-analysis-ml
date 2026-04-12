from __future__ import annotations

from typing import Dict, List


def generate_ranked_insights(results: Dict, quality_summary: Dict) -> List[str]:
    insights: List[str] = []

    ml_results = results.get("ml_results") or {}
    if ml_results.get("problem_type") == "regression":
        r2_value = float(ml_results.get("r2_score", 0.0))
        insights.append(ml_results.get("interpretation", "🚨 Model is unreliable - R² too low"))
        insights.append(
            f"{ml_results.get('strongest_feature', 'Top feature')} is the strongest predictor but explains only {r2_value:.1%} of price variance."
        )

        if r2_value < 0.2:
            insights.append("Missing key features: likely location, neighborhood, amenities, and condition are driving prices.")
            insights.append("Action: Collect additional features or use non-linear models with interaction effects.")

    total_missing = int(quality_summary.get("total_missing", 0))
    if total_missing == 0:
        insights.append("Data Quality: Clean data (no missing values), but feature selection is weak.")
    else:
        insights.append(f"Data Quality: Found {total_missing} missing values that should be handled before modeling.")

    return insights[:5]
