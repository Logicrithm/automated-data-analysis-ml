from __future__ import annotations

from typing import Dict, List


def _deduplicate(items: List[Dict]) -> List[Dict]:
    seen = set()
    deduped: List[Dict] = []
    for item in items:
        normalized = str(item.get("content", "")).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(item)
    return deduped


def _confidence(confidence_levels: Dict[str, str], category: str, default: str = "MEDIUM") -> str:
    return confidence_levels.get(category, default)


def _insight(content: str, severity: str, confidence: str, actionable: bool, category: str) -> Dict:
    return {
        "content": content,
        "severity": severity,
        "confidence": confidence,
        "actionable": actionable,
        "category": category,
    }


def generate_ranked_insights(results: Dict, quality_summary: Dict, confidence_levels: Dict[str, str]) -> List[Dict]:
    insights: List[Dict] = []

    ml_results = results.get("ml_results") or {}
    multicollinearity = results.get("multicollinearity") or {}
    r2_value = float(ml_results.get("r2_score", 0.0))
    if ml_results.get("problem_type") == "regression":
        target_column = ml_results.get("target_column", "target")
        strongest_feature = ml_results.get("strongest_feature", "top feature")
        explained_pct = max(0.0, min(100.0, r2_value * 100))
        unexplained_pct = 100.0 - explained_pct
        insights.append(
            _insight(
                (
                    f"Model explains only {explained_pct:.1f}% of {target_column} variance with "
                    f"{strongest_feature} as strongest predictor, leaving {unexplained_pct:.1f}% unexplained."
                ),
                "CRITICAL" if r2_value < 0.5 else "MEDIUM",
                _confidence(confidence_levels, "MODEL_PERFORMANCE"),
                True,
                "MODEL_PERFORMANCE",
            )
        )

        high_vif_pairs = multicollinearity.get("high_vif_pairs") or []
        high_vif_features = multicollinearity.get("high_vif_features") or []
        if high_vif_pairs:
            pair = high_vif_pairs[0]
            feature_a = pair["feature_a"]
            feature_b = pair["feature_b"]
            corr = float(pair["correlation"])
            vif_map = {item["feature"]: item["vif"] for item in high_vif_features}
            insights.append(
                _insight(
                    (
                        f"Strong multicollinearity detected: {feature_a} (VIF={vif_map.get(feature_a, 0.0):.1f}) "
                        f"correlates {corr:.2f} with {feature_b} (VIF={vif_map.get(feature_b, 0.0):.1f}). "
                        "This distorts model coefficients."
                    ),
                    "HIGH",
                    _confidence(confidence_levels, "MULTICOLLINEARITY", "HIGH"),
                    True,
                    "MULTICOLLINEARITY",
                )
            )
        elif high_vif_features:
            top = high_vif_features[0]
            insights.append(
                _insight(
                    f"Strong multicollinearity detected: {top['feature']} has VIF {top['vif']:.1f}, which can distort coefficients.",
                    "HIGH",
                    _confidence(confidence_levels, "MULTICOLLINEARITY", "MEDIUM"),
                    True,
                    "MULTICOLLINEARITY",
                )
            )

        if r2_value < 0.6:
            insights.append(
                _insight(
                    "Likely missing: neighborhood/location quality, property condition rating, property age, lot size, amenities proximity—these typically drive 40-60% of price variance.",
                    "HIGH",
                    _confidence(confidence_levels, "FEATURES"),
                    True,
                    "FEATURES",
                )
            )

    total_missing = int(quality_summary.get("total_missing", 0))
    duplicate_rows = int(quality_summary.get("duplicate_rows", 0))
    if total_missing == 0:
        quality_msg = (
            f"Data quality is excellent (zero missing values, {duplicate_rows} duplicates detected). "
            "However, feature selection is insufficient - domain knowledge needed for additional features."
            if r2_value < 0.6
            else f"Data quality is strong (zero missing values, {duplicate_rows} duplicates detected)."
        )
        insights.append(
            _insight(
                quality_msg,
                "MEDIUM",
                _confidence(confidence_levels, "DATA_QUALITY"),
                bool(r2_value < 0.6),
                "DATA_QUALITY",
            )
        )
    else:
        insights.append(
            _insight(
                f"Data quality issue detected: found {total_missing} missing values and {duplicate_rows} duplicate rows that should be handled before modeling.",
                "HIGH",
                _confidence(confidence_levels, "DATA_QUALITY", "HIGH"),
                True,
                "DATA_QUALITY",
            )
        )

    insights.append(
        _insight(
            "Recommendations: (1) Collect location-based features, (2) Add property condition ratings, (3) Try non-linear models (Random Forest), (4) Apply feature interaction terms.",
            "MEDIUM",
            _confidence(confidence_levels, "RECOMMENDATIONS"),
            True,
            "FEATURES",
        )
    )

    severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    deduped = _deduplicate(insights)
    deduped.sort(key=lambda item: severity_rank.get(item.get("severity"), 99))
    return deduped[:6]
