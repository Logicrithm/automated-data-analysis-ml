from __future__ import annotations


def build_evidence(signals: dict, feature_analysis: dict, ml_results: dict, diagnosis: dict) -> dict:
    _ = diagnosis  # Reserved for future evidence enrichments from RCA output.
    weak_features = feature_analysis.get("weak_features", 0)
    total_features = signals.get("n_features", 1)
    weak_feature_pct = int((weak_features / max(total_features, 1)) * 100)

    correlations = signals.get("correlations", [])
    if not correlations and "max_correlation" in signals:
        correlations = [signals.get("max_correlation", 0.0)]
    strongest_corr = max([abs(c) for c in correlations]) if correlations else 0.0

    redundant_pairs = feature_analysis.get("redundant_pairs", [])
    num_redundant = len(redundant_pairs)
    max_redundancy = max([p.get("correlation", 0) for p in redundant_pairs], default=0.0)

    r2_score = float(ml_results.get("r2_score", 0.0))
    data_quality_score = signals.get("data_quality_score", 0)
    missing_pct = signals.get("missing_percentage", 0)

    return {
        "weak_features": weak_features,
        "total_features": total_features,
        "weak_feature_pct": weak_feature_pct,
        "strongest_correlation": round(float(strongest_corr), 2),
        "redundant_pairs_count": num_redundant,
        "max_redundancy_correlation": round(float(max_redundancy), 2),
        "r2_score": round(float(r2_score), 4),
        "r2_percentage": round(float(r2_score) * 100, 1),
        "data_quality_score": data_quality_score,
        "missing_percentage": missing_pct,
        "has_weak_features": weak_feature_pct > 50,
        "has_redundancy": num_redundant > 2,
        "poor_model_fit": r2_score < 0.3,
        "poor_data_quality": data_quality_score < 70,
    }
