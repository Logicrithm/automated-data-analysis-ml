from __future__ import annotations


def build_evidence(
    signals: dict, feature_analysis: dict, ml_results: dict, diagnosis: dict | None = None
) -> dict:
    _ = diagnosis

    weak_features = int(feature_analysis.get("weak_features", 0))
    predictor_count = int(feature_analysis.get("predictor_count", 0))
    weak_feature_pct = int(round((weak_features / max(predictor_count, 1)) * 100))

    correlations = signals.get("correlations", [])
    if not correlations and "max_correlation" in signals:
        correlations = [signals.get("max_correlation", 0.0)]
    strongest_corr = max([abs(float(c)) for c in correlations], default=0.0)

    redundant_pairs = feature_analysis.get("redundant_pairs", [])
    num_redundant = len(redundant_pairs)
    max_redundancy = max([float(p.get("correlation", 0.0)) for p in redundant_pairs], default=0.0)

    r2_score = float(ml_results.get("r2_score", 0.0))
    data_quality_score = float(signals.get("data_quality_score", 0.0))
    missing_pct = float(signals.get("missing_percentage", 0.0))

    model_comparison = ml_results.get("model_comparison", {}) or {}
    models = model_comparison.get("models", []) if isinstance(model_comparison, dict) else []

    linear_r2 = None
    best_r2 = None
    for m in models:
        name = str(m.get("name", "")).lower()
        r2v = float(m.get("r2_score", 0.0))
        if "linear regression" in name:
            linear_r2 = r2v
        if best_r2 is None or r2v > best_r2:
            best_r2 = r2v

    nonlinear_gain = 0.0
    if linear_r2 is not None and best_r2 is not None:
        nonlinear_gain = best_r2 - linear_r2

    return {
        "weak_features": weak_features,
        "total_features": predictor_count,
        "weak_feature_pct": weak_feature_pct,
        "strongest_correlation": round(strongest_corr, 2),
        "redundant_pairs_count": int(num_redundant),
        "max_redundancy_correlation": round(max_redundancy, 2),
        "r2_score": round(r2_score, 4),
        "r2_percentage": round(r2_score * 100, 1),
        "data_quality_score": round(data_quality_score, 1),
        "missing_percentage": round(missing_pct, 1),
        "linear_r2": round(float(linear_r2), 4) if linear_r2 is not None else None,
        "best_model_r2": round(float(best_r2), 4) if best_r2 is not None else None,
        "nonlinear_gain": round(float(nonlinear_gain), 4),
        "strongest_corr": round(strongest_corr, 2),
        "redundant_pairs": int(num_redundant),
        "max_corr": round(max_redundancy, 2),
        "has_weak_features": weak_feature_pct > 50,
        "has_redundancy": num_redundant > 2,
        "poor_model_fit": r2_score < 0.3,
        "poor_data_quality": data_quality_score < 70,
    }
