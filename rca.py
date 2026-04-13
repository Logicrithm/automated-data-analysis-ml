from __future__ import annotations

from typing import Dict


def diagnose(signals: Dict, ml_results: Dict) -> Dict:
    r2 = float(ml_results.get("r2_score", 0.0))
    avg_corr = float(signals.get("avg_correlation", 0.0))
    max_corr = float(signals.get("max_correlation", 0.0))
    quality = float(signals.get("data_quality_score", 0.0))

    if r2 < 0.2:
        model_perf = "critical"
    elif r2 < 0.4:
        model_perf = "weak"
    elif r2 < 0.7:
        model_perf = "fair"
    else:
        model_perf = "good"

    if avg_corr < 0.2:
        feature_strength = "weak"
    elif avg_corr < 0.4:
        feature_strength = "moderate"
    else:
        feature_strength = "strong"

    if max_corr >= 0.95:
        multicollinearity = "critical"
    elif max_corr > 0.85:
        multicollinearity = "high"
    elif max_corr > 0.7:
        multicollinearity = "moderate"
    elif max_corr > 0.5:
        multicollinearity = "low"
    else:
        multicollinearity = "none"

    if quality < 60:
        data_quality = "poor"
    elif quality < 75:
        data_quality = "fair"
    elif quality < 90:
        data_quality = "good"
    else:
        data_quality = "excellent"

    return {
        "model_perf": model_perf,
        "feature_strength": feature_strength,
        "multicollinearity": multicollinearity,
        "data_quality": data_quality,
    }
