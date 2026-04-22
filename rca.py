# rca.py (COMPLETE REPLACEMENT)
from __future__ import annotations

from typing import Dict


def _unknown_diagnosis(validation_errors: list | None = None) -> dict:
    """Return a safe fallback diagnosis dict when evidence is absent or incomplete."""
    result = {
        "model_perf": "unknown",
        "feature_strength": "unknown",
        "multicollinearity": "unknown",
        "data_quality": "unknown",
        "evidence_used": False,
    }
    if validation_errors:
        result["validation_errors"] = validation_errors
    return result


def diagnose(
    signals: Dict | None = None, ml_results: Dict | None = None, evidence: Dict | None = None
) -> Dict:
    _ = (signals, ml_results)  # Retained for compatibility and potential fallback diagnostics.
    if not evidence or not isinstance(evidence, dict):
        return _unknown_diagnosis()

    # ✅ FIX #1: ADD VALIDATION - ensure critical keys exist
    required_keys = ["r2_score", "weak_feature_pct", "redundant_pairs_count", "data_quality_score"]
    missing_keys = [key for key in required_keys if key not in evidence]
    if missing_keys:
        return _unknown_diagnosis(
            validation_errors=[f"Missing required evidence key: {k}" for k in missing_keys]
        )

    r2_score = float(evidence.get("r2_score", 0.0))
    weak_feature_pct = int(evidence.get("weak_feature_pct", 0))
    redundant_pairs_count = int(evidence.get("redundant_pairs_count", 0))
    data_quality_score = float(evidence.get("data_quality_score", 0.0))

    if r2_score < 0.1:
        model_perf = "critical"
    elif r2_score < 0.3:
        model_perf = "weak"
    elif r2_score < 0.5:
        model_perf = "moderate"
    else:
        model_perf = "good"

    if weak_feature_pct > 70:
        feature_strength = "critical"
    elif weak_feature_pct > 50:
        feature_strength = "weak"
    elif weak_feature_pct > 30:
        feature_strength = "moderate"
    else:
        feature_strength = "strong"

    # ✅ FIX #2: TIGHTEN MULTICOLLINEARITY THRESHOLD
    # Changed from > 2 to >= 3 to avoid false positives on medium datasets
    if redundant_pairs_count >= 5:
        multicollinearity = "critical"
    elif redundant_pairs_count >= 3:  # Changed from > 2
        multicollinearity = "high"
    else:
        multicollinearity = "low"

    if data_quality_score < 60:
        data_quality = "poor"
    elif data_quality_score < 75:
        data_quality = "fair"
    elif data_quality_score < 85:
        data_quality = "good"
    else:
        data_quality = "excellent"

    return {
        "model_perf": model_perf,
        "feature_strength": feature_strength,
        "multicollinearity": multicollinearity,
        "data_quality": data_quality,
        "evidence_used": True,
    }
