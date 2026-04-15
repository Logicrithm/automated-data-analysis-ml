from __future__ import annotations

from typing import Dict, List, Tuple


def _clip_confidence(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 2)


def _legacy_diagnosis(decision: str, evidence: Dict) -> Dict:
    weak_feature_pct = int(evidence.get("weak_feature_pct", 0))
    redundant_pairs_count = int(evidence.get("redundant_pairs_count", 0))
    r2_score = float(evidence.get("r2_score", 0.0))
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

    if redundant_pairs_count > 5:
        multicollinearity = "critical"
    elif redundant_pairs_count > 2:
        multicollinearity = "high"
    elif redundant_pairs_count > 0:
        multicollinearity = "moderate"
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

    if decision == "DATA_ISSUE":
        model_perf = model_perf if model_perf != "good" else "moderate"

    return {
        "model_perf": model_perf,
        "feature_strength": feature_strength,
        "multicollinearity": multicollinearity,
        "data_quality": data_quality,
    }


def _make_response(
    decision: str,
    severity: str,
    confidence: float,
    dominant_signal: str,
    secondary_signals: List[str],
    evidence: Dict,
) -> Dict:
    response = {
        "decision": decision,
        "severity": severity,
        "confidence": _clip_confidence(confidence),
        "dominant_signal": dominant_signal,
        "secondary_signals": [item for item in secondary_signals if item],
    }
    response.update(_legacy_diagnosis(decision, evidence))
    return response


def decide_root_cause(evidence: Dict | None) -> Dict:
    """
    Decide one primary root cause from evidence using a deterministic hierarchy:
    data -> structure -> model -> features.
    """
    if not evidence:
        return _make_response(
            "UNKNOWN",
            "LOW",
            0.35,
            "No evidence was provided",
            ["Run evidence builder before decisioning"],
            {},
        )

    data_quality_score = float(evidence.get("data_quality_score", 0.0))
    missing_percentage = float(evidence.get("missing_percentage", 0.0))
    redundant_pairs_count = int(evidence.get("redundant_pairs_count", 0))
    max_redundancy = float(evidence.get("max_redundancy_correlation", 0.0))
    poor_model_fit = bool(evidence.get("poor_model_fit", False))
    r2_score = float(evidence.get("r2_score", 0.0))
    weak_feature_pct = int(evidence.get("weak_feature_pct", 0))
    strongest_correlation = float(evidence.get("strongest_correlation", 0.0))
    total_features = int(evidence.get("total_features", 0))

    if data_quality_score < 70 or missing_percentage >= 10:
        severity = "CRITICAL" if data_quality_score < 60 or missing_percentage >= 20 else "HIGH"
        return _make_response(
            "DATA_ISSUE",
            severity,
            0.78 + (0.15 if data_quality_score < 60 else 0.0),
            f"Data quality score {data_quality_score:.1f}/100; missing {missing_percentage:.1f}%",
            [
                "Poor data quality blocks reliable modeling",
                f"R² currently {r2_score:.3f}",
            ],
            evidence,
        )

    if redundant_pairs_count > 2 or max_redundancy >= 0.85:
        severity = "HIGH" if redundant_pairs_count <= 5 else "CRITICAL"
        return _make_response(
            "MULTICOLLINEARITY",
            severity,
            0.74 + min(redundant_pairs_count, 8) * 0.02,
            f"{redundant_pairs_count} feature pairs above redundancy threshold (max {max_redundancy:.2f})",
            [
                f"Weak feature percentage {weak_feature_pct}%",
                f"Model fit R²={r2_score:.3f}",
            ],
            evidence,
        )

    if poor_model_fit and strongest_correlation >= 0.2:
        return _make_response(
            "NON_LINEARITY",
            "HIGH" if r2_score < 0.2 else "MEDIUM",
            0.69 + min(strongest_correlation, 0.5) * 0.3,
            f"Strongest feature correlation {strongest_correlation:.2f} but R² only {r2_score:.3f}",
            [
                "Linear model underfits available signal",
                f"Weak feature percentage {weak_feature_pct}%",
            ],
            evidence,
        )

    if total_features < 5 and poor_model_fit:
        return _make_response(
            "FEATURE_GAP",
            "HIGH",
            0.7,
            f"Only {total_features} features available with weak fit (R²={r2_score:.3f})",
            [
                f"Weak feature percentage {weak_feature_pct}%",
                "Feature volume is too low for robust generalization",
            ],
            evidence,
        )

    if weak_feature_pct > 50 or strongest_correlation < 0.15:
        severity = "HIGH" if weak_feature_pct > 70 else "MEDIUM"
        return _make_response(
            "WEAK_FEATURES",
            severity,
            0.65 + min(weak_feature_pct, 90) / 300,
            f"{weak_feature_pct}% weak features; strongest correlation {strongest_correlation:.2f}",
            [
                f"Model fit R²={r2_score:.3f}",
                "Feature signal is insufficient for stable prediction",
            ],
            evidence,
        )

    return _make_response(
        "WEAK_FEATURES",
        "LOW",
        0.55,
        "No critical risk thresholds crossed",
        [
            f"R²={r2_score:.3f}",
            f"Weak feature percentage {weak_feature_pct}%",
        ],
        evidence,
    )
