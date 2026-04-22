from __future__ import annotations

from typing import Dict, List

BASE_DATA_ISSUE_CONFIDENCE = 0.78
DATA_ISSUE_CRITICAL_CONFIDENCE_BOOST = 0.15
BASE_MULTICOLLINEARITY_CONFIDENCE = 0.74
MULTICOLLINEARITY_PAIR_CAP = 8
MULTICOLLINEARITY_PAIR_WEIGHT = 0.02
BASE_NON_LINEARITY_CONFIDENCE = 0.69
MAX_CORRELATION_CONFIDENCE_CAP = 0.5
NON_LINEARITY_CORRELATION_WEIGHT = 0.3
BASE_WEAK_FEATURE_CONFIDENCE = 0.65
WEAK_FEATURE_PERCENT_CAP = 90
WEAK_FEATURE_PERCENT_WEIGHT_DENOMINATOR = 300
MEANINGFUL_IMPROVEMENT_THRESHOLD = 0.05


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

    if redundant_pairs_count >= 5:
        multicollinearity = "critical"
    elif redundant_pairs_count >= 3:
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
    if "r2_score" in evidence:
        response["r2_score"] = float(evidence.get("r2_score", 0.0))
    if "best_improvement" in evidence:
        response["best_improvement"] = float(evidence.get("best_improvement", 0.0))
    if "redundant_pairs_count" in evidence:
        response["redundant_pairs_count"] = int(evidence.get("redundant_pairs_count", 0))
    response.update(_legacy_diagnosis(decision, evidence))
    return response


def decide_root_cause(evidence: Dict | None) -> Dict:
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
    nonlinear_gain = float(evidence.get("nonlinear_gain", 0.0))
    best_improvement = float(evidence.get("best_improvement", 0.0))
    best_model_r2 = evidence.get("best_model_r2")
    best_model_r2 = float(best_model_r2) if best_model_r2 is not None else -1.0

    if data_quality_score < 70 or missing_percentage >= 10:
        severity = "CRITICAL" if data_quality_score < 60 or missing_percentage >= 20 else "HIGH"
        return _make_response(
            "DATA_ISSUE",
            severity,
            BASE_DATA_ISSUE_CONFIDENCE + (DATA_ISSUE_CRITICAL_CONFIDENCE_BOOST if data_quality_score < 60 else 0.0),
            f"Data quality score {data_quality_score:.1f}/100; missing {missing_percentage:.1f}%",
            [f"R² currently {r2_score:.3f}"],
            evidence,
        )

    if r2_score < 0.10:
        improvement_note = (
            f"best improvement +{best_improvement * 100:.1f}% is negligible"
            if best_improvement < MEANINGFUL_IMPROVEMENT_THRESHOLD
            else f"best improvement +{best_improvement * 100:.1f}% still leaves very low fit"
        )
        return _make_response(
            "WEAK_SIGNAL",
            "CRITICAL",
            0.9 if best_improvement < MEANINGFUL_IMPROVEMENT_THRESHOLD else 0.82,
            f"Model fit is critically low (R²={r2_score:.3f}); {improvement_note}",
            [
                f"Strongest predictor correlation {strongest_correlation:.2f}",
                f"Redundant pairs observed: {redundant_pairs_count} (max {max_redundancy:.2f})",
            ],
            evidence,
        )

    if redundant_pairs_count >= 3:
        severity = "HIGH" if redundant_pairs_count <= 5 else "CRITICAL"
        return _make_response(
            "MULTICOLLINEARITY",
            severity,
            BASE_MULTICOLLINEARITY_CONFIDENCE + min(redundant_pairs_count, MULTICOLLINEARITY_PAIR_CAP) * MULTICOLLINEARITY_PAIR_WEIGHT,
            f"{redundant_pairs_count} feature pairs above redundancy threshold (max {max_redundancy:.2f})",
            [f"Weak feature percentage {weak_feature_pct}%", f"Model fit R²={r2_score:.3f}"],
            evidence,
        )

    if weak_feature_pct >= 80 and strongest_correlation < 0.12 and best_model_r2 < 0.20:
        return _make_response(
            "WEAK_FEATURES",
            "HIGH",
            BASE_WEAK_FEATURE_CONFIDENCE + min(weak_feature_pct, WEAK_FEATURE_PERCENT_CAP) / WEAK_FEATURE_PERCENT_WEIGHT_DENOMINATOR,
            f"{weak_feature_pct}% weak features; strongest correlation {strongest_correlation:.2f}",
            [f"Best model R²={best_model_r2:.3f}", f"Linear model R²={r2_score:.3f}"],
            evidence,
        )

    if poor_model_fit and nonlinear_gain >= 0.15 and best_model_r2 >= 0.25:
        return _make_response(
            "NON_LINEARITY",
            "HIGH" if r2_score < 0.2 else "MEDIUM",
            BASE_NON_LINEARITY_CONFIDENCE + min(nonlinear_gain, MAX_CORRELATION_CONFIDENCE_CAP) * NON_LINEARITY_CORRELATION_WEIGHT,
            f"Linear fit is weak (R²={r2_score:.3f}); non-linear gain={nonlinear_gain:.3f}",
            [f"Best non-linear model R²={best_model_r2:.3f}"],
            evidence,
        )

    if total_features < 3 and poor_model_fit and weak_feature_pct >= 85 and strongest_correlation < 0.10:
        return _make_response(
            "FEATURE_GAP",
            "HIGH",
            0.70,
            f"Only {total_features} features available with weak fit (R²={r2_score:.3f})",
            [f"Weak feature percentage {weak_feature_pct}%"],
            evidence,
        )

    if weak_feature_pct > 50 or strongest_correlation < 0.15:
        severity = "HIGH" if weak_feature_pct > 70 else "MEDIUM"
        return _make_response(
            "WEAK_FEATURES",
            severity,
            BASE_WEAK_FEATURE_CONFIDENCE + min(weak_feature_pct, WEAK_FEATURE_PERCENT_CAP) / WEAK_FEATURE_PERCENT_WEIGHT_DENOMINATOR,
            f"{weak_feature_pct}% weak features; strongest correlation {strongest_correlation:.2f}",
            [f"Model fit R²={r2_score:.3f}"],
            evidence,
        )

    return _make_response(
        "WEAK_FEATURES",
        "LOW",
        0.55,
        "No critical risk thresholds crossed",
        [f"R²={r2_score:.3f}", f"Weak feature percentage {weak_feature_pct}%"],
        evidence,
    )
