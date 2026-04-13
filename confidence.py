from __future__ import annotations

from typing import Dict


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def calculate_confidence(signals: Dict, diagnosis: Dict, verdict: Dict) -> Dict:
    quality = _clamp(float(signals.get("data_quality_score", 0.0)) / 100.0)
    feature_strength = diagnosis.get("feature_strength")
    feature_score = {"weak": 0.35, "moderate": 0.65, "strong": 0.85}.get(feature_strength, 0.5)
    verdict_conf = _clamp(float(verdict.get("confidence", 0.5)))

    overall = _clamp((0.4 * quality) + (0.35 * feature_score) + (0.25 * verdict_conf))
    return {
        "data": round(quality, 2),
        "diagnosis": round(feature_score, 2),
        "verdict": round(verdict_conf, 2),
        "overall": round(overall, 2),
    }
