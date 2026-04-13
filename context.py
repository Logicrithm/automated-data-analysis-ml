from __future__ import annotations

from typing import Dict, List

import pandas as pd

DOMAIN_KEYWORDS = {
    "real_estate": {"price", "sqft", "bedroom", "bath", "zipcode", "neighborhood", "lot", "property"},
    "healthcare": {"patient", "diagnosis", "blood", "cholesterol", "heart", "treatment", "bmi"},
    "business": {"revenue", "profit", "sales", "cost", "margin", "customer", "churn"},
}


def infer_domain(signals: Dict, df: pd.DataFrame) -> Dict:
    column_text = " ".join(str(col).lower() for col in df.columns)
    evidence: List[str] = []
    scores = {domain: sum(1 for token in words if token in column_text) for domain, words in DOMAIN_KEYWORDS.items()}

    domain = max(scores, key=scores.get, default="generic")
    best_score = int(scores.get(domain, 0))
    if best_score <= 0:
        domain = "generic"

    if domain != "generic":
        evidence.append(f"{best_score} domain keyword matches in column names")

    if signals.get("target_type") == "regression" and domain == "real_estate":
        evidence.append("Regression target aligns with common real_estate pricing tasks")
    if signals.get("target_type") == "classification" and domain == "healthcare":
        evidence.append("Classification target aligns with diagnosis-style healthcare tasks")

    if domain == "generic":
        confidence = 0.35
        evidence.append("No strong domain keyword pattern detected")
    else:
        confidence = min(1.0, 0.25 + (best_score * 0.15) + (0.1 if len(evidence) > 1 else 0.0))

    return {
        "domain": domain,
        "confidence": round(float(confidence), 2),
        "reasoning": evidence,
    }
