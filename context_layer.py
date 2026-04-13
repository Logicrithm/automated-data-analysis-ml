from __future__ import annotations

from typing import Dict, List

import pandas as pd


DOMAIN_KEYWORDS = {
    "real_estate": {"price", "sqft", "bedroom", "bath", "zipcode", "neighborhood", "lot", "renovation"},
    "healthcare": {"age", "blood", "pressure", "diagnosis", "patient", "heart", "cholesterol"},
    "business": {"revenue", "profit", "cost", "margin", "expense", "sales", "customer"},
}

DOMAIN_CONTEXT = {
    "real_estate": "Housing price prediction dataset",
    "healthcare": "Healthcare outcomes analysis dataset",
    "business": "Business performance analytics dataset",
    "generic": "Generic tabular analytics dataset",
}

DOMAIN_LIKELY_FEATURES = {
    "real_estate": ["location", "condition", "age", "amenities"],
    "healthcare": ["lifestyle", "medical_history", "treatment", "biomarkers"],
    "business": ["marketing", "segment", "seasonality", "pricing_strategy"],
    "generic": ["domain_features", "interaction_terms", "external_signals", "time_features"],
}
CLASSIFICATION_SUFFIXES = ("class", "label", "diagnosis")


def infer_context(df: pd.DataFrame, rca_output: Dict) -> Dict:
    column_tokens = " ".join(col.lower() for col in df.columns)
    score_by_domain = {
        domain: sum(1 for keyword in keywords if keyword in column_tokens)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }

    domain = max(score_by_domain, key=score_by_domain.get) if score_by_domain else "generic"
    best_score = score_by_domain.get(domain, 0)
    if best_score == 0:
        domain = "generic"

    high_corr_count = len((rca_output or {}).get("high_vif_pairs") or [])
    if domain == "real_estate" and high_corr_count > 0:
        best_score += 1

    target_column = str((rca_output or {}).get("target_column", ""))
    target_series = df[target_column] if target_column in df.columns else pd.Series(dtype=float)
    target_lower = target_column.lower()
    if target_series.empty:
        problem_type = "classification" if target_lower.endswith(CLASSIFICATION_SUFFIXES) else "regression"
    else:
        unique_count = int(target_series.nunique(dropna=True))
        unique_ratio = unique_count / max(len(target_series), 1)
        is_discrete_numeric = pd.api.types.is_numeric_dtype(target_series) and (
            unique_count <= 20 or unique_ratio <= 0.05
        )
        problem_type = "classification" if (not pd.api.types.is_numeric_dtype(target_series) or is_discrete_numeric) else "regression"

    if best_score >= 4:
        confidence = "HIGH"
    elif best_score >= 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "domain": domain,
        "confidence": confidence,
        "context": DOMAIN_CONTEXT.get(domain, DOMAIN_CONTEXT["generic"]),
        "likely_features": DOMAIN_LIKELY_FEATURES.get(domain, DOMAIN_LIKELY_FEATURES["generic"]),
        "problem_type": problem_type,
        "scores": score_by_domain,
    }
