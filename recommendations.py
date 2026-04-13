from __future__ import annotations

from typing import Dict, List, Tuple

# Rule tuple format: (priority, action, impact, effort)

RULES: Dict[str, Dict[str, List[Tuple[str, str, str, str]]]] = {
    "real_estate": {
        "feature_gap": [
            ("CRITICAL", "Add location/neighborhood features", "HIGH", "HIGH"),
            ("HIGH", "Add property condition and age", "HIGH", "MEDIUM"),
        ],
        "multicollinearity": [
            ("CRITICAL", "Remove redundant square-footage features", "MEDIUM", "LOW"),
        ],
        "model_mismatch": [
            ("HIGH", "Benchmark tree-based regressors", "HIGH", "LOW"),
        ],
        "data_quality": [
            ("CRITICAL", "Fix missing and inconsistent property records", "HIGH", "MEDIUM"),
        ],
        "unknown": [("MEDIUM", "Continue feature engineering iteration", "MEDIUM", "MEDIUM")],
    },
    "healthcare": {
        "feature_gap": [
            ("CRITICAL", "Add clinical history and treatment variables", "HIGH", "HIGH"),
            ("HIGH", "Add vitals and biomarker trends", "HIGH", "MEDIUM"),
        ],
        "multicollinearity": [("HIGH", "Consolidate highly correlated lab measures", "MEDIUM", "MEDIUM")],
        "model_mismatch": [("HIGH", "Evaluate non-linear and calibrated models", "HIGH", "MEDIUM")],
        "data_quality": [("CRITICAL", "Resolve missing patient and diagnosis records", "HIGH", "HIGH")],
        "unknown": [("MEDIUM", "Collect additional domain features", "MEDIUM", "MEDIUM")],
    },
    "business": {
        "feature_gap": [
            ("CRITICAL", "Add customer segment and campaign variables", "HIGH", "MEDIUM"),
            ("HIGH", "Add seasonality and promotion signals", "HIGH", "MEDIUM"),
        ],
        "multicollinearity": [("HIGH", "Drop duplicated revenue/cost aggregates", "MEDIUM", "LOW")],
        "model_mismatch": [("HIGH", "Try gradient boosting with interactions", "HIGH", "LOW")],
        "data_quality": [("CRITICAL", "Clean transaction gaps and duplicates", "HIGH", "MEDIUM")],
        "unknown": [("MEDIUM", "Continue targeted feature engineering", "MEDIUM", "MEDIUM")],
    },
    "generic": {
        "feature_gap": [
            ("CRITICAL", "Add domain-specific predictors", "HIGH", "HIGH"),
            ("HIGH", "Create interaction features", "MEDIUM", "LOW"),
        ],
        "multicollinearity": [("HIGH", "Remove highly correlated columns", "MEDIUM", "LOW")],
        "model_mismatch": [("HIGH", "Benchmark non-linear models", "MEDIUM", "LOW")],
        "data_quality": [("CRITICAL", "Improve missingness and consistency", "HIGH", "MEDIUM")],
        "unknown": [("MEDIUM", "Perform exploratory feature analysis", "MEDIUM", "MEDIUM")],
    },
}


def recommend(domain: str, diagnosis: Dict, verdict: Dict | None = None) -> Dict:
    selected_domain = domain if domain in RULES else "generic"
    primary_issue = (verdict or {}).get("primary_issue")
    if not primary_issue:
        if diagnosis.get("data_quality") in {"poor", "fair"}:
            primary_issue = "data_quality"
        elif diagnosis.get("feature_strength") == "weak":
            primary_issue = "feature_gap"
        elif diagnosis.get("multicollinearity") in {"high", "critical"}:
            primary_issue = "multicollinearity"
        elif diagnosis.get("model_perf") == "critical":
            primary_issue = "model_mismatch"
        else:
            primary_issue = "unknown"

    items = (
        RULES[selected_domain].get(primary_issue)
        or RULES[selected_domain].get("unknown")
        or RULES["generic"]["unknown"]
    )
    recommendations = [
        {
            "priority": priority,
            "action": action,
            "impact": impact,
            "effort": effort,
        }
        for priority, action, impact, effort in items
    ]
    return {"primary_issue": primary_issue, "recommendations": recommendations}
