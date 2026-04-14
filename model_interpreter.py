from __future__ import annotations


def interpret_models(ml_results: dict, diagnosis: dict) -> dict:
    """
    Simple model interpretation: What do results MEAN?

    Returns:
    {
        'interpretation': str,
        'is_data_limited': bool,
        'is_model_limited': bool,
    }
    """

    try:
        r2 = float(ml_results.get("r2_score", 0))
        feature_strength = diagnosis.get("feature_strength", "unknown")

        # RULE 1: If R² low and features weak → DATA problem
        if r2 < 0.3 and feature_strength == "weak":
            return {
                "interpretation": "Model is limited by weak features, not model choice. Add better predictors.",
                "is_data_limited": True,
                "is_model_limited": False,
            }

        # RULE 2: If R² low but features OK → MODEL problem
        if r2 < 0.3 and feature_strength in ["moderate", "strong"]:
            return {
                "interpretation": "Features are adequate but model cannot capture relationships. Try non-linear models.",
                "is_data_limited": False,
                "is_model_limited": True,
            }

        # RULE 3: If R² good → Model is working
        if r2 >= 0.5:
            return {
                "interpretation": "Model performs well. Focus on feature engineering for marginal gains.",
                "is_data_limited": False,
                "is_model_limited": False,
            }

        # DEFAULT
        return {
            "interpretation": "Model shows acceptable performance. Investigate specific failure patterns.",
            "is_data_limited": False,
            "is_model_limited": False,
        }

    except Exception:
        return {
            "interpretation": "Unable to interpret model results.",
            "is_data_limited": False,
            "is_model_limited": False,
        }
