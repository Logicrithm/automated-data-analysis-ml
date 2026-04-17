from __future__ import annotations

from typing import Any, Dict

BASELINE_QUALITY_THRESHOLD = 70.0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def calculate_financial_impact(results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate ROI, cost-benefit and break-even timeline from analysis evidence."""
    evidence = results.get("evidence", {}) or {}
    diagnosis = results.get("diagnosis", {}) or {}
    recommendations = results.get("recommendations", []) or []
    financial_cfg = (config or {}).get("financial", {})

    r2 = _to_float(evidence.get("r2_score"), 0.0)
    data_quality_score = _to_float(evidence.get("data_quality_score"), 0.0)
    weak_feature_pct = max(0.0, _to_float(evidence.get("weak_feature_pct"), 0.0))

    monthly_decisions = max(1, _to_int(financial_cfg.get("monthly_decision_volume"), 400))
    avg_decision_value = max(1.0, _to_float(financial_cfg.get("average_decision_value"), 1200.0))
    implementation_base_cost = max(0.0, _to_float(financial_cfg.get("implementation_base_cost"), 18000.0))
    per_recommendation_cost = max(0.0, _to_float(financial_cfg.get("implementation_cost_per_recommendation"), 2500.0))
    loss_sensitivity = max(0.0, _to_float(financial_cfg.get("loss_sensitivity"), 0.08))
    quality_penalty_weight = max(0.0, _to_float(financial_cfg.get("quality_penalty_weight"), 0.05))

    reliability_gap = max(0.0, 1.0 - r2)
    quality_gap = max(0.0, (BASELINE_QUALITY_THRESHOLD - data_quality_score) / BASELINE_QUALITY_THRESHOLD)
    weak_feature_factor = 1.0 + (weak_feature_pct / 100.0)

    annual_value_at_stake = monthly_decisions * avg_decision_value * 12.0
    loss_ratio = (loss_sensitivity * reliability_gap) + (quality_penalty_weight * quality_gap)
    estimated_avoidable_loss = max(0.0, annual_value_at_stake * loss_ratio * weak_feature_factor)

    implementation_cost = implementation_base_cost + (len(recommendations) * per_recommendation_cost)
    net_benefit = estimated_avoidable_loss - implementation_cost
    roi_percent = (net_benefit / implementation_cost * 100.0) if implementation_cost > 0 else 0.0

    monthly_benefit = estimated_avoidable_loss / 12.0
    break_even_months = (implementation_cost / monthly_benefit) if monthly_benefit > 0 else None

    decision_name = str(diagnosis.get("decision", "UNKNOWN"))
    if break_even_months is not None:
        break_even_text = f"{break_even_months:.1f}"
        narrative = (
            f"For {decision_name}, improving data and feature reliability is estimated to protect "
            f"${estimated_avoidable_loss:,.0f} annually, with projected ROI of {roi_percent:.1f}% "
            f"and break-even in {break_even_text} months."
        )
    else:
        narrative = (
            f"For {decision_name}, projected annual benefit is ${estimated_avoidable_loss:,.0f}; "
            "break-even cannot be estimated."
        )

    return {
        "annual_value_at_stake": round(annual_value_at_stake, 2),
        "estimated_avoidable_loss": round(estimated_avoidable_loss, 2),
        "implementation_cost": round(implementation_cost, 2),
        "net_benefit": round(net_benefit, 2),
        "roi_percent": round(roi_percent, 2),
        "break_even_months": round(break_even_months, 2) if break_even_months is not None else None,
        "currency_symbol": financial_cfg.get("currency_symbol", "$"),
        "narrative": narrative,
        "assumptions": {
            "monthly_decision_volume": monthly_decisions,
            "average_decision_value": avg_decision_value,
            "loss_sensitivity": loss_sensitivity,
            "quality_penalty_weight": quality_penalty_weight,
        },
    }
