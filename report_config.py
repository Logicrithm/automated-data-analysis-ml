from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_report_config() -> Dict[str, Any]:
    """Return configurable defaults for professional report rendering."""
    return {
        "report_title": "Enterprise Data Analysis Report",
        "classification": "INTERNAL CONFIDENTIAL",
        "confidentiality_notice": "Confidential - Internal Use Only. Unauthorized distribution is prohibited.",
        "document_version": "v2.0",
        "organization": "Logicrithm Analytics",
        "report_period": "Current analysis period",
        "generated_timestamp_utc": _utc_now_iso(),
        "severity_colors": {
            "CRITICAL": "#B91C1C",
            "HIGH": "#DC2626",
            "MEDIUM": "#D97706",
            "LOW": "#15803D",
            "INFO": "#1D4ED8",
        },
        "financial": {
            "currency_symbol": "$",
            "monthly_decision_volume": 400,
            "average_decision_value": 1200.0,
            "implementation_base_cost": 18000.0,
            "implementation_cost_per_recommendation": 2500.0,
            "loss_sensitivity": 0.08,
            "quality_penalty_weight": 0.05,
        },
    }
