from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path
from typing import Dict, List


def _warning_block(ml_results: Dict) -> str:
    if not ml_results:
        return ""

    severity = ml_results.get("severity", "ok")
    classes = {
        "critical": "warning critical",
        "warning": "warning warning-level",
        "ok": "warning ok",
    }
    css_class = classes.get(severity, "warning")
    message = ml_results.get("interpretation", "")
    return f'<div class="{css_class}">{message}</div>'


def _visual_block(name: str, path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return f'<div class="viz"><h4>{html.escape(name.replace("_", " ").title())}</h4><p>Visualization file not found.</p></div>'

    if file_path.suffix.lower() == ".html":
        return (
            f'<div class="viz"><h4>{html.escape(name.replace("_", " ").title())}</h4>'
            f'{file_path.read_text(encoding="utf-8")}</div>'
        )

    mime_type = mimetypes.guess_type(str(file_path))[0] or "image/png"
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return (
        f'<div class="viz"><h4>{html.escape(name.replace("_", " ").title())}</h4>'
        f'<img src="data:{mime_type};base64,{encoded}" alt="{html.escape(name)}"/></div>'
    )


def build_html_report(results: Dict, visuals: Dict[str, str]) -> str:
    insights: List[Dict] = results.get("insights", [])
    overview = results.get("overview", {})
    ml_results = results.get("ml_results", {})
    multicollinearity = results.get("multicollinearity", {})

    visual_html = "".join(_visual_block(name, path) for name, path in visuals.items())
    severity_class = {
        "CRITICAL": "critical",
        "HIGH": "warning-level",
        "MEDIUM": "ok",
        "LOW": "info",
    }
    insight_cards = []
    for insight in insights:
        severity = str(insight.get("severity", "MEDIUM")).upper()
        confidence = str(insight.get("confidence", "MEDIUM")).upper()
        category = str(insight.get("category", "GENERAL")).upper()
        actionable = "Yes" if bool(insight.get("actionable", False)) else "No"
        content = html.escape(str(insight.get("content", "")))
        insight_cards.append(
            f"""
            <div class="insight-card {severity_class.get(severity, 'info')}">
              <div class="badges">
                <span class="badge severity">{severity}</span>
                <span class="badge confidence">Confidence: {confidence}</span>
                <span class="badge category">{category}</span>
                <span class="badge action">Actionable: {actionable}</span>
              </div>
              <p>{content}</p>
            </div>
            """
        )
    high_vif_pairs = multicollinearity.get("high_vif_pairs") or []
    multicollinearity_banner = ""
    if high_vif_pairs:
        pair = high_vif_pairs[0]
        multicollinearity_banner = (
            '<div class="warning critical">'
            f"⚠️ Multicollinearity Warning: {html.escape(pair['feature_a'])} and {html.escape(pair['feature_b'])} "
            f"have correlation {pair['correlation']:.2f}. Review coefficient interpretation carefully."
            "</div>"
        )

    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Automated Data Analysis Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .section {{ margin-bottom: 28px; }}
    .warning {{ padding: 12px 14px; border-radius: 8px; font-weight: 700; margin: 10px 0; }}
    .critical {{ background: #fee2e2; color: #991b1b; border: 1px solid #ef4444; }}
    .warning-level {{ background: #fff7ed; color: #9a3412; border: 1px solid #fb923c; }}
    .ok {{ background: #fef9c3; color: #854d0e; border: 1px solid #eab308; }}
    .info {{ background: #eff6ff; color: #1e3a8a; border: 1px solid #60a5fa; }}
    .insight-card {{ border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
    .badge {{ border-radius: 999px; padding: 4px 10px; font-size: 12px; font-weight: 700; background: rgba(255, 255, 255, 0.75); }}
    .viz {{ margin-bottom: 16px; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Automated Data Analysis Report</h1>

  <div class="section">
    <h2>Key Insights (Ranked by Severity)</h2>
    {_warning_block(ml_results)}
    {multicollinearity_banner}
    {''.join(insight_cards) or '<p>No insights were generated.</p>'}
  </div>

  <div class="section">
    <h2>Visualizations</h2>
    {visual_html or '<p>No visualizations were generated for this dataset.</p>'}
  </div>

  <div class="section">
    <h2>Dataset Summary</h2>
    <p>Rows: {overview.get('rows', 0):,} | Columns: {overview.get('columns', 0)}</p>
    <p>Target Column: {ml_results.get('target_column', 'None')}</p>
    <p>R² Score: {ml_results.get('r2_score', 'N/A')}</p>
  </div>
</body>
</html>
"""
