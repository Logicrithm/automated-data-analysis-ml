from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path
from typing import Dict, List


def _visual_block(name: str, path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return f'<div class="viz"><h4>{html.escape(name.replace("_", " ").title())}</h4><p>Visualization file not found.</p></div>'

    if file_path.suffix.lower() == ".html":
        return (
            f'<div class="viz"><h4>{html.escape(name.replace("_", " ").title())}</h4>'
            f"{file_path.read_text(encoding='utf-8')}</div>"
        )

    mime_type = mimetypes.guess_type(str(file_path))[0] or "image/png"
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return (
        f'<div class="viz"><h4>{html.escape(name.replace("_", " ").title())}</h4>'
        f'<img src="data:{mime_type};base64,{encoded}" alt="{html.escape(name)}"/></div>'
    )


def _priority_class(priority: str) -> str:
    normalized = (priority or "MEDIUM").lower()
    return normalized if normalized in {"critical", "high", "medium", "low"} else "medium"


def _render_recommendations(items: List[Dict]) -> str:
    if not items:
        return "<p>No recommendations generated.</p>"
    rows = []
    for item in items:
        priority = str(item.get("priority", "MEDIUM")).upper()
        rows.append(
            "<div class='recommendation {css}'><strong>{priority}</strong>: {action} "
            "<span class='meta'>(Impact: {impact}, Effort: {effort})</span></div>".format(
                css=_priority_class(priority),
                priority=html.escape(priority),
                action=html.escape(str(item.get("action", ""))),
                impact=html.escape(str(item.get("impact", ""))),
                effort=html.escape(str(item.get("effort", ""))),
            )
        )
    return "".join(rows)


def build_html_report(results: Dict, visuals: Dict[str, str]) -> str:
    overview = results.get("overview", {})
    context = results.get("context", {})
    diagnosis = results.get("diagnosis", {})
    verdict = results.get("verdict", {})
    confidence = results.get("confidence", {})
    recommendations = (results.get("recommendations") or {}).get("recommendations") or []
    model_results = results.get("ml_results", {})

    primary_issue = str(verdict.get("primary_issue", "unknown"))
    warning_banner = ""
    if primary_issue in {"data_quality", "feature_gap", "multicollinearity"}:
        warning_banner = (
            "<div class='banner'><strong>Warning:</strong> Primary issue detected - {issue}.</div>".format(
                issue=html.escape(primary_issue.replace("_", " ").title())
            )
        )
    warning_section_html = warning_banner if warning_banner else "<p>No critical warning flags.</p>"

    visual_html = "".join(_visual_block(name, path) for name, path in visuals.items())

    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Automated Data Analysis Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .section {{ margin-bottom: 24px; }}
    .banner {{ background: #fee2e2; color: #991b1b; border: 1px solid #dc2626; border-radius: 8px; padding: 10px 12px; }}
    .executive {{ background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; }}
    .diag-grid {{ display: grid; grid-template-columns: repeat(2, minmax(180px, 1fr)); gap: 8px; }}
    .diag-item {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px; }}
    .recommendation {{ padding: 8px 10px; border-radius: 6px; margin: 6px 0; border-left: 4px solid; }}
    .critical {{ background: #fee2e2; border-color: #dc2626; }}
    .high {{ background: #fef3c7; border-color: #f59e0b; }}
    .medium {{ background: #dbeafe; border-color: #3b82f6; }}
    .low {{ background: #ecfeff; border-color: #06b6d4; }}
    .meta {{ color: #374151; font-size: 0.9em; }}
    .viz {{ margin-bottom: 16px; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Automated Data Analysis Report</h1>

  <div class="section">{warning_section_html}</div>

  <div class="section executive">
    <h2>Executive Summary</h2>
    <p><strong>Primary Issue:</strong> {html.escape(primary_issue.replace('_', ' ').title())}</p>
    <p><strong>Verdict Confidence:</strong> {float(verdict.get('confidence', 0.0)):.2f}</p>
    <p><strong>Overall Confidence:</strong> {float(confidence.get('overall', 0.0)):.2f}</p>
    <p><strong>Model R²:</strong> {html.escape(str(model_results.get('r2_score', 'N/A')))}</p>
    <p><strong>Target:</strong> {html.escape(str(model_results.get('target_column', 'None')))}</p>
  </div>

  <div class="section">
    <h2>Context</h2>
    <p><strong>Domain:</strong> {html.escape(str(context.get('domain', 'generic')))}</p>
    <p><strong>Domain Confidence:</strong> {float(context.get('confidence', 0.0)):.2f}</p>
    <p><strong>Evidence:</strong> {html.escape(', '.join(context.get('reasoning', []) or ['No evidence recorded']))}</p>
  </div>

  <div class="section">
    <h2>Diagnosis</h2>
    <div class="diag-grid">
      <div class="diag-item"><strong>Model Performance:</strong> {html.escape(str(diagnosis.get('model_perf', 'unknown')))}</div>
      <div class="diag-item"><strong>Feature Strength:</strong> {html.escape(str(diagnosis.get('feature_strength', 'unknown')))}</div>
      <div class="diag-item"><strong>Multicollinearity:</strong> {html.escape(str(diagnosis.get('multicollinearity', 'unknown')))}</div>
      <div class="diag-item"><strong>Data Quality:</strong> {html.escape(str(diagnosis.get('data_quality', 'unknown')))}</div>
    </div>
  </div>

  <div class="section">
    <h2>Recommendations</h2>
    {_render_recommendations(recommendations)}
  </div>

  <div class="section">
    <h2>Visualizations</h2>
    {visual_html or '<p>No visualizations were generated for this dataset.</p>'}
  </div>

  <div class="section">
    <h2>Dataset Summary</h2>
    <p>Rows: {overview.get('rows', 0):,} | Columns: {overview.get('columns', 0)}</p>
  </div>
</body>
</html>
"""
