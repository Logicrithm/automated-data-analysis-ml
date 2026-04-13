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


def _actions_html(actions: List[Dict]) -> str:
    if not actions:
        return "<div class='recommendations'><div class='action medium'>No recommended actions.</div></div>"
    allowed_priorities = {"critical", "high", "medium", "low"}
    rows = []
    for action in actions:
        raw_priority = str(action.get("priority", "MEDIUM")).lower()
        priority = raw_priority if raw_priority in allowed_priorities else "medium"
        safe_css_class = html.escape(priority, quote=True)
        rows.append(
            "<div class='action {css_class}'>Action ({label}): {action_name}</div>".format(
                css_class=safe_css_class,
                label=html.escape(str(action.get("priority", "MEDIUM"))),
                action_name=html.escape(str(action.get("action", ""))),
            )
        )
    return f"<div class='recommendations'>{''.join(rows)}</div>"


def build_html_report(results: Dict, visuals: Dict[str, str]) -> str:
    insights: List[Dict] = results.get("insights", [])
    overview = results.get("overview", {})
    ml_results = results.get("ml_results", {})
    confidence = results.get("confidence", {})
    context = results.get("context", {})
    data_quality = (results.get("data_quality") or {}).get("data_quality", {})
    quality_grade = (results.get("data_quality") or {}).get("grade", "N/A")
    model_comparison = (results.get("model_comparison") or {}).get("models") or []
    best_model = (results.get("model_comparison") or {}).get("best_model")

    visual_html = "".join(_visual_block(name, path) for name, path in visuals.items())
    severity_class = {"CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium", "LOW": "low"}

    insight_cards = []
    for insight in insights:
        severity = str(insight.get("severity", "MEDIUM")).upper()
        insight_conf = insight.get("confidence") or confidence
        reliability = float((insight_conf or {}).get("reliability", 0.0))
        content = html.escape(str(insight.get("content", "")))
        root_cause = html.escape(str(insight.get("root_cause", "")))
        actions_html = _actions_html(insight.get("actions") or [])
        insight_cards.append(
            f"""
            <div class="insight {severity_class.get(severity, 'medium')}">
              <span class="severity">{"🚨 " if severity == "CRITICAL" else ""}{severity}</span>
              <span class="confidence">{(insight_conf or {}).get("finding", 0):.2f}</span>
              <span class="reliability">{reliability:.0%}</span>
              <p>{content}</p>
              <p><strong>Root cause:</strong> {root_cause}</p>
              {actions_html}
            </div>
            """
        )

    model_rows = []
    for model in model_comparison:
        model_rows.append(
            "<tr><td>{name}</td><td>{r2}</td><td>{rmse}</td><td>{time}s</td><td>{rec}</td></tr>".format(
                name=html.escape(str(model.get("name", ""))),
                r2=html.escape(str(model.get("r2_score", ""))),
                rmse=html.escape(str(model.get("rmse", ""))),
                time=html.escape(str(model.get("training_time", ""))),
                rec=html.escape(str(model.get("recommendation", ""))),
            )
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
    .insight {{ padding: 12px 14px; border-radius: 8px; margin: 10px 0; border-left: 4px solid; }}
    .critical {{ background: #fee2e2; color: #991b1b; border-color: #dc2626; }}
    .high {{ background: #fef3c7; color: #92400e; border-color: #f59e0b; }}
    .medium {{ background: #dbeafe; color: #1e40af; border-color: #3b82f6; }}
    .low {{ background: #ecfeff; color: #155e75; border-color: #06b6d4; }}
    .severity, .confidence, .reliability {{ display: inline-block; margin-right: 8px; font-weight: 700; }}
    .recommendations {{ margin-top: 8px; }}
    .action {{ padding: 6px 8px; border-radius: 6px; margin: 4px 0; }}
    .action.critical {{ background: #fecaca; }}
    .action.high {{ background: #fde68a; }}
    .action.medium {{ background: #bfdbfe; }}
    .model-comparison {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    .model-comparison th, .model-comparison td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    .quality-scorecard {{ padding: 12px; border-radius: 8px; background: #f8fafc; border: 1px solid #cbd5e1; }}
    .viz {{ margin-bottom: 16px; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Automated Data Analysis Report</h1>
  <div class="section">
    <h2>Context</h2>
    <p>Domain: <strong>{html.escape(str(context.get('domain', 'generic')))}</strong> ({html.escape(str(context.get('confidence', 'LOW')))} confidence)</p>
    <p>{html.escape(str(context.get('context', 'Generic dataset')))}</p>
  </div>

  <div class="section">
    <h2>Key Insights (Ranked by Severity)</h2>
    {''.join(insight_cards) or '<p>No insights were generated.</p>'}
  </div>

  <div class="section">
    <h2>Model Comparison</h2>
    <table class="model-comparison">
      <tr><th>Model</th><th>R² Score</th><th>RMSE</th><th>Speed</th><th>Recommendation</th></tr>
      {''.join(model_rows) or '<tr><td colspan="5">No model comparison available.</td></tr>'}
    </table>
    <p><strong>Best model:</strong> {html.escape(str(best_model or 'N/A'))}</p>
  </div>

  <div class="section">
    <h2>Data Quality</h2>
    <div class="quality-scorecard">
      Overall Score: {data_quality.get('overall_score', 0):.1f}/100 (Grade: {quality_grade})
      <br/>Completeness: {data_quality.get('completeness', 0)} | Uniqueness: {data_quality.get('uniqueness', 0)} |
      Consistency: {data_quality.get('consistency', 0)} | Plausibility: {data_quality.get('plausibility', 0)} |
      Feature Richness: {data_quality.get('feature_richness', 0)}
    </div>
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
    <p>Overall confidence: {(confidence or {}).get('overall', 0):.2f}</p>
  </div>
</body>
</html>
"""
