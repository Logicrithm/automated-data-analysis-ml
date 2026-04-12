from __future__ import annotations

import html
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


def build_html_report(results: Dict, visuals: Dict[str, str]) -> str:
    insights: List[str] = results.get("insights", [])
    overview = results.get("overview", {})
    ml_results = results.get("ml_results", {})

    visual_html = "".join(
        f'<div class="viz"><h4>{html.escape(name.replace("_", " ").title())}</h4><img src="{html.escape(Path(path).name)}" alt="{html.escape(name)}"/></div>'
        for name, path in visuals.items()
    )

    insight_items = "".join(f"<li>{html.escape(insight)}</li>" for insight in insights)

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
    .warning-level {{ background: #fef3c7; color: #92400e; border: 1px solid #f59e0b; }}
    .ok {{ background: #dcfce7; color: #166534; border: 1px solid #22c55e; }}
    .viz {{ margin-bottom: 16px; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; }}
    ul {{ padding-left: 22px; }}
    li {{ margin-bottom: 8px; }}
  </style>
</head>
<body>
  <h1>Automated Data Analysis Report</h1>

  <div class="section">
    <h2>Key Insights (Ranked)</h2>
    {_warning_block(ml_results)}
    <ul>{insight_items}</ul>
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
