from __future__ import annotations

import base64
import html
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from financial_impact import calculate_financial_impact
from report_config import get_report_config
from risk_assessment import generate_risk_assessment


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _weak_feature_display(value: Any) -> str:
    weak_pct = int(max(0.0, _to_float(value)))
    if weak_pct <= 100:
        return f"{weak_pct}%"
    return "High proportion of features shows low correlation with target (value capped for display purposes)"


def _visual_block(name: str, path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return f'<div class="viz-card"><h4>{html.escape(name.replace("_", " ").title())}</h4><p>Visualization not found.</p></div>'

    if file_path.suffix.lower() == ".html":
        try:
            content = file_path.read_text(encoding="utf-8")
            return f'<div class="viz-card"><h4>{html.escape(name.replace("_", " ").title())}</h4>{content}</div>'
        except OSError:
            return f'<div class="viz-card"><h4>{html.escape(name)}</h4><p>Could not read HTML visualization.</p></div>'

    try:
        mime_type = mimetypes.guess_type(str(file_path))[0] or "image/png"
        encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        return (
            f'<div class="viz-card"><h4>{html.escape(name.replace("_", " ").title())}</h4>'
            f'<img src="data:{mime_type};base64,{encoded}" alt="{html.escape(name)}" /></div>'
        )
    except OSError:
        return f'<div class="viz-card"><h4>{html.escape(name)}</h4><p>Could not read visualization file.</p></div>'


def _severity_badge(severity: str, colors: Dict[str, str]) -> str:
    sev = _safe_text(severity, "INFO").upper()
    color = colors.get(sev, colors.get("INFO", "#1D4ED8"))
    return f'<span class="severity-badge" style="background:{html.escape(color)}">{html.escape(sev)}</span>'


def _statistical_rows(stats: Iterable[Dict[str, Any]]) -> str:
    rows = []
    for item in stats:
        rows.append(
            "<tr>"
            f"<td>{html.escape(_safe_text(item.get('feature')))}</td>"
            f"<td>{_to_float(item.get('mean')):.4f}</td>"
            f"<td>{_to_float(item.get('median')):.4f}</td>"
            f"<td>{_to_float(item.get('std_dev')):.4f}</td>"
            f"<td>{_to_float(item.get('q1')):.4f}</td>"
            f"<td>{_to_float(item.get('q3')):.4f}</td>"
            "</tr>"
        )
    return "".join(rows) if rows else "<tr><td colspan='6'>No numeric statistical summary available.</td></tr>"


def _risk_matrix_html(risk: Dict[str, Any], colors: Dict[str, str]) -> str:
    matrix = risk.get("matrix") or [[0, 0, 0, 0] for _ in range(4)]
    band = _safe_text(risk.get("overall_risk_band"), "MEDIUM").upper()

    def cell_class(impact: int, likelihood: int) -> str:
        score = impact * likelihood
        if score >= 12:
            return "critical"
        if score >= 8:
            return "high"
        if score >= 4:
            return "medium"
        return "low"

    rows = []
    for impact in range(4, 0, -1):
        cells = []
        for likelihood in range(1, 5):
            count = int(matrix[impact - 1][likelihood - 1])
            cells.append(
                f"<td class='risk-cell {cell_class(impact, likelihood)}'>"
                f"<div class='risk-score'>{impact * likelihood}</div>"
                f"<div class='risk-count'>{count} risk(s)</div>"
                "</td>"
            )
        rows.append(f"<tr><th>Impact {impact}</th>{''.join(cells)}</tr>")

    highest = risk.get("highest_risk", {}) or {}
    risk_list = "".join(
        f"<li><strong>{html.escape(_safe_text(item.get('name')))}:</strong> "
        f"L{int(item.get('likelihood', 1))} × I{int(item.get('impact', 1))} - "
        f"{html.escape(_safe_text(item.get('description')))}</li>"
        for item in (risk.get("risks") or [])
    )

    return f"""
    <div class="card">
      <p><strong>Overall Risk Band:</strong> {_severity_badge(band, colors)}</p>
      <p><strong>Highest Risk:</strong> {html.escape(_safe_text(highest.get('name')))} (L{int(highest.get('likelihood', 1))} × I{int(highest.get('impact', 1))})</p>
      <table class="risk-matrix-table">
        <thead><tr><th></th><th>Likelihood 1</th><th>Likelihood 2</th><th>Likelihood 3</th><th>Likelihood 4</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <ul class="compact-list">{risk_list}</ul>
    </div>
    """


def _toc_entries() -> List[Tuple[str, str]]:
    return [
        ("section-executive", "1. Executive Summary"),
        ("section-toc", "2. Table of Contents"),
        ("section-risk", "3. Risk Assessment Matrix"),
        ("section-financial", "4. Financial Impact Analysis"),
        ("section-findings", "5. Detailed Findings"),
        ("section-recommendations", "6. Recommendations"),
        ("section-visualizations", "7. Visualizations"),
        ("section-appendix", "8. Appendix"),
    ]


def build_professional_html_report(results: Dict[str, Any], visuals: Dict[str, str], config: Dict[str, Any] | None = None) -> str:
    report_cfg = get_report_config()
    if config:
        report_cfg.update(config)

    severity_colors = report_cfg.get("severity_colors", {})
    now_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    diagnosis = results.get("diagnosis", {}) or {}
    evidence = results.get("evidence", {}) or {}
    recommendations = results.get("recommendations", []) or []
    context = results.get("context", {}) or {}
    overview = results.get("overview", {}) or {}
    data_quality = (results.get("data_quality") or {}).get("data_quality", {})
    quality_grade = _safe_text((results.get("data_quality") or {}).get("grade"), "N/A")
    ml_results = results.get("ml_results", {}) or {}
    final_output = _safe_text(results.get("final_output"), "Business-facing final summary unavailable.")
    business_impact = _safe_text(results.get("business_impact"), "Business impact summary unavailable.")
    statistical_summary = results.get("statistical_summary", []) or []

    risk = generate_risk_assessment(results)
    financial = calculate_financial_impact(results, report_cfg)

    decision_name = _safe_text(diagnosis.get("decision"), "UNKNOWN")
    severity = _safe_text(diagnosis.get("severity"), "MEDIUM").upper()
    dominant_signal = _safe_text(diagnosis.get("dominant_signal"), "No dominant signal captured.")
    model_r2 = _to_float(evidence.get("r2_score"), _to_float(ml_results.get("r2_score"), 0.0))

    toc_html = "".join(
        f"<li><a href='#{anchor}'>{html.escape(label)}</a><span class='toc-page'>auto</span></li>"
        for anchor, label in _toc_entries()
    )

    recommendation_html = "".join(
        f"<div class='recommendation-card'>"
        f"{_severity_badge(_safe_text(rec.get('priority'), 'MEDIUM').upper(), severity_colors)}"
        f"<h4>{html.escape(_safe_text(rec.get('action')))}</h4>"
        f"<p>{html.escape(_safe_text(rec.get('reason')))}</p>"
        f"<p><strong>Impact:</strong> {html.escape(_safe_text(rec.get('impact'), 'UNKNOWN'))} | "
        f"<strong>Effort:</strong> {html.escape(_safe_text(rec.get('effort'), 'UNKNOWN'))}</p>"
        "</div>"
        for rec in recommendations
    ) or "<p>No recommendations available.</p>"

    visual_html = "".join(_visual_block(name, path) for name, path in visuals.items()) or "<p>No visualizations available.</p>"

    quality_issue_rows = "".join(
        f"<tr><td>{html.escape(_safe_text(issue.get('column')))}</td><td>{html.escape(_safe_text(issue.get('issue')))}</td><td>{int(issue.get('count', 0))}</td></tr>"
        for issue in (results.get("quality_issues") or [])
    ) or "<tr><td colspan='3'>No explicit data quality issues detected.</td></tr>"

    data_dictionary_rows = "".join(
        f"<tr><td>{html.escape(_safe_text(col))}</td><td>{html.escape(_safe_text(dtype))}</td></tr>"
        for col, dtype in (overview.get("column_types") or {}).items()
    ) or "<tr><td colspan='2'>Column metadata unavailable.</td></tr>"

    footer_text = (
        f"{_safe_text(report_cfg.get('classification'))} | "
        f"{_safe_text(report_cfg.get('document_version'))} | "
        f"Generated UTC: {html.escape(_safe_text(report_cfg.get('generated_timestamp_utc'), now_utc))}"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{html.escape(_safe_text(report_cfg.get('report_title')))}</title>
<style>
:root {{ --critical:{severity_colors.get('CRITICAL', '#B91C1C')}; --high:{severity_colors.get('HIGH', '#DC2626')}; --medium:{severity_colors.get('MEDIUM', '#D97706')}; --low:{severity_colors.get('LOW', '#15803D')}; --info:{severity_colors.get('INFO', '#1D4ED8')}; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:'Segoe UI', Roboto, Arial, sans-serif; color:#111827; background:#f3f4f6; line-height:1.5; }}
.page {{ max-width:1100px; margin:0 auto; background:white; padding:24px 24px 70px; }}
.cover {{ min-height:88vh; display:flex; flex-direction:column; justify-content:center; border:2px solid #e5e7eb; padding:40px; }}
.cover h1 {{ margin:0 0 16px; font-size:2.1rem; }}
.cover .meta {{ font-size:1rem; margin:4px 0; }}
.section {{ margin:20px 0; padding:18px; border:1px solid #e5e7eb; border-radius:10px; background:#fff; }}
.section-title {{ margin:0 0 10px; font-size:1.2rem; color:#0f172a; border-bottom:2px solid #e5e7eb; padding-bottom:6px; }}
.card {{ background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:12px; }}
.grid-2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; }}
.severity-badge {{ color:#fff; padding:3px 9px; border-radius:999px; font-size:0.75rem; font-weight:700; display:inline-block; }}
.toc li {{ display:flex; justify-content:space-between; border-bottom:1px dotted #cbd5e1; padding:6px 0; }}
.toc a {{ color:#1d4ed8; text-decoration:none; }}
.toc-page {{ color:#6b7280; font-size:0.9rem; }}
.risk-matrix-table, table {{ width:100%; border-collapse:collapse; }}
th, td {{ border:1px solid #d1d5db; padding:8px; text-align:left; vertical-align:top; }}
.risk-cell.low {{ background:#dcfce7; }}
.risk-cell.medium {{ background:#fef3c7; }}
.risk-cell.high {{ background:#fee2e2; }}
.risk-cell.critical {{ background:#fecaca; }}
.risk-score {{ font-weight:700; }}
.risk-count {{ font-size:0.8rem; color:#334155; }}
.compact-list {{ margin:10px 0 0 18px; }}
.recommendation-card {{ border-left:4px solid #d1d5db; background:#f8fafc; padding:10px; margin:10px 0; border-radius:6px; }}
.viz-card {{ margin:12px 0; padding:10px; border:1px solid #e5e7eb; border-radius:8px; background:#fff; }}
.viz-card img {{ max-width:100%; border:1px solid #d1d5db; border-radius:4px; }}
.page-break {{ break-before:page; page-break-before:always; }}
.page-footer {{ position:fixed; left:0; right:0; bottom:0; background:#0f172a; color:#f8fafc; padding:8px 16px; font-size:11px; display:flex; justify-content:space-between; }}
.page-number::after {{ content:"Page " counter(page); }}

@media print {{
  body {{ background:#fff; }}
  .page {{ max-width:100%; padding:0 0 55px; }}
  .section {{ break-inside:avoid; page-break-inside:avoid; }}
  .page-footer {{ position:fixed; bottom:0; }}
}}
@media (max-width:768px) {{
  .page {{ padding:12px 12px 75px; }}
}}
</style>
</head>
<body>
<div class="page">
  <section class="cover">
    <h1>{html.escape(_safe_text(report_cfg.get('report_title')))}</h1>
    <p class="meta"><strong>Organization:</strong> {html.escape(_safe_text(report_cfg.get('organization')))}</p>
    <p class="meta"><strong>Classification:</strong> {html.escape(_safe_text(report_cfg.get('classification')))}</p>
    <p class="meta"><strong>Report Date (UTC):</strong> {html.escape(now_utc)}</p>
    <p class="meta"><strong>Reporting Period:</strong> {html.escape(_safe_text(report_cfg.get('report_period')))}</p>
    <p class="meta"><strong>Version:</strong> {html.escape(_safe_text(report_cfg.get('document_version')))}</p>
  </section>

  <section id="section-executive" class="section page-break">
    <h2 class="section-title">Executive Summary</h2>
    <p><strong>Decision:</strong> {html.escape(decision_name)} {_severity_badge(severity, severity_colors)}</p>
    <p><strong>Business Implication:</strong> {html.escape(business_impact)}</p>
    <p><strong>Model Fit:</strong> R²={model_r2:.3f}. <strong>Primary Signal:</strong> {html.escape(dominant_signal)}</p>
    <p><strong>Leadership Brief:</strong> {html.escape(final_output)}</p>
  </section>

  <section id="section-toc" class="section">
    <h2 class="section-title">Table of Contents</h2>
    <ol class="toc">{toc_html}</ol>
  </section>

  <section id="section-risk" class="section">
    <h2 class="section-title">Risk Assessment Matrix (4×4)</h2>
    {_risk_matrix_html(risk, severity_colors)}
  </section>

  <section id="section-financial" class="section">
    <h2 class="section-title">Financial Impact Analysis</h2>
    <div class="grid-2">
      <div class="card">
        <p><strong>Annual Value at Stake:</strong> {financial.get('currency_symbol', '$')}{financial.get('annual_value_at_stake', 0):,.2f}</p>
        <p><strong>Estimated Avoidable Loss:</strong> {financial.get('currency_symbol', '$')}{financial.get('estimated_avoidable_loss', 0):,.2f}</p>
        <p><strong>Implementation Cost:</strong> {financial.get('currency_symbol', '$')}{financial.get('implementation_cost', 0):,.2f}</p>
        <p><strong>Net Benefit:</strong> {financial.get('currency_symbol', '$')}{financial.get('net_benefit', 0):,.2f}</p>
      </div>
      <div class="card">
        <p><strong>ROI:</strong> {financial.get('roi_percent', 0):.2f}%</p>
        <p><strong>Break-even Timeline:</strong> {financial.get('break_even_months', 'N/A')} months</p>
        <p>{html.escape(_safe_text(financial.get('narrative')))}</p>
      </div>
    </div>
  </section>

  <section id="section-findings" class="section page-break">
    <h2 class="section-title">Detailed Findings</h2>
    <h3>Data Overview</h3>
    <table>
      <tr><th>Rows</th><th>Columns</th><th>Target</th><th>Problem Type</th></tr>
      <tr><td>{int(overview.get('rows', 0)):,}</td><td>{int(overview.get('columns', 0))}</td><td>{html.escape(_safe_text(ml_results.get('target_column')))}</td><td>{html.escape(_safe_text(ml_results.get('problem_type')))}</td></tr>
    </table>

    <h3>Statistical Summary</h3>
    <table>
      <tr><th>Feature</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Q1</th><th>Q3</th></tr>
      {_statistical_rows(statistical_summary)}
    </table>

    <h3>Data Quality Report</h3>
    <div class="grid-2">
      <div class="card">
        <p><strong>Overall Score:</strong> {_to_float(data_quality.get('overall_score')):.1f}/100 (Grade {html.escape(quality_grade)})</p>
        <p><strong>Completeness:</strong> {_to_float(data_quality.get('completeness')):.1f}%</p>
        <p><strong>Uniqueness:</strong> {_to_float(data_quality.get('uniqueness')):.1f}%</p>
        <p><strong>Consistency:</strong> {_to_float(data_quality.get('consistency')):.1f}%</p>
        <p><strong>Plausibility:</strong> {_to_float(data_quality.get('plausibility')):.1f}%</p>
      </div>
      <div class="card">
        <table>
          <tr><th>Column</th><th>Issue</th><th>Count</th></tr>
          {quality_issue_rows}
        </table>
      </div>
    </div>

    <h3>Root Cause Analysis</h3>
    <p><strong>Primary Decision:</strong> {html.escape(decision_name)}</p>
    <p><strong>Dominant Signal:</strong> {html.escape(dominant_signal)}</p>

    <h3>Model Performance</h3>
    <p><strong>R² Score:</strong> {model_r2:.4f}</p>
    <p><strong>Strongest Correlation:</strong> {_to_float(evidence.get('strongest_correlation')):.2f}</p>
    <p><strong>Weak Features:</strong> {html.escape(_weak_feature_display(evidence.get('weak_feature_pct')))}</p>
  </section>

  <section id="section-recommendations" class="section">
    <h2 class="section-title">Recommendations (with indicative timeline)</h2>
    <p><strong>Suggested Timeline:</strong> 0-30 days (critical data fixes), 31-60 days (feature/model refinements), 61-90 days (operational hardening)</p>
    {recommendation_html}
  </section>

  <section id="section-visualizations" class="section">
    <h2 class="section-title">Visualizations</h2>
    {visual_html}
  </section>

  <section id="section-appendix" class="section page-break">
    <h2 class="section-title">Appendix</h2>
    <h3>Methodology</h3>
    <ol>
      <li>Dataset validation and schema checks</li>
      <li>Signal extraction and quality scoring</li>
      <li>Regression model benchmarking and evidence synthesis</li>
      <li>Decision engine severity scoring and recommendation generation</li>
      <li>Risk and financial impact quantification for business action planning</li>
    </ol>

    <h3>Data Dictionary</h3>
    <table><tr><th>Column</th><th>Detected Type</th></tr>{data_dictionary_rows}</table>

    <h3>Assumptions</h3>
    <ul>
      <li>Financial metrics are scenario-based estimates using configured operational assumptions.</li>
      <li>Model quality indicators depend on available historical signal strength.</li>
      <li>Risk levels are relative and should be validated with domain leadership.</li>
    </ul>

    <h3>Limitations</h3>
    <ul>
      <li>No causal guarantees; findings indicate statistical associations.</li>
      <li>Outputs are sensitive to feature completeness and labeling quality.</li>
      <li>Break-even estimates exclude external market and policy shocks.</li>
    </ul>

    <h3>Glossary</h3>
    <ul>
      <li><strong>R²:</strong> Proportion of variance explained by the model.</li>
      <li><strong>ROI:</strong> Return on investment from estimated benefit vs implementation cost.</li>
      <li><strong>Likelihood × Impact:</strong> Risk prioritization scoring matrix.</li>
      <li><strong>Weak Feature:</strong> Predictor with low target correlation.</li>
    </ul>
  </section>
</div>

<footer class="page-footer">
  <span>{html.escape(footer_text)}</span>
  <span>{html.escape(_safe_text(report_cfg.get('confidentiality_notice')))}</span>
  <span class="page-number"></span>
</footer>
</body>
</html>
"""
