from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path
from typing import Dict, List


def _visual_block(name: str, path: str) -> str:
    """Embed visualization files (images or HTML) into report."""
    file_path = Path(path)
    if not file_path.exists():
        return f'<div class="viz"><h4>{html.escape(name.replace("_", " ").title())}</h4><p>Visualization file not found.</p></div>'

    if file_path.suffix.lower() == ".html":
        try:
            content = file_path.read_text(encoding='utf-8')
            return f'<div class="viz"><h4>{html.escape(name.replace("_", " ").title())}</h4>{content}</div>'
        except:
            return f'<div class="viz"><h4>{html.escape(name)}</h4><p>Could not read HTML file.</p></div>'

    try:
        mime_type = mimetypes.guess_type(str(file_path))[0] or "image/png"
        encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        return f'<div class="viz"><h4>{html.escape(name.replace("_", " ").title())}</h4><img src="data:{mime_type};base64,{encoded}" alt="{html.escape(name)}" style="max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px;"/></div>'
    except:
        return f'<div class="viz"><h4>{html.escape(name)}</h4><p>Could not read visualization file.</p></div>'


def build_html_report(results: Dict, visuals: Dict[str, str]) -> str:
    """
    Generate comprehensive HTML report with verdict and structured diagnosis.
    """
    
    # Extract all data
    signals = results.get("signals", {})
    context = results.get("context", {})
    diagnosis = results.get("diagnosis", {})
    verdict = results.get("verdict", {})
    recommendations = results.get("recommendations", [])
    confidence = results.get("confidence", {})
    model_comparison = (results.get("model_comparison") or {}).get("models") or []
    best_model = (results.get("model_comparison") or {}).get("best_model")
    data_quality = (results.get("data_quality") or {}).get("data_quality", {})
    quality_grade = (results.get("data_quality") or {}).get("grade", "N/A")
    overview = results.get("overview", {})
    ml_results = results.get("ml_results", {})
    
    # Determine warning level
    primary_issue = verdict.get("primary_issue", "unknown")
    is_critical = primary_issue in ["data_quality", "feature_gap", "multicollinearity"]
    verdict_confidence = verdict.get("confidence", 0)
    
    # Build warning banner
    warning_html = ""
    if is_critical:
        if verdict.get("is_data_problem"):
            warning_html = '<div class="warning-banner critical">🚨 DATA QUALITY ISSUE - Results may be unreliable</div>'
        elif verdict.get("is_feature_problem"):
            warning_html = '<div class="warning-banner high">⚠️ FEATURE PROBLEM DETECTED - Missing or redundant features</div>'
        elif verdict.get("is_model_problem"):
            warning_html = '<div class="warning-banner medium">ℹ️ MODEL MISMATCH - Consider different algorithms</div>'
    
    # Build executive summary
    executive_html = f"""
    <div class="executive-summary">
        <h3>📊 Executive Summary</h3>
        <div class="summary-grid">
            <div class="summary-card">
                <strong>Main Issue:</strong>
                <span class="value">{html.escape(str(primary_issue)).upper()}</span>
                <span class="confidence">Confidence: {verdict_confidence:.0%}</span>
            </div>
            <div class="summary-card">
                <strong>Model Status:</strong>
                <span class="value">{html.escape(str(diagnosis.get('model_perf', 'unknown')).upper())}</span>
                <span class="r2">R² = {float(ml_results.get('r2_score', 0)):.1%}</span>
            </div>
            <div class="summary-card">
                <strong>Recommended Action:</strong>
                <span class="value">{html.escape(str(recommendations[0].get('action', 'Review data')[:40])) if recommendations else 'No action'}</span>
            </div>
        </div>
    </div>
    """
    
    # Build context section
    context_html = f"""
    <div class="context-section">
        <h3>🌐 Domain Context</h3>
        <div class="context-card">
            <div class="context-item">
                <strong>Detected Domain:</strong>
                <span class="domain">{html.escape(str(context.get('domain', 'generic')))}</span>
                <span class="confidence">({context.get('confidence', 0):.0%} confidence)</span>
            </div>
            <div class="context-reasoning">
                <strong>Evidence:</strong>
                <ul>
    """
    
    for reason in context.get("reasoning", []):
        context_html += f"<li>{html.escape(str(reason))}</li>"
    
    context_html += """
                </ul>
            </div>
        </div>
    </div>
    """
    
    # Build diagnosis section
    diagnosis_html = f"""
    <div class="diagnosis-section">
        <h3>🔍 Root Cause Analysis</h3>
        <div class="diagnosis-grid">
            <div class="diagnosis-item {html.escape(str(diagnosis.get('model_perf', 'unknown')).lower())}">
                <strong>Model Performance:</strong>
                <span class="value">{html.escape(str(diagnosis.get('model_perf', 'unknown')).upper())}</span>
            </div>
            <div class="diagnosis-item {html.escape(str(diagnosis.get('feature_strength', 'unknown')).lower())}">
                <strong>Feature Strength:</strong>
                <span class="value">{html.escape(str(diagnosis.get('feature_strength', 'unknown')).upper())}</span>
            </div>
            <div class="diagnosis-item {html.escape(str(diagnosis.get('multicollinearity', 'unknown')).lower())}">
                <strong>Multicollinearity:</strong>
                <span class="value">{html.escape(str(diagnosis.get('multicollinearity', 'unknown')).upper())}</span>
            </div>
            <div class="diagnosis-item {html.escape(str(diagnosis.get('data_quality', 'unknown')).lower())}">
                <strong>Data Quality:</strong>
                <span class="value">{html.escape(str(diagnosis.get('data_quality', 'unknown')).upper())}</span>
            </div>
        </div>
    </div>
    """
    
    # Build recommendations section
    rec_html = "<div class='recommendations-section'><h3>💡 Recommendations (Prioritized)</h3>"
    for i, rec in enumerate(recommendations[:5], 1):
        priority_class = html.escape(str(rec.get("priority", "MEDIUM")).lower())
        rec_html += f"""
        <div class="recommendation {priority_class}">
            <div class="rec-header">
                <span class="priority-badge">[{html.escape(str(rec.get('priority', 'MEDIUM')))}]</span>
                <span class="action">{html.escape(str(rec.get('action', '')))}</span>
            </div>
            <div class="rec-body">
                <p><strong>Reason:</strong> {html.escape(str(rec.get('reason', '')))}</p>
                <div class="rec-meta">
                    <span><strong>Impact:</strong> {html.escape(str(rec.get('impact', 'UNKNOWN')))}</span>
                    <span><strong>Effort:</strong> {html.escape(str(rec.get('effort', 'UNKNOWN')))}</span>
                </div>
            </div>
        </div>
        """
    rec_html += "</div>"
    
    # Build model comparison section
    model_html = "<div class='model-section'><h3>📈 Model Comparison</h3><table class='model-table'><tr><th>Model</th><th>R² Score</th><th>RMSE</th><th>Speed</th><th>Recommendation</th></tr>"
    
    for model in model_comparison:
        model_html += f"""
        <tr>
            <td>{html.escape(str(model.get("name", "")))}</td>
            <td>{float(model.get("r2_score", 0)):.3f}</td>
            <td>{float(model.get("rmse", 0)):.0f}</td>
            <td>{float(model.get("training_time", 0)):.3f}s</td>
            <td>{html.escape(str(model.get("recommendation", "")))}</td>
        </tr>
        """
    
    model_html += f"""
    </table>
    <p><strong>Best Model:</strong> {html.escape(str(best_model or 'N/A'))}</p>
    </div>
    """
    
    # Build data quality section
    quality_html = f"""
    <div class="quality-section">
        <h3>📊 Data Quality Assessment</h3>
        <div class="quality-scorecard">
            <div class="quality-score">
                <strong>Overall Score:</strong>
                <span class="score">{float(data_quality.get('overall_score', 0)):.1f}/100</span>
                <span class="grade">(Grade: {html.escape(str(quality_grade))})</span>
            </div>
            <div class="quality-metrics">
                <div class="metric">
                    <strong>Completeness:</strong> {float(data_quality.get('completeness', 0)):.0f}%
                </div>
                <div class="metric">
                    <strong>Uniqueness:</strong> {float(data_quality.get('uniqueness', 0)):.0f}%
                </div>
                <div class="metric">
                    <strong>Consistency:</strong> {float(data_quality.get('consistency', 0)):.0f}%
                </div>
                <div class="metric">
                    <strong>Plausibility:</strong> {float(data_quality.get('plausibility', 0)):.0f}%
                </div>
            </div>
        </div>
    </div>
    """
    
    # Build visualizations section
    visual_html = "".join(_visual_block(name, path) for name, path in visuals.items())
    viz_section = f"""
    <div class="visualizations-section">
        <h3>📊 Visualizations</h3>
        {visual_html or '<p>No visualizations available.</p>'}
    </div>
    """ if visual_html else ""
    
    # Build dataset summary
    summary_html = f"""
    <div class="summary-section">
        <h3>📋 Dataset Summary</h3>
        <div class="summary-info">
            <p><strong>Rows:</strong> {int(overview.get('rows', 0)):,}</p>
            <p><strong>Columns:</strong> {int(overview.get('columns', 0))}</p>
            <p><strong>Target Column:</strong> {html.escape(str(ml_results.get('target_column', 'None')))}</p>
            <p><strong>Problem Type:</strong> {html.escape(str(ml_results.get('problem_type', 'None')))}</p>
        </div>
    </div>
    """
    
    # Combine all HTML with CSS
    html_content = f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Automated Data Analysis Report</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                    line-height: 1.6; color: #1f2937; background: #f9fafb; }}
            
            .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
            
            h1 {{ font-size: 2.5em; margin: 20px 0 10px 0; color: #111827; }}
            h2 {{ font-size: 1.8em; margin: 20px 0 10px 0; color: #1f2937; }}
            h3 {{ font-size: 1.3em; margin: 15px 0 10px 0; color: #374151; }}
            
            .section {{ background: white; padding: 20px; border-radius: 8px; margin: 15px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            
            .warning-banner {{ padding: 15px; border-radius: 6px; margin: 20px 0; font-weight: bold; border-left: 5px solid; }}
            .warning-banner.critical {{ background: #fee2e2; color: #991b1b; border-color: #dc2626; }}
            .warning-banner.high {{ background: #fef3c7; color: #92400e; border-color: #f59e0b; }}
            .warning-banner.medium {{ background: #dbeafe; color: #1e40af; border-color: #3b82f6; }}
            
            .executive-summary {{ background: #f0f9ff; padding: 16px; border-radius: 8px; border-left: 4px solid #0284c7; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 15px; }}
            .summary-card {{ padding: 15px; background: white; border-radius: 6px; border: 1px solid #e5e7eb; }}
            .summary-card strong {{ display: block; font-size: 0.85em; color: #666; text-transform: uppercase; margin-bottom: 5px; }}
            .summary-card .value {{ font-size: 1.3em; font-weight: bold; color: #1f2937; }}
            .summary-card .confidence, .summary-card .r2 {{ display: block; font-size: 0.85em; color: #888; margin-top: 5px; }}
            
            .context-card {{ padding: 15px; background: #f0f9ff; border-radius: 6px; border-left: 4px solid #0284c7; }}
            .context-item {{ margin-bottom: 10px; }}
            .context-item strong {{ display: inline; }}
            .context-item .domain {{ font-weight: bold; color: #0284c7; }}
            .context-reasoning {{ margin-top: 10px; }}
            .context-reasoning strong {{ display: block; margin-bottom: 8px; }}
            .context-reasoning ul {{ list-style: none; padding-left: 20px; }}
            .context-reasoning li {{ margin: 5px 0; padding-left: 15px; position: relative; }}
            .context-reasoning li:before {{ content: "✓"; position: absolute; left: 0; color: #0284c7; font-weight: bold; }}
            
            .diagnosis-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 15px; }}
            .diagnosis-item {{ padding: 15px; border-radius: 6px; border-left: 4px solid #cbd5e1; }}
            .diagnosis-item strong {{ display: block; font-size: 0.85em; color: #666; text-transform: uppercase; margin-bottom: 8px; }}
            .diagnosis-item .value {{ font-size: 1.2em; font-weight: bold; }}
            
            .diagnosis-item.critical {{ background: #fee2e2; border-color: #dc2626; color: #991b1b; }}
            .diagnosis-item.weak {{ background: #fef3c7; border-color: #f59e0b; color: #92400e; }}
            .diagnosis-item.poor {{ background: #fee2e2; border-color: #dc2626; color: #991b1b; }}
            .diagnosis-item.moderate, .diagnosis-item.fair {{ background: #dbeafe; border-color: #3b82f6; color: #1e40af; }}
            .diagnosis-item.good, .diagnosis-item.excellent {{ background: #dcfce7; border-color: #16a34a; color: #166534; }}
            .diagnosis-item.none, .diagnosis-item.low {{ background: #f0fdf4; border-color: #86efac; color: #166534; }}
            
            .recommendation {{ padding: 15px; border-radius: 6px; margin: 12px 0; border-left: 4px solid #cbd5e1; }}
            .recommendation.critical {{ background: #fee2e2; border-color: #dc2626; }}
            .recommendation.high {{ background: #fef3c7; border-color: #f59e0b; }}
            .recommendation.medium {{ background: #dbeafe; border-color: #3b82f6; }}
            .recommendation.low {{ background: #f0fdf4; border-color: #16a34a; }}
            
            .rec-header {{ display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }}
            .priority-badge {{ font-weight: bold; font-size: 0.85em; }}
            .action {{ flex: 1; font-weight: bold; }}
            
            .rec-body {{ font-size: 0.95em; }}
            .rec-body p {{ margin: 5px 0; }}
            .rec-meta {{ display: flex; gap: 15px; font-size: 0.9em; margin-top: 8px; color: #666; }}
            
            .model-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            .model-table th {{ background: #f3f4f6; font-weight: bold; padding: 12px; text-align: left; border: 1px solid #e5e7eb; }}
            .model-table td {{ padding: 10px 12px; border: 1px solid #e5e7eb; }}
            .model-table tr:nth-child(even) {{ background: #fafbfc; }}
            
            .quality-scorecard {{ background: #f0fdf4; padding: 15px; border-radius: 6px; border-left: 4px solid #16a34a; }}
            .quality-score {{ margin-bottom: 15px; }}
            .quality-score strong {{ display: inline; }}
            .score {{ font-size: 1.5em; font-weight: bold; color: #16a34a; margin: 0 10px; }}
            .grade {{ font-size: 0.9em; color: #666; }}
            
            .quality-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 10px; }}
            .metric {{ padding: 10px; background: white; border-radius: 4px; border: 1px solid #e5e7eb; }}
            .metric strong {{ display: block; font-size: 0.85em; color: #666; margin-bottom: 5px; }}
            
            .viz {{ margin: 20px 0; }}
            .viz h4 {{ margin-bottom: 10px; }}
            
            .summary-info {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }}
            .summary-info p {{ margin: 8px 0; }}
            .summary-info strong {{ display: inline; color: #374151; }}
            
            table {{ width: 100%; border-collapse: collapse; }}
            table th, table td {{ border: 1px solid #e5e7eb; padding: 10px; text-align: left; }}
            table th {{ background: #f3f4f6; font-weight: bold; }}
            
            ul {{ list-style: none; padding-left: 0; }}
            li {{ margin: 8px 0; }}
            
            @media (max-width: 768px) {{
                .summary-grid, .diagnosis-grid, .quality-metrics {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Automated Data Analysis Report</h1>
            
            {warning_html}
            
            <div class="section">
                {executive_html}
            </div>
            
            <div class="section">
                {context_html}
            </div>
            
            <div class="section">
                {diagnosis_html}
            </div>
            
            <div class="section">
                {rec_html}
            </div>
            
            <div class="section">
                {model_html}
            </div>
            
            <div class="section">
                {quality_html}
            </div>
            
            {viz_section}
            
            <div class="section">
                {summary_html}
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content
