from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from analysis import choose_target_column, run_regression_analysis
from html_report import build_html_report
from insights_generator import generate_ranked_insights
from visualization import generate_visualizations


class DataAnalyzer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict = {
            "overview": {},
            "quality_issues": [],
            "ml_results": {},
            "insights": [],
            "visualizations": {},
        }

    def load_data(self) -> pd.DataFrame:
        self.data = pd.read_csv(self.data_path)
        return self.data

    def analyze_overview(self) -> Dict:
        if self.data is None:
            raise ValueError("Data is not loaded")
        overview = {
            "rows": int(self.data.shape[0]),
            "columns": int(self.data.shape[1]),
            "column_types": {
                col: ("numerical" if pd.api.types.is_numeric_dtype(self.data[col]) else "categorical")
                for col in self.data.columns
            },
        }
        self.results["overview"] = overview
        return overview

    def quality_detection(self) -> Dict:
        if self.data is None:
            raise ValueError("Data is not loaded")
        missing = self.data.isnull().sum()
        total_missing = int(missing.sum())
        issues = [
            {
                "column": col,
                "issue": "missing_values",
                "count": int(count),
            }
            for col, count in missing.items()
            if count > 0
        ]
        summary = {"total_missing": total_missing, "issues": issues}
        self.results["quality_issues"] = issues
        return summary

    def ml_pipeline(self, target_column: Optional[str] = None) -> Dict:
        if self.data is None:
            raise ValueError("Data is not loaded")

        selected_target = choose_target_column(self.data, target_column)
        if not selected_target:
            results = {
                "problem_type": "none",
                "target_column": None,
                "severity": "warning",
                "interpretation": "⚠️ No suitable numerical target detected.",
            }
            self.results["ml_results"] = results
            return results

        ml_results = run_regression_analysis(self.data, selected_target)
        self.results["ml_results"] = ml_results
        return ml_results

    def visualizations(self, output_dir: str) -> Dict[str, str]:
        if self.data is None:
            raise ValueError("Data is not loaded")
        target_column = self.results.get("ml_results", {}).get("target_column")
        if not target_column:
            self.results["visualizations"] = {}
            return {}

        visuals = generate_visualizations(self.data, target_column, output_dir)
        self.results["visualizations"] = visuals
        return visuals

    def generate_insights(self, quality_summary: Dict) -> list[str]:
        insights = generate_ranked_insights(self.results, quality_summary)
        self.results["insights"] = insights
        return insights

    def generate_html_report(self, output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        html_content = build_html_report(self.results, self.results.get("visualizations", {}))
        html_path = output_path / "report.html"
        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(html_content)
        return str(html_path)

    def export_json(self, output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "analysis_results.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2, ensure_ascii=False)
        return str(json_path)

    def run_full_analysis(self, output_dir: str = "./output", target_column: Optional[str] = None) -> Dict:
        self.load_data()
        self.analyze_overview()
        quality_summary = self.quality_detection()
        self.ml_pipeline(target_column)
        self.visualizations(output_dir)
        self.generate_insights(quality_summary)

        return {
            "html": self.generate_html_report(output_dir),
            "json": self.export_json(output_dir),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run automated analysis")
    parser.add_argument("data_path", help="Path to csv data file")
    parser.add_argument("output_dir", nargs="?", default="./output", help="Output directory")
    parser.add_argument("target_column", nargs="?", default=None, help="Optional target column")
    args = parser.parse_args()

    analyzer = DataAnalyzer(args.data_path)
    output = analyzer.run_full_analysis(args.output_dir, args.target_column)

    print(f"HTML report: {output['html']}")
    print(f"JSON output: {output['json']}")
    print("Top insights:")
    for item in analyzer.results.get("insights", []):
        print(f"- {item}")
