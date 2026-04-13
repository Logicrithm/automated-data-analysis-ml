from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis import choose_target_column, run_regression_analysis
from confidence_calculator import calculate_confidence_levels
from html_report import build_html_report
from insights_generator import generate_ranked_insights
from multicollinearity_detection import detect_multicollinearity
from visualization import generate_visualizations

RANDOM_STATE = 42


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
        duplicate_rows = int(self.data.duplicated().sum())
        issues = [
            {
                "column": col,
                "issue": "missing_values",
                "count": int(count),
            }
            for col, count in missing.items()
            if count > 0
        ]
        if duplicate_rows > 0:
            issues.append({"column": "__all__", "issue": "duplicate_rows", "count": duplicate_rows})
        summary = {"total_missing": total_missing, "duplicate_rows": duplicate_rows, "issues": issues}
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

    def multicollinearity_pipeline(self, target_column: Optional[str]) -> Dict:
        if self.data is None:
            raise ValueError("Data is not loaded")
        if not target_column or target_column not in self.data.columns:
            summary = {"high_vif_features": [], "high_vif_pairs": [], "vif": []}
            self.results["multicollinearity"] = summary
            return summary

        feature_df = self.data.drop(columns=[target_column])
        feature_df = pd.get_dummies(feature_df, drop_first=True)
        feature_df = feature_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
        feature_df = feature_df.fillna(0)
        if feature_df.empty or feature_df.shape[1] < 2:
            summary = {"high_vif_features": [], "high_vif_pairs": [], "vif": []}
            self.results["multicollinearity"] = summary
            return summary

        vif_df = detect_multicollinearity(feature_df)
        vif_records = [
            {"feature": str(row["Feature"]), "vif": float(row["VIF"])}
            for _, row in vif_df.iterrows()
        ]
        high_vif_features = [entry for entry in vif_records if entry["vif"] >= 5.0]
        corr_matrix = feature_df.corr().abs()
        high_vif_pairs = []
        for i, feature_a in enumerate(corr_matrix.columns):
            for j, feature_b in enumerate(corr_matrix.columns):
                if j <= i:
                    continue
                corr = float(corr_matrix.iloc[i, j])
                if corr >= 0.75 and (
                    any(item["feature"] == feature_a for item in high_vif_features)
                    or any(item["feature"] == feature_b for item in high_vif_features)
                ):
                    high_vif_pairs.append({"feature_a": feature_a, "feature_b": feature_b, "correlation": corr})

        high_vif_pairs.sort(key=lambda item: item["correlation"], reverse=True)
        summary = {
            "high_vif_features": sorted(high_vif_features, key=lambda item: item["vif"], reverse=True),
            "high_vif_pairs": high_vif_pairs,
            "vif": sorted(vif_records, key=lambda item: item["vif"], reverse=True),
        }
        self.results["multicollinearity"] = summary
        ml_results = self.results.get("ml_results", {})
        ml_results["vif"] = summary["vif"]
        if summary["high_vif_pairs"]:
            top_pair = summary["high_vif_pairs"][0]
            ml_results["multicollinearity_warning"] = (
                f"⚠️ Multicollinearity warning: {top_pair['feature_a']} and {top_pair['feature_b']} "
                f"have correlation {top_pair['correlation']:.2f}."
            )
        elif summary["high_vif_features"]:
            ml_results["multicollinearity_warning"] = "⚠️ Multicollinearity warning: high VIF detected."
        else:
            ml_results["multicollinearity_warning"] = ""
        return summary

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

    def generate_insights(self, quality_summary: Dict, confidence_levels: Dict[str, str]) -> List[Dict]:
        insights = generate_ranked_insights(self.results, quality_summary, confidence_levels)
        self.results["insights"] = insights
        return insights

    def validate_consistency(self) -> None:
        ml_results = self.results.get("ml_results", {})
        if ml_results.get("problem_type") != "regression":
            return

        r2_value = float(ml_results.get("r2_score", 0.0))
        if not np.isfinite(r2_value):
            raise ValueError("R² score must be finite.")

        strongest_feature = ml_results.get("strongest_feature")
        standardized = ml_results.get("standardized_importance") or []
        if strongest_feature and standardized:
            if strongest_feature != standardized[0].get("feature"):
                raise ValueError("Strongest predictor must match standardized importance ranking.")

        insights = self.results.get("insights", [])
        insight_contents = [item.get("content", "").strip() for item in insights if isinstance(item, dict)]
        if len(insight_contents) != len(set(insight_contents)):
            raise ValueError("Insights must be deduplicated.")

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
        np.random.seed(RANDOM_STATE)
        self.load_data()
        self.analyze_overview()
        quality_summary = self.quality_detection()
        ml_results = self.ml_pipeline(target_column)
        multicollinearity_summary = self.multicollinearity_pipeline(ml_results.get("target_column"))
        confidence_levels = calculate_confidence_levels(self.results, quality_summary, multicollinearity_summary)
        self.results["confidence_levels"] = confidence_levels
        self.visualizations(output_dir)
        self.generate_insights(quality_summary, confidence_levels)
        self.validate_consistency()

        return {
            "html": self.generate_html_report(output_dir),
            "json": self.export_json(output_dir),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run automated analysis")
    parser.add_argument("data_path", help="Path to CSV data file")
    parser.add_argument("output_dir", nargs="?", default="./output", help="Output directory")
    parser.add_argument("target_column", nargs="?", default=None, help="Optional target column")
    args = parser.parse_args()

    analyzer = DataAnalyzer(args.data_path)
    output = analyzer.run_full_analysis(args.output_dir, args.target_column)

    print(f"HTML report: {output['html']}")
    print(f"JSON output: {output['json']}")
    print("Top insights:")
    for item in analyzer.results.get("insights", []):
        if isinstance(item, dict):
            print(f"- [{item.get('severity')}/{item.get('confidence')}] {item.get('content')}")
        else:
            print(f"- {item}")
