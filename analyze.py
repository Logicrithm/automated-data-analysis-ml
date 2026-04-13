from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis import choose_target_column, run_regression_analysis
from confidence_calculator import calculate_weighted_confidence
from context import infer_domain
from data_quality_scorer import calculate_data_quality_score
from html_report import build_html_report
from insights_generator import generate_ranked_insights
from model_comparison import train_multiple_models
from multicollinearity_detection import detect_multicollinearity
from rca import diagnose
from recommendations_new import recommend
from conflict_resolver import resolve_conflicts
from signals import extract_signals
from visualization import generate_visualizations

RANDOM_STATE = 42


class DataAnalyzer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict = {
            "overview": {},
            "signals": {},
            "context": {},
            "diagnosis": {},
            "verdict": {},
            "recommendations": {},
            "confidence": {},
            "quality_issues": [],
            "ml_results": {},
            "insights": [],
            "visualizations": {},
            "data_quality": {},
            "model_comparison": {},
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

    def calculate_data_quality(self) -> Dict:
        """Calculate comprehensive data quality score"""
        if self.data is None:
            raise ValueError("Data is not loaded")
        
        quality_result = calculate_data_quality_score(self.data)
        self.results["data_quality"] = quality_result
        return quality_result

    def detect_multicollinearity(self) -> Dict:
        """Detect multicollinearity in numeric features"""
        if self.data is None:
            raise ValueError("Data is not loaded")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return {"high_vif_pairs": [], "vif_data": []}
        
        X = self.data[numeric_cols]
        vif_data = detect_multicollinearity(X)
        
        # Find high VIF pairs (correlation > 0.75)
        high_vif_pairs = []
        numeric_data = self.data[numeric_cols].corr()
        
        for i in range(len(numeric_data.columns)):
            for j in range(i+1, len(numeric_data.columns)):
                corr = abs(numeric_data.iloc[i, j])
                if corr > 0.75:
                    high_vif_pairs.append({
                        "feature_a": numeric_data.columns[i],
                        "feature_b": numeric_data.columns[j],
                        "correlation": float(numeric_data.iloc[i, j]),
                    })
        
        vif_list = vif_data.to_dict('records') if hasattr(vif_data, 'to_dict') else []
        
        return {
            "high_vif_pairs": high_vif_pairs,
            "vif_data": vif_list,
        }

    def generate_model_comparison(self) -> Dict:
        """Train and compare multiple models"""
        ml_results = self.results.get("ml_results", {})
        if ml_results.get("problem_type") != "regression":
            return {"models": [], "best_model": None}
        
        target_column = ml_results.get("target_column")
        if not target_column or target_column not in self.data.columns:
            return {"models": [], "best_model": None}
        
        X = self.data.select_dtypes(include=[np.number]).drop(target_column, axis=1, errors='ignore')
        y = self.data[target_column]
        
        if X.empty or len(X) < 20:
            return {"models": [], "best_model": None}
        
        comparison = train_multiple_models(X, y)
        self.results["model_comparison"] = comparison
        return comparison

    def generate_insights(self, quality_summary: Dict, multicollinearity: Dict) -> List[Dict]:
        """Generate ranked insights with confidence"""
        insights = generate_ranked_insights(self.results, quality_summary, multicollinearity)
        self.results["insights"] = insights
        return insights

    def calculate_confidence_scores(self) -> Dict:
        """Calculate multi-dimensional confidence"""
        ml_results = self.results.get("ml_results", {})
        data_quality_obj = self.results.get("data_quality", {})
        signals = self.results.get("signals", {})
        
        # Extract quality metrics
        data_quality_score = (data_quality_obj.get("data_quality", {}).get("overall_score", 50) or 50) / 100
        
        # Extract model performance
        r2 = float(ml_results.get("r2_score", 0.0))
        model_performance = min(1.0, r2 + 0.3)  # Normalize
        
        # Extract feature relevance
        feature_importance = ml_results.get("standardized_importance", [])
        feature_relevance = feature_importance[0].get("importance", 0.0) if feature_importance else 0.3
        
        # Domain confidence
        context = self.results.get("context", {})
        domain_confidence = context.get("confidence", 0.5)
        
        # Actionability
        recommendations = self.results.get("recommendations", [])
        has_actions = bool(recommendations) if isinstance(recommendations, list) else False
        actionability = 0.9 if has_actions else 0.5
        
        confidence = calculate_weighted_confidence(
            data_quality_score,
            model_performance,
            feature_relevance,
            domain_confidence,
            actionability,
        )
        
        self.results["confidence"] = confidence
        return confidence

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
        unique_contents = {str(i.get("content", "")) for i in insights}
        if len(insights) != len(unique_contents):
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
        """
        Complete analysis pipeline with conflict resolution and structured diagnosis.
        
        Flow:
        1. Load and analyze data
        2. ML pipeline
        3. Extract signals (NEW)
        4. Infer domain (UPGRADED)
        5. Data quality
        6. Visualizations
        7. Model comparison
        8. RCA diagnosis (NEW)
        9. Resolve conflicts (NEW - CRITICAL)
        10. Generate recommendations (UPGRADED)
        11. Generate insights
        12. Calculate confidence
        13. Validate
        14. Generate reports
        """
        np.random.seed(RANDOM_STATE)
        
        # Step 1-2: Load and analyze
        self.load_data()
        self.analyze_overview()
        quality_summary = self.quality_detection()
        
        # Step 3: ML pipeline
        self.ml_pipeline(target_column)
        ml_results = self.results.get("ml_results", {})
        target_col = ml_results.get("target_column") or target_column
        
        # Step 4: Extract signals (NEW)
        self.results["signals"] = extract_signals(self.data, target_col)
        
        # Step 5: Infer domain (UPGRADED)
        self.results["context"] = infer_domain(self.results["signals"], self.data)
        
        # Step 6: Data quality
        self.calculate_data_quality()
        
        # Step 7: Visualizations
        self.visualizations(output_dir)
        
        # Step 8: Model comparison
        self.generate_model_comparison()
        
        # Step 9: RCA diagnosis (NEW)
        self.results["diagnosis"] = diagnose(self.results["signals"], ml_results)
        
        # Step 10: Resolve conflicts (NEW - CRITICAL)
        self.results["verdict"] = resolve_conflicts(
            self.results["signals"],
            self.results["context"].get("domain", "generic"),
            self.results["diagnosis"],
            []
        )
        
        # Step 11: Generate recommendations (UPGRADED)
        self.results["recommendations"] = recommend(
            self.results["context"].get("domain", "generic"),
            self.results["diagnosis"]
        )
        
        # Step 12: Multicollinearity detection
        multicollinearity = self.detect_multicollinearity()
        
        # Step 13: Generate insights
        self.generate_insights(quality_summary, multicollinearity)
        
        # Step 14: Calculate confidence
        self.calculate_confidence_scores()
        
        # Step 15: Validate
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
        content = item.get("content", "No content") if isinstance(item, dict) else str(item)
        print(f"- {content}")
