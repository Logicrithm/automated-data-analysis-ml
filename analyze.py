from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis import choose_target_column, run_regression_analysis
from business_impact import generate_business_impact
from confidence_calculator import calculate_weighted_confidence
from consistency_validator import validate_consistency as validate_output_consistency
from context import infer_domain
from causal_layer import build_causal_explanation
from data_quality_scorer import calculate_data_quality_score
from dataset_validator import validate_dataset
from decision_engine import decide_root_cause
from deep_summary import generate_deep_summary
from evidence import build_evidence
from feature_analysis import analyze_features
from final_output import generate_final_output
from html_report import build_html_report
from insights_generator import generate_ranked_insights
from model_comparison import train_multiple_models
from model_interpreter import interpret_models
from multicollinearity_detection import detect_multicollinearity
from recommendation_engine import generate_recommendations_v2
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
            "recommendations": [],
            "feature_analysis": {},
            "model_interpretation": {},
            "deep_summary": {},
            "evidence": {},
            "causal_layer": {},
            "validation_report": {},
            "confidence": {},
            "quality_issues": [],
            "ml_results": {},
            "insights": [],
            "visualizations": {},
            "data_quality": {},
            "model_comparison": {},
            "final_output": "",
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
            for j in range(i + 1, len(numeric_data.columns)):
                corr = abs(numeric_data.iloc[i, j])
                if corr > 0.75:
                    high_vif_pairs.append(
                        {
                            "feature_a": numeric_data.columns[i],
                            "feature_b": numeric_data.columns[j],
                            "correlation": float(numeric_data.iloc[i, j]),
                        }
                    )

        vif_list = vif_data.to_dict("records") if hasattr(vif_data, "to_dict") else []

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

        X = self.data.select_dtypes(include=[np.number]).drop(target_column, axis=1, errors="ignore")
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
        np.random.seed(RANDOM_STATE)

        # Load data
        self.load_data()

        # Early hard validation gate
        gate = validate_dataset(self.data, target_column)
        if not gate.get("valid", False):
            message = f"This dataset is insufficient for reliable analysis due to: {gate.get('reason', 'validation failure')}."
            self.results["final_output"] = message
            return {"final_output": message}

        # Step 1-2: Overview + quality summary
        self.analyze_overview()
        quality_summary = self.quality_detection()

        # Step 3: ML pipeline
        self.ml_pipeline(target_column)
        ml_results = self.results.get("ml_results", {})
        target_col = ml_results.get("target_column") or target_column

        # If no suitable target found, fail cleanly (business-facing)
        if not target_col:
            message = "This dataset is insufficient for reliable analysis due to: no suitable numerical target column."
            self.results["final_output"] = message
            return {"final_output": message}

        # Step 4: Signals + context
        self.results["signals"] = extract_signals(self.data, target_col)
        self.results["context"] = infer_domain(self.results["signals"], self.data)

        # Step 5: data quality + visuals + model comparison
        self.calculate_data_quality()
        self.visualizations(output_dir)
        self.generate_model_comparison()

        # Step 6: insights/confidence
        multicollinearity = self.detect_multicollinearity()
        self.generate_insights(quality_summary, multicollinearity)
        self.calculate_confidence_scores()
        self.validate_consistency()

        # Step 7: Feature analysis + evidence
        feature_analysis = analyze_features(self.data, target_col)
        self.results["feature_analysis"] = feature_analysis
        ml_results["model_comparison"] = self.results.get("model_comparison", {})

        evidence = build_evidence(
            self.results["signals"],
            feature_analysis,
            ml_results,
            self.results.get("diagnosis"),
        )
        self.results["evidence"] = evidence

        # Step 8: Decision + causality
        decision = decide_root_cause(evidence)
        self.results["diagnosis"] = decision

        causal_layer = build_causal_explanation(evidence, decision, ml_results)
        self.results["causal_layer"] = causal_layer

        # Step 9: Recommendations
        recommendations = generate_recommendations_v2(
            decision,
            evidence,
            causal_layer,
            self.results.get("context"),
        )
        self.results["recommendations"] = recommendations

        # Step 10: Model interpretation + deep summary
        model_interpretation = interpret_models(ml_results, self.results["diagnosis"])
        self.results["model_interpretation"] = model_interpretation

        deep_summary = generate_deep_summary(
            evidence,
            decision,
            causal_layer,
        )

        validated = validate_output_consistency(
            decision,
            recommendations,
            deep_summary,
            self.results.get("verdict"),
        )
        self.results["recommendations"] = validated.get("recommendations", recommendations)
        self.results["deep_summary"] = validated.get("deep_summary", deep_summary)
        self.results["verdict"] = validated.get("verdict", {})
        self.results["validation_report"] = validated.get("validation_report", {})

        # Step 11: Business impact + strict final output
        business_impact = generate_business_impact(self.results["diagnosis"], self.results["evidence"])

        final_text = generate_final_output(
            decision=self.results["diagnosis"],
            evidence=self.results["evidence"],
            causal_layer=self.results["causal_layer"],
            recommendations=self.results["recommendations"],
            business_impact=business_impact,
        )
        self.results["final_output"] = final_text

        # Still store artifacts for internal debugging/audit, but user-facing output is final_output only
        html_path = self.generate_html_report(output_dir)
        json_path = self.export_json(output_dir)

        return {
            "final_output": final_text,
            "html": html_path,
            "json": json_path,
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

    # STRICT USER-FACING OUTPUT
    print(output.get("final_output", "No final output generated."))
