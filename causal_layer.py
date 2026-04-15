from __future__ import annotations

from typing import Dict, List


def build_causal_explanation(evidence: Dict | None, decision: Dict | None, ml_results: Dict | None = None) -> Dict:
    evidence = evidence or {}
    decision = decision or {}
    ml_results = ml_results or {}

    decision_name = str(decision.get("decision", "UNKNOWN"))
    r2_percentage = float(evidence.get("r2_percentage", 0.0))
    weak_feature_pct = int(evidence.get("weak_feature_pct", 0))
    strongest_correlation = float(evidence.get("strongest_correlation", 0.0))
    redundant_pairs_count = int(evidence.get("redundant_pairs_count", 0))
    max_redundancy = float(evidence.get("max_redundancy_correlation", 0.0))
    data_quality_score = float(evidence.get("data_quality_score", 0.0))
    missing_percentage = float(evidence.get("missing_percentage", 0.0))

    if decision_name == "DATA_ISSUE":
        root_cause = "Data reliability is insufficient for trustworthy model learning."
        evidence_chain = [
            f"Data quality score is {data_quality_score:.1f}/100.",
            f"Missing value rate is {missing_percentage:.1f}%.",
            "Training signal is degraded before model selection becomes meaningful.",
        ]
        model_impact = f"Model results are unstable; current R² of {r2_percentage:.1f}% is not actionable until data quality improves."
        domain_context = "Prioritize dataset cleanup and schema consistency before further modeling changes."
    elif decision_name == "MULTICOLLINEARITY":
        root_cause = "Multiple features encode overlapping information, causing unstable coefficient attribution."
        evidence_chain = [
            f"Detected {redundant_pairs_count} redundant feature pairs.",
            f"Maximum inter-feature correlation reaches {max_redundancy:.2f}.",
            "Redundant signal makes linear parameter estimates sensitive to small data changes.",
        ]
        model_impact = f"Predictive fit can appear inconsistent (R² {r2_percentage:.1f}%) with poor interpretability and fragile coefficients."
        domain_context = "Feature pruning or dimensionality reduction is needed to recover robust signal ownership."
    elif decision_name == "NON_LINEARITY":
        root_cause = "Linear assumptions fail to capture nonlinear relationships present in the data."
        evidence_chain = [
            f"Strongest feature correlation is {strongest_correlation:.2f}.",
            f"Overall fit remains low with R² at {r2_percentage:.1f}%.",
            "Available signal exists but linear structure underfits the response pattern.",
        ]
        model_impact = "Linear regression leaves large variance unexplained; nonlinear learners should increase captured structure."
        domain_context = "Use tree-based methods or interaction transforms aligned with domain behavior."
    elif decision_name == "FEATURE_GAP":
        total_features = int(evidence.get("total_features", 0))
        root_cause = "The dataset lacks sufficient domain variables to explain the target."
        evidence_chain = [
            f"Only {total_features} usable features were available.",
            f"Weak feature ratio is {weak_feature_pct}%.",
            "Limited feature breadth constrains predictive coverage.",
        ]
        model_impact = f"Current models plateau around R² {r2_percentage:.1f}% because key explanatory drivers are missing."
        domain_context = "Collect additional domain-grounded inputs before algorithm-level tuning."
    else:
        root_cause = "Feature signal strength is insufficient for high-confidence prediction."
        evidence_chain = [
            f"{weak_feature_pct}% of features are weakly correlated with the target.",
            f"Strongest observed target correlation is {strongest_correlation:.2f}.",
            f"Resulting fit remains limited at R² {r2_percentage:.1f}%.",
        ]
        model_impact = "Low-signal features limit explainable variance and increase prediction uncertainty."
        domain_context = "Feature engineering and domain variable enrichment are the highest-leverage next steps."

    target_name = ml_results.get("target_column", "target")
    return {
        "root_cause": root_cause,
        "evidence_chain": evidence_chain,
        "model_impact": model_impact,
        "domain_context": f"For target '{target_name}': {domain_context}",
    }
