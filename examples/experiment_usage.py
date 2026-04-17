import pandas as pd
import sys
import argparse
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.experiments import ExperimentPipeline


def run_experiment_example(csv_path: str, target_column: str) -> Dict:
    """Run all validation experiments and return structured comparison results."""
    data = pd.read_csv(csv_path)
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    pipeline = ExperimentPipeline(X, y)
    pipeline.run_all()
    return pipeline.get_structured_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment pipeline on a CSV file.")
    parser.add_argument("csv_path", help="Path to CSV dataset.")
    parser.add_argument("target_column", help="Name of target column.")
    args = parser.parse_args()

    results = run_experiment_example(args.csv_path, args.target_column)
    print(results)
