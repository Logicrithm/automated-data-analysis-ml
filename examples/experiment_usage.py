import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.experiments import ExperimentPipeline


def run_experiment_example(csv_path: str, target_column: str) -> dict:
    data = pd.read_csv(csv_path)
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    pipeline = ExperimentPipeline(X, y)
    pipeline.run_all()
    return pipeline.get_structured_results()


if __name__ == "__main__":
    SAMPLE_PATH = "/tmp/ada_exp_smoke.csv"
    TARGET = "target"
    results = run_experiment_example(SAMPLE_PATH, TARGET)
    print(results)
