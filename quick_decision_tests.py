import os
import numpy as np
import pandas as pd
import subprocess
import tempfile

np.random.seed(42)

PROJECT_CMD = ["python", "analyze.py"]  # adjust if needed


def run_case(csv_path, target):
    cmd = PROJECT_CMD + [csv_path, "./output", target]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip()


def make_multicollinearity_df(n=300):
    x1 = np.random.normal(0, 1, n)
    x2 = x1 * 0.98 + np.random.normal(0, 0.02, n)  # highly correlated with x1
    x3 = np.random.normal(0, 1, n)
    y = 50 + 10 * x1 + np.random.normal(0, 0.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "Target": y})


def make_nonlinear_df(n=300):
    x = np.random.uniform(-3, 3, n)
    z = np.random.normal(0, 1, n)
    y = x**2 + 0.2 * z + np.random.normal(0, 0.3, n)  # strong non-linear
    return pd.DataFrame({"x": x, "z": z, "Target": y})


def make_weak_signal_df(n=300):
    a = np.random.normal(0, 1, n)
    b = np.random.normal(0, 1, n)
    c = np.random.normal(0, 1, n)
    y = np.random.normal(0, 1, n)  # almost independent target
    return pd.DataFrame({"a": a, "b": b, "c": c, "Target": y})


def extract_problem_block(output_text):
    # grabs the first 2 lines after "Problem:"
    lines = output_text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() == "problem:":
            return lines[i + 1].strip() if i + 1 < len(lines) else "N/A"
    return "N/A"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        cases = {
            "multicollinearity": make_multicollinearity_df(),
            "nonlinear": make_nonlinear_df(),
            "weak_signal": make_weak_signal_df(),
        }

        for name, df in cases.items():
            path = os.path.join(tmp, f"{name}.csv")
            df.to_csv(path, index=False)

            out, err = run_case(path, "Target")
            problem_line = extract_problem_block(out)

            print(f"\n=== CASE: {name.upper()} ===")
            print(f"Detected: {problem_line}")
            print(out[:700])  # short preview
            if err:
                print("ERR:", err[:300])


if __name__ == "__main__":
    main()