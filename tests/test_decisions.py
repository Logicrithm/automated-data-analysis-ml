import os
import tempfile
import subprocess
import numpy as np
import pandas as pd

np.random.seed(42)

PROJECT_CMD = ["python", "analyze.py"]

def run_case(csv_path, target="Target"):
    r = subprocess.run(PROJECT_CMD + [csv_path, "./output", target], capture_output=True, text=True)
    return r.stdout

def detect_label(output_text: str) -> str:
    for line in output_text.splitlines():
        if "(" in line and "Severity" in line:
            return line.split("(")[0].strip()
    return "UNKNOWN"

def make_multicollinearity_df(n=300):
    x1 = np.random.normal(0, 1, n)
    x2 = x1 * 0.98 + np.random.normal(0, 0.02, n)
    x3 = np.random.normal(0, 1, n)
    y = 50 + 10 * x1 + np.random.normal(0, 0.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "Target": y})

def make_nonlinear_df(n=300):
    x = np.random.uniform(-3, 3, n)
    z = np.random.normal(0, 1, n)
    y = x**2 + 0.2 * z + np.random.normal(0, 0.3, n)
    return pd.DataFrame({"x": x, "z": z, "Target": y})

def make_weak_signal_df(n=300):
    a = np.random.normal(0, 1, n)
    b = np.random.normal(0, 1, n)
    c = np.random.normal(0, 1, n)
    y = np.random.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "Target": y})

def test_decisions():
    with tempfile.TemporaryDirectory() as tmp:
        cases = [
            ("multi.csv", make_multicollinearity_df(), "MULTICOLLINEARITY"),
            ("nonlin.csv", make_nonlinear_df(), "NON_LINEARITY"),
            ("weak.csv", make_weak_signal_df(), "WEAK_FEATURES"),
        ]
        for fname, df, expected in cases:
            path = os.path.join(tmp, fname)
            df.to_csv(path, index=False)
            out = run_case(path)
            detected = detect_label(out)
            assert detected == expected, f"{fname}: expected {expected}, got {detected}"