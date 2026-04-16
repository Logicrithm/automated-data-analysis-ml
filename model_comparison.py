from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

RANDOM_STATE = 42
MIN_TRAIN_TEST_SAMPLES = 20


def _model_record(
    name: str,
    r2_value: float,
    rmse: float,
    training_time: float,
    interpretability: str,
    recommendation: str,
) -> Dict:
    return {
        "name": name,
        "r2_score": round(float(r2_value), 3),
        "rmse": round(float(rmse), 3),
        "training_time": round(float(training_time), 4),
        "interpretability": interpretability,
        "recommendation": recommendation,
    }


def train_multiple_models(X: pd.DataFrame, y: pd.Series) -> Dict:
    if X.empty or len(X) < MIN_TRAIN_TEST_SAMPLES:
        return {"models": [], "best_model": None}

    # Clean obvious invalids
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if len(X) < MIN_TRAIN_TEST_SAMPLES:
        return {"models": [], "best_model": None}

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model_specs = [
        (
            "Linear Regression",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", LinearRegression()),
                ]
            ),
            "HIGH",
        ),
        (
            "Random Forest",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=200)),
                ]
            ),
            "MEDIUM",
        ),
        (
            "XGBoost",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBRegressor(
                            random_state=RANDOM_STATE,
                            n_estimators=300,
                            learning_rate=0.05,
                            max_depth=6,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="reg:squarederror",
                        ),
                    ),
                ]
            ),
            "LOW",
        ),
    ]

    raw_records: List[Dict] = []
    for name, model, interpretability in model_specs:
        try:
            start = time.perf_counter()
            model.fit(x_train, y_train)
            duration = time.perf_counter() - start
            preds = model.predict(x_test)
            r2_value = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            raw_records.append(
                {
                    "name": name,
                    "r2": r2_value,
                    "rmse": rmse,
                    "training_time": duration,
                    "interpretability": interpretability,
                }
            )
        except Exception:
            continue

    best = max(raw_records, key=lambda item: item["r2"]) if raw_records else None
    best_name = best["name"] if best else None

    records: List[Dict] = []
    for item in raw_records:
        if best_name and item["name"] == best_name:
            recommendation = "BEST - Recommended"
        elif item["r2"] < 0.3:
            recommendation = "BASELINE - Too weak"
        else:
            recommendation = "GOOD - Use if speed matters"

        records.append(
            _model_record(
                item["name"],
                item["r2"],
                item["rmse"],
                item["training_time"],
                item["interpretability"],
                recommendation,
            )
        )

    return {"models": records, "best_model": best_name}