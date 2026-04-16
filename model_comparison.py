from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

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

    # Guard against runtime failures on messy datasets by sanitizing invalid rows/values.
    clean_X = X.replace([np.inf, -np.inf], np.nan).copy()
    clean_y = y.replace([np.inf, -np.inf], np.nan)
    valid_target_mask = clean_y.notna()
    clean_X = clean_X.loc[valid_target_mask]
    clean_y = clean_y.loc[valid_target_mask]
    if clean_X.empty or len(clean_X) < MIN_TRAIN_TEST_SAMPLES:
        return {"models": [], "best_model": None}

    if clean_X.isna().values.any():
        # Caller already passes numeric features, so median imputation is appropriate here.
        clean_X = clean_X.fillna(clean_X.median(numeric_only=True)).fillna(0.0)

    x_train, x_test, y_train, y_test = train_test_split(
        clean_X, clean_y, test_size=0.2, random_state=RANDOM_STATE
    )
    model_specs = [
        ("Linear Regression", LinearRegression(), "HIGH"),
        ("Random Forest", RandomForestRegressor(random_state=RANDOM_STATE), "MEDIUM"),
    ]
    if XGBRegressor is not None:
        model_specs.append(
            (
                "XGBoost",
                XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbosity=0,
                ),
                "MEDIUM",
            )
        )
    else:
        model_specs.append(
            ("Gradient Boosting", GradientBoostingRegressor(random_state=RANDOM_STATE), "MEDIUM")
        )

    raw_records: List[Dict] = []
    for name, model, interpretability in model_specs:
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
