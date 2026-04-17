from __future__ import annotations

from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


class ExperimentPipeline:
    """Run controlled experiments to validate model-improvement recommendations.

    Parameters:
        X: Feature matrix as DataFrame or array-like.
        y: Target values as Series or array-like.
        test_size: Fraction of rows reserved for test evaluation.
        random_state: Random seed used for split/model reproducibility.
    """

    def __init__(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.test_size = test_size

        self.X, self.y = self._prepare_inputs(X, y)
        if self.X.empty:
            raise ValueError("ExperimentPipeline requires at least one usable feature.")
        if len(self.X) < 5:
            raise ValueError("ExperimentPipeline requires at least 5 rows for evaluation.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self.results: Dict[str, Dict] = {}

    def _prepare_inputs(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if isinstance(X, pd.DataFrame):
            feature_df = X.copy()
        else:
            feature_df = pd.DataFrame(X)

        feature_df = pd.get_dummies(feature_df, drop_first=True)
        feature_df = feature_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

        if feature_df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)

        # Median fill handles most missing numeric values; fallback to 0.0 for all-NaN columns.
        median_values = feature_df.median(numeric_only=True)
        feature_df = feature_df.fillna(median_values)
        if feature_df.isna().values.any():
            feature_df = feature_df.fillna(0.0)

        if isinstance(y, pd.Series):
            target = y.copy()
        else:
            target = pd.Series(y)

        target = target.replace([np.inf, -np.inf], np.nan)

        combined = feature_df.copy()
        combined["__target__"] = target
        combined = combined.dropna(subset=["__target__"])

        clean_X = combined.drop(columns=["__target__"])
        clean_y = combined["__target__"]
        return clean_X, clean_y

    def _create_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(random_state=self.random_state)

    def _evaluate_model(
        self,
        method: str,
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> float:
        result: Dict = {
            "method": method,
            "r2": None,
            "cv_r2_mean": None,
            "cv_r2_std": None,
            "metadata": metadata or {},
            "error": None,
        }

        try:
            model = self._create_model()
            model.fit(X_train, self.y_train)
            predictions = model.predict(X_test)
            r2_value = float(r2_score(self.y_test, predictions))

            # Min 2 folds is required for CV; max 5 keeps validation lightweight.
            # Using len(y_train)//2 keeps roughly >=2 samples per fold for stability.
            cv_folds = max(2, min(5, len(self.y_train) // 2))
            if cv_folds >= 2:
                cv_scores = cross_val_score(model, X_train, self.y_train, cv=cv_folds, scoring="r2")
                result["cv_r2_mean"] = float(np.mean(cv_scores))
                result["cv_r2_std"] = float(np.std(cv_scores))

            result["r2"] = r2_value
        except Exception as exc:
            result["error"] = str(exc)
            result["r2"] = float("-inf")

        self.results[method] = result
        return float(result["r2"])

    def run_baseline(self) -> float:
        return self._evaluate_model("baseline", self.X_train, self.X_test)

    def _calculate_vif(self, X_scaled: np.ndarray) -> List[float]:
        vif_scores: List[float] = []
        for i in range(X_scaled.shape[1]):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    vif = float(variance_inflation_factor(X_scaled, i))
            except Exception:
                vif = float("inf")
            vif_scores.append(vif)
        return vif_scores

    def run_remove_vif(self, threshold: float = 5.0) -> float:
        """Drop features with VIF above threshold (default 5.0 indicates high multicollinearity)."""
        metadata: Dict = {"threshold": threshold, "removed_features": [], "vif_scores": {}}
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.X_train)
            vif_scores = self._calculate_vif(X_train_scaled)

            columns = list(self.X_train.columns)
            metadata["vif_scores"] = {columns[i]: float(vif_scores[i]) for i in range(len(columns))}

            keep_columns = [columns[i] for i, vif in enumerate(vif_scores) if vif <= threshold]
            removed = [columns[i] for i, vif in enumerate(vif_scores) if vif > threshold]

            if not keep_columns:
                keep_columns = columns
                removed = []

            metadata["removed_features"] = removed

            X_train_vif = self.X_train[keep_columns]
            X_test_vif = self.X_test[keep_columns]
            return self._evaluate_model("remove_vif", X_train_vif, X_test_vif, metadata)
        except Exception as exc:
            metadata["error"] = str(exc)
            return self._evaluate_model("remove_vif", self.X_train, self.X_test, metadata)

    def run_pca(self, variance_ratio: float = 0.95) -> float:
        """Apply PCA keeping at least `variance_ratio` cumulative explained variance."""
        metadata: Dict = {"variance_ratio": variance_ratio, "n_components": None}
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_test_scaled = scaler.transform(self.X_test)

            pca = PCA(n_components=variance_ratio)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)

            metadata["n_components"] = int(pca.n_components_)
            metadata["explained_variance_ratio"] = float(np.sum(pca.explained_variance_ratio_))

            return self._evaluate_model("pca", X_train_pca, X_test_pca, metadata)
        except Exception as exc:
            metadata["error"] = str(exc)
            return self._evaluate_model("pca", self.X_train, self.X_test, metadata)

    def run_feature_engineering(self) -> float:
        X_train_eng = self.X_train.copy()
        X_test_eng = self.X_test.copy()

        metadata: Dict = {"added_features": []}

        try:
            columns = list(self.X_train.columns)
            if not columns:
                return self._evaluate_model("feature_engineering", X_train_eng, X_test_eng, metadata)

            existing_names = set(X_train_eng.columns)

            base_feature = columns[0]
            squared_name = f"{base_feature}_squared"
            if squared_name in existing_names:
                squared_name = f"{base_feature}_squared_eng"
            X_train_eng[squared_name] = self.X_train[base_feature] ** 2
            X_test_eng[squared_name] = self.X_test[base_feature] ** 2
            metadata["added_features"].append(squared_name)
            existing_names.add(squared_name)

            if len(columns) >= 2:
                interaction_name = f"{columns[0]}_x_{columns[1]}"
                if interaction_name in existing_names:
                    interaction_name = f"{interaction_name}_eng"
                X_train_eng[interaction_name] = self.X_train[columns[0]] * self.X_train[columns[1]]
                X_test_eng[interaction_name] = self.X_test[columns[0]] * self.X_test[columns[1]]
                metadata["added_features"].append(interaction_name)

            return self._evaluate_model("feature_engineering", X_train_eng, X_test_eng, metadata)
        except Exception as exc:
            metadata["error"] = str(exc)
            return self._evaluate_model("feature_engineering", self.X_train, self.X_test, metadata)

    def run_all(self) -> Dict[str, Dict]:
        self.run_baseline()
        self.run_remove_vif()
        self.run_pca()
        self.run_feature_engineering()
        return self.results

    def get_best_result(self) -> Dict:
        if "baseline" not in self.results:
            self.run_baseline()

        scored_methods = {
            method: details
            for method, details in self.results.items()
            if details.get("error") is None and details.get("r2") is not None and np.isfinite(details["r2"])
        }

        if not scored_methods:
            return {
                "method": None,
                "improvement_percent": 0.0,
                "improvement_absolute": 0.0,
                "improvement_basis": "not_available",
                "all_results": self.results,
            }

        best_method = max(scored_methods, key=lambda method: scored_methods[method]["r2"])
        baseline_r2 = self.results.get("baseline", {}).get("r2", float("nan"))
        best_r2 = scored_methods[best_method]["r2"]

        improvement_absolute = float(best_r2 - baseline_r2) if np.isfinite(baseline_r2) else 0.0

        if np.isfinite(baseline_r2) and baseline_r2 > 0:
            improvement_percent = float((improvement_absolute / baseline_r2) * 100)
            improvement_basis = "relative_to_baseline_r2"
        elif np.isfinite(baseline_r2) and baseline_r2 < 0:
            # For negative baseline R², normalize by its magnitude to express recovery.
            improvement_percent = float((improvement_absolute / abs(baseline_r2)) * 100)
            improvement_basis = "relative_to_abs_baseline_r2"
        elif improvement_absolute > 0:
            improvement_percent = float("inf")
            improvement_basis = "improvement_from_zero_baseline"
        else:
            improvement_percent = 0.0
            improvement_basis = "no_improvement"

        return {
            "method": best_method,
            "improvement_percent": improvement_percent,
            "improvement_absolute": improvement_absolute,
            "improvement_basis": improvement_basis,
            "all_results": self.results,
        }

    def get_structured_results(self) -> Dict:
        if not self.results:
            self.run_all()
        best = self.get_best_result()
        return {
            "experiments": self.results,
            "best_method": best["method"],
            "improvement_percent": best["improvement_percent"],
            "improvement_absolute": best["improvement_absolute"],
            "improvement_basis": best["improvement_basis"],
        }

    def print_summary(self) -> None:
        if not self.results:
            self.run_all()

        ordered_methods = ["baseline", "remove_vif", "pca", "feature_engineering"]
        print("\nExperiment Results")
        print("=" * 50)

        for method in ordered_methods:
            result = self.results.get(method)
            if not result:
                continue
            r2_value = result.get("r2")
            label = method.replace("_", " ").title()
            if r2_value is None or not np.isfinite(r2_value):
                print(f"{label:22}: failed ({result.get('error', 'unknown error')})")
                continue
            cv_mean = result.get("cv_r2_mean")
            cv_text = "n/a" if cv_mean is None or not np.isfinite(cv_mean) else f"{cv_mean:.4f}"
            print(f"{label:22}: R²={r2_value:.4f} | CV={cv_text}")

        best = self.get_best_result()
        method = best["method"]
        if method:
            name = method.replace("_", " ").title()
            improvement = best["improvement_percent"]
            if np.isinf(improvement):
                print(f"\nBest method: {name} (improvement from zero baseline)")
            else:
                print(f"\nBest method: {name} ({improvement:.2f}% vs baseline)")
