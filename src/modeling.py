"""Time Series Modeling Module for Renewable Energy Forecasting.

Extensible module using Strategy Pattern for multiple ML models
supporting time-series specific operations, cross-validation, and interpretability.
"""

import warnings

# from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")


# Core Strategy Interface
class ModelStrategy(Protocol):
    """Abstract base class for all modeling strategies."""

    model: Any
    is_fitted: bool
    scaler: Any
    feature_names: list[str] | None
    params: dict[str, Any]

    def build_model(self, **params) -> None:
        """Build the model with given parameters."""
        ...

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelStrategy":
        """Fit the model to training data."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        ...

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance if available."""
        ...

    def requires_scaling(self) -> bool:
        """Check if model requires feature scaling."""
        ...

    def supports_shap(self) -> bool:
        """Check if model supports SHAP analysis."""
        ...


# Concrete Strategy Implementations
class XGBoostStrategy(ModelStrategy):
    """XGBoost implementation for time-series forecasting."""

    def build_model(self, **params):
        try:
            from xgboost import XGBRegressor

            default_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
            }
            default_params.update(params)
            self.model = XGBRegressor(**default_params)
        except ImportError:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost",
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostStrategy":
        if self.model is None:
            self.build_model(**self.params)

        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def get_feature_importance(self) -> dict[str, float] | None:
        if not self.is_fitted or self.feature_names is None:
            return None
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

    def requires_scaling(self) -> bool:
        return False  # Tree-based models don't require scaling


class LightGBMStrategy(ModelStrategy):
    """LightGBM implementation for time-series forecasting."""

    def build_model(self, **params):
        try:
            from lightgbm import LGBMRegressor

            default_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
                "verbose": -1,
            }
            default_params.update(params)
            self.model = LGBMRegressor(**default_params)
        except ImportError:
            raise ImportError(
                "LightGBM not installed. Install with: pip install lightgbm",
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMStrategy":
        if self.model is None:
            self.build_model(**self.params)

        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def get_feature_importance(self) -> dict[str, float] | None:
        if not self.is_fitted or self.feature_names is None:
            return None
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

    def requires_scaling(self) -> bool:
        return False


class CatBoostStrategy(ModelStrategy):
    """CatBoost implementation for time-series forecasting."""

    def build_model(self, **params):
        try:
            from catboost import CatBoostRegressor

            default_params = {
                "iterations": 100,
                "depth": 6,
                "learning_rate": 0.1,
                "random_seed": 42,
                "verbose": False,
            }
            default_params.update(params)
            self.model = CatBoostRegressor(**default_params)
        except ImportError:
            raise ImportError(
                "CatBoost not installed. Install with: pip install catboost",
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostStrategy":
        if self.model is None:
            self.build_model(**self.params)

        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def get_feature_importance(self) -> dict[str, float] | None:
        if not self.is_fitted or self.feature_names is None:
            return None
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

    def requires_scaling(self) -> bool:
        return False


class HoltWintersStrategy(ModelStrategy):
    """Holt-Winters implementation for time-series forecasting."""

    def build_model(self, **params):
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            self.model_class = ExponentialSmoothing
            default_params = {
                "trend": "add",
                "seasonal": "add",
                "seasonal_periods": 24,  # Default for hourly data
            }
            default_params.update(params)
            self.model_params = default_params
        except ImportError:
            raise ImportError(
                "Statsmodels not installed. Install with: pip install statsmodels",
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HoltWintersStrategy":
        # Holt-Winters uses only the target variable
        self.model = self.model_class(y, **self.model_params).fit()
        self.is_fitted = True
        self.forecast_steps = len(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        # For Holt-Winters, predict based on the number of rows in X
        forecast = self.model.forecast(steps=len(X))
        return np.array(forecast)

    def get_feature_importance(self) -> dict[str, float] | None:
        return None  # Holt-Winters doesn't provide feature importance

    def requires_scaling(self) -> bool:
        return False

    def supports_shap(self) -> bool:
        return False


class ARIMAStrategy(ModelStrategy):
    """ARIMA implementation for time-series forecasting."""

    def build_model(self, **params):
        try:
            from statsmodels.tsa.arima.model import ARIMA

            self.model_class = ARIMA
            default_params = {
                "order": (1, 1, 1),  # (p, d, q)
            }
            default_params.update(params)
            self.model_params = default_params
        except ImportError:
            raise ImportError(
                "Statsmodels not installed. Install with: pip install statsmodels",
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ARIMAStrategy":
        self.model = self.model_class(y, **self.model_params).fit()
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        forecast = self.model.forecast(steps=len(X))
        return np.array(forecast)

    def get_feature_importance(self) -> dict[str, float] | None:
        return None

    def requires_scaling(self) -> bool:
        return False

    def supports_shap(self) -> bool:
        return False


class CustomModelStrategy(ModelStrategy):
    """Wrapper for custom user-defined models."""

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.custom_model = model
        self.model = model

    def build_model(self, **params):
        # Custom model is already provided
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CustomModelStrategy":
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def get_feature_importance(self) -> dict[str, float] | None:
        if hasattr(self.model, "feature_importances_") and self.feature_names:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, "coef_") and self.feature_names:
            return dict(zip(self.feature_names, np.abs(self.model.coef_)))
        return None


# Context Class - Main Interface
class TimeSeriesModeler:
    """Main interface for time-series modeling with multiple strategies.

    Handles preprocessing, model selection, cross-validation, and interpretability.
    """

    def __init__(self):
        self.strategy: ModelStrategy | None = None
        self.scaler: StandardScaler | MinMaxScaler | None = None
        self.is_fitted = False
        self.cv_results: dict | None = None
        self.has_cyclical_features = False

    def set_strategy(self, strategy: ModelStrategy) -> "TimeSeriesModeler":
        """Set the modeling strategy."""
        self.strategy = strategy
        return self

    def detect_cyclical_features(self, X: pd.DataFrame) -> bool:
        """Detect if dataset contains cyclical encoded features.

        Look for common patterns like sin/cos encodings.
        """
        cyclical_patterns = ["_sin", "_cos", "sin_", "cos_", "cyclical"]
        cyclical_cols = [
            col
            for col in X.columns
            if any(pattern in col.lower() for pattern in cyclical_patterns)
        ]

        # Also check for values in [-1, 1] range which might indicate cyclical encoding
        potential_cyclical = []
        for col in X.columns:
            if X[col].dtype in ["float64", "float32"]:
                if X[col].min() >= -1.1 and X[col].max() <= 1.1:
                    potential_cyclical.append(col)

        self.has_cyclical_features = (
            len(cyclical_cols) > 0 or len(potential_cyclical) > len(X.columns) * 0.3
        )
        return self.has_cyclical_features

    def preprocess_data(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        fit_scaler: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Preprocess data with intelligent scaling detection."""
        X_processed = X.copy()

        # Detect cyclical features
        self.detect_cyclical_features(X)

        # Apply scaling only if needed and no cyclical features detected
        if (
            self.strategy
            and self.strategy.requires_scaling()
            and not self.has_cyclical_features
        ):
            if fit_scaler:
                self.scaler = StandardScaler()
                X_processed = pd.DataFrame(
                    self.scaler.fit_transform(X_processed),
                    columns=X.columns,
                    index=X.index,
                )
            elif self.scaler:
                X_processed = pd.DataFrame(
                    self.scaler.transform(X_processed),
                    columns=X.columns,
                    index=X.index,
                )

        return X_processed, y

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TimeSeriesModeler":
        """Fit the model with preprocessing."""
        if not self.strategy:
            raise ValueError("Strategy must be set before fitting")

        X_processed, y_processed = self.preprocess_data(X, y, fit_scaler=True)
        self.strategy.fit(X_processed, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with preprocessing."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_processed, _ = self.preprocess_data(X, fit_scaler=False)
        return self.strategy.predict(X_processed)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        test_size: int = None,
    ) -> dict[str, list[float]]:
        """Perform time-series cross-validation."""
        if not self.strategy:
            raise ValueError("Strategy must be set before cross-validation")

        # Use appropriate test_size for time series
        if test_size is None:
            test_size = len(X) // (cv_folds + 1)

        tscv = TimeSeriesSplit(n_splits=cv_folds, test_size=test_size)

        scores = {"mse": [], "mae": [], "r2": []}

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Create a new strategy instance for each fold
            fold_strategy = self.strategy.__class__(**self.strategy.params)
            fold_modeler = TimeSeriesModeler().set_strategy(fold_strategy)

            # Fit and predict
            fold_modeler.fit(X_train, y_train)
            y_pred = fold_modeler.predict(X_test)

            # Calculate metrics
            scores["mse"].append(mean_squared_error(y_test, y_pred))
            scores["mae"].append(mean_absolute_error(y_test, y_pred))
            scores["r2"].append(r2_score(y_test, y_pred))

        self.cv_results = {
            metric: {"mean": np.mean(values), "std": np.std(values), "scores": values}
            for metric, values in scores.items()
        }

        return self.cv_results

    def grid_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict[str, list],
        cv_folds: int = 3,
    ) -> dict[str, Any]:
        """Perform grid search with time-series cross-validation."""
        if not self.strategy:
            raise ValueError("Strategy must be set before grid search")

        from itertools import product

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        best_score = float("inf")
        best_params = None
        results = []

        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))

            # Create strategy with current parameters
            strategy = self.strategy.__class__(**params)
            modeler = TimeSeriesModeler().set_strategy(strategy)

            # Perform cross-validation
            cv_results = modeler.cross_validate(X, y, cv_folds=cv_folds)
            mean_mse = cv_results["mse"]["mean"]

            results.append(
                {
                    "params": params,
                    "mse_mean": mean_mse,
                    "mse_std": cv_results["mse"]["std"],
                    "mae_mean": cv_results["mae"]["mean"],
                    "r2_mean": cv_results["r2"]["mean"],
                },
            )

            if mean_mse < best_score:
                best_score = mean_mse
                best_params = params

        # Refit with best parameters
        if best_params:
            best_strategy = self.strategy.__class__(**best_params)
            self.set_strategy(best_strategy)
            self.fit(X, y)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "cv_results": results,
        }

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance from the current strategy."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        return self.strategy.get_feature_importance()

    def get_shap_values(self, X: pd.DataFrame, max_display: int = 10):
        """Calculate SHAP values for model interpretability."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to calculate SHAP values")

        if not self.strategy.supports_shap():
            raise ValueError("Current strategy does not support SHAP analysis")

        try:
            import shap

            X_processed, _ = self.preprocess_data(X, fit_scaler=False)

            # Create explainer based on model type
            if hasattr(self.strategy.model, "predict"):
                explainer = shap.Explainer(self.strategy.model, X_processed)
                shap_values = explainer(X_processed)

                return {
                    "shap_values": shap_values,
                    "feature_names": X_processed.columns.tolist(),
                    "expected_value": explainer.expected_value,
                }
            else:
                raise ValueError("Model does not support SHAP analysis")

        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")


class ModelFactory:
    """Factory class for creating model strategies."""

    @staticmethod
    def create_strategy(model_type: str, **params) -> ModelStrategy:
        """Create a strategy based on model type."""
        strategies = {
            "xgboost": XGBoostStrategy,
            "lightgbm": LightGBMStrategy,
            "catboost": CatBoostStrategy,
            "holt_winters": HoltWintersStrategy,
            "arima": ARIMAStrategy,
        }

        if model_type.lower() not in strategies:
            available = ", ".join(strategies.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. Available: {available}",
            )

        return strategies[model_type.lower()](**params)

    @staticmethod
    def create_custom_strategy(model, **params) -> CustomModelStrategy:
        """Create a custom strategy with user-provided model."""
        return CustomModelStrategy(model, **params)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    print("Time Series Modeling Module - Example Usage")
    print("=" * 50)

    # Generate sample time-series data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=1000, freq="H")

    # Create features with cyclical encoding
    df = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * dates.hour / 24),
            "hour_cos": np.cos(2 * np.pi * dates.hour / 24),
            "day_sin": np.sin(2 * np.pi * dates.day / 31),
            "temperature": 20
            + 10 * np.sin(2 * np.pi * dates.dayofyear / 365)
            + np.random.normal(0, 2, 1000),
            "lag_1": np.random.randn(1000),
        },
        index=dates,
    )

    # Target variable (renewable energy production)
    target = (
        50
        + 30 * df["hour_sin"]
        + 20 * df["temperature"] / 20
        + 10 * df["day_sin"]
        + np.random.normal(0, 5, 1000)
    )

    # Split data
    split_idx = int(0.8 * len(df))
    X_train, X_test = df[:split_idx], df[split_idx:]
    y_train, y_test = target[:split_idx], target[split_idx:]

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Example 1: XGBoost with cross-validation
    print("\n1. XGBoost Strategy with Cross-Validation")
    print("-" * 40)

    modeler = TimeSeriesModeler()
    xgb_strategy = ModelFactory.create_strategy("xgboost", n_estimators=50)
    modeler.set_strategy(xgb_strategy)

    # Fit and predict
    modeler.fit(X_train, y_train)
    predictions = modeler.predict(X_test)

    # Cross-validation
    cv_results = modeler.cross_validate(X_train, y_train, cv_folds=3)
    print(f"CV MSE: {cv_results['mse']['mean']:.4f} ± {cv_results['mse']['std']:.4f}")

    # Feature importance
    importance = modeler.get_feature_importance()
    if importance:
        print("Feature Importance:")
        for feature, imp in sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"  {feature}: {imp:.4f}")

    print(f"\nTest MSE: {mean_squared_error(y_test, predictions):.4f}")
    print(f"Cyclical features detected: {modeler.has_cyclical_features}")
