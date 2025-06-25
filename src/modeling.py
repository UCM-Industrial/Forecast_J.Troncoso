"""Time Series Modeling Module for Renewable Energy Forecasting.

Clean Strategy Pattern implementation with base classes for different model families.
"""

import warnings
from typing import Any, Protocol

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class ModelStrategy(Protocol):
    """Protocol for all modeling strategies."""

    model: Any
    is_fitted: bool
    feature_names: list[str] | None
    params: dict[str, Any]
    model_class: type

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelStrategy": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def get_feature_importance(self) -> dict[str, float] | None: ...

    def requires_scaling(self) -> bool: ...

    def supports_shap(self) -> bool: ...


class TreeBasedStrategy(ModelStrategy):
    """Base strategy for tree-based models (XGBoost, LightGBM, CatBoost)."""

    def __init__(self, model_class: type, **params: Any) -> None:
        """Initialize the tree-based strategy.

        Args:
            model_class: The tree-based model class (e.g., xgboost.XGBRegressor).
            **params: Model-specific parameters.
        """
        self.model_class = model_class
        self.params = params or {"random_state": 12}
        self.model = self.model_class(**self.params)
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TreeBasedStrategy":
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def get_feature_importance(self) -> dict[str, float] | None:
        if not self.is_fitted or not self.feature_names:
            return None
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

    def requires_scaling(self) -> bool:
        return False

    def supports_shap(self) -> bool:
        return True


class LinearModelStrategy(ModelStrategy):
    """Base strategy for linear models that require scaling."""

    def __init__(self, model_class, **params) -> None:
        """Initialize the linear model strategy.

        Args:
            model_class: The linear model class (e.g., sklearn.linear_model.Ridge).
            **params: Model-specific parameters.
        """
        self.model_class = model_class
        self.params = params or {}
        self.model = self.model_class(**self.params)
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearModelStrategy":
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def get_feature_importance(self) -> dict[str, float] | None:
        if not self.is_fitted or not self.feature_names:
            return None
        importance = np.abs(self.model.coef_)
        return dict(zip(self.feature_names, importance))

    def requires_scaling(self) -> bool:
        return True

    def supports_shap(self) -> bool:
        return True


class UnivariateTimeSeriesStrategy(ModelStrategy):
    """Base strategy for univariate time series models (Holt-Winters, ARIMA)."""

    def __init__(self, model_class, **params) -> None:
        """Initialize the univariate strategy.

        Args:
            model_class: The model class (e.g., statsmodels.tsa.holtwinters.ExponentialSmoothing).
            **params: Model-specific parameters.
        """
        self.model_class = model_class
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.params = params or {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "UnivariateTimeSeriesStrategy":
        self.model = self.model_class(y, **self.params).fit()
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
    """Strategy for custom user-provided models."""

    def __init__(self, model, **params) -> None:
        """Initialize with custom model.

        Args:
            model: Pre-instantiated model object.
            **params: Additional parameters (unused but kept for compatibility).
        """
        self.model = model
        self.is_fitted = False
        self.feature_names = None

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
        if not self.is_fitted or not self.feature_names:
            return None
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        elif hasattr(self.model, "coef_"):
            return dict(zip(self.feature_names, np.abs(self.model.coef_)))
        return None

    def requires_scaling(self) -> bool:
        # Conservative approach - assume scaling needed unless it's a tree model
        model_name = self.model.__class__.__name__.lower()
        tree_models = ["xgb", "lgb", "catboost", "randomforest", "extratrees"]
        return not any(tree in model_name for tree in tree_models)

    def supports_shap(self) -> bool:
        return hasattr(self.model, "predict")


class ModelFactory:
    """Factory class for creating model strategies."""

    _strategies = {  # noqa: RUF012
        "xgboost": ("xgboost", "XGBRegressor", TreeBasedStrategy),
        "lightgbm": ("lightgbm", "LGBMRegressor", TreeBasedStrategy),
        "catboost": ("catboost", "CatBoostRegressor", TreeBasedStrategy),
        "ridge": ("sklearn.linear_model", "Ridge", LinearModelStrategy),
        "lasso": ("sklearn.linear_model", "Lasso", LinearModelStrategy),
        "elasticnet": ("sklearn.linear_model", "ElasticNet", LinearModelStrategy),
        "holt-winters": (
            "statsmodels.tsa.holtwinters",
            "ExponentialSmoothing",
            UnivariateTimeSeriesStrategy,
        ),
        "arima": ("statsmodels.tsa.arima.model", "ARIMA", UnivariateTimeSeriesStrategy),
    }

    @staticmethod
    def _import_class(module_name: str, class_name: str) -> type:
        """Dynamically import a class."""
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError:
            raise ImportError(
                f"{module_name.split('.')[0]} not installed. Please install it.",
            )

    @classmethod
    def create_strategy(cls, model_type: str, **params: Any) -> ModelStrategy:
        """Create a strategy based on model type."""
        model_type = model_type.lower().replace(" ", "-")
        if model_type not in cls._strategies:
            raise ValueError(
                f"Unknown model type: {model_type}. Available: {', '.join(cls._strategies.keys())}",
            )

        module_name, class_name, strategy_class = cls._strategies[model_type]
        model_class = cls._import_class(module_name, class_name)

        # Default params for specific models
        if model_type == "catboost" and "verbose" not in params:
            params["verbose"] = False
        elif model_type == "holt-winters":
            defaults = {"trend": "add", "seasonal": "add", "seasonal_periods": 12}
            defaults.update(params)
            params = defaults
        elif model_type == "arima":
            defaults = {"order": (1, 1, 1)}
            defaults.update(params)
            params = defaults

        return strategy_class(model_class, **params)


class TimeSeriesModeler:
    """Main interface for time-series modeling with strategy pattern."""

    def __init__(self, strategy: ModelStrategy | None = None) -> None:
        """Initialize modeler with optional strategy."""
        self.strategy = strategy
        self.scaler = None
        self.is_fitted: bool = False
        self.cv_results: dict[str, Any] | None = None

    # def set_strategy(self, strategy: ModelStrategy) -> "TimeSeriesModeler":
    #     """Set the modeling strategy."""
    #     self.strategy = strategy
    #     self.is_fitted = False  # Reset fitted status when strategy changes
    #     return self

    def _preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        fit_scaler: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Preprocess data with optional scaling."""
        if not self.strategy:
            raise ValueError("Strategy must be set before preprocessing")

        X_processed = X.copy()

        # Apply scaling if needed
        if self.strategy.requires_scaling():
            if fit_scaler:
                self.scaler = StandardScaler()
                X_processed = pd.DataFrame(
                    self.scaler.fit_transform(X_processed),
                    columns=X.columns,
                    index=X.index,
                )
            elif self.scaler is not None:
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

        X_processed, _ = self._preprocess(X, y, fit_scaler=True)
        self.strategy.fit(X_processed, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with preprocessing."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_processed, _ = self._preprocess(X, fit_scaler=False)
        return self.strategy.predict(X_processed)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
    ) -> dict[str, dict[str, float]]:
        """Perform time-series cross-validation."""
        if not self.strategy:
            raise ValueError("Strategy must be set before cross-validation")

        # test_size = len(X) // (cv_folds + 1)
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        scores = {"mse": [], "mae": [], "r2": []}

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Create new modeler for this fold
            # fold_modeler = TimeSeriesModeler(
            #     self.strategy.__class__(
            #         self.strategy.model_class
            #         if hasattr(self.strategy, "model_class")
            #         else self.strategy.model.__class__,
            #         **getattr(self.strategy, "params", {}),
            #     ),
            # )

            # fold_modeler.fit(X_train, y_train)
            # y_pred = fold_modeler.predict(X_test)
            fold_strategy = self.strategy.__class__(
                self.strategy.model_class,
                **self.strategy.params,
            )
            fold_modeler = TimeSeriesModeler(fold_strategy).fit(X_train, y_train)
            y_pred = fold_modeler.predict(X_test)

            scores["mse"].append(mean_squared_error(y_test, y_pred))
            scores["mae"].append(mean_absolute_error(y_test, y_pred))
            scores["r2"].append(r2_score(y_test, y_pred))

        self.cv_results = {
            "model_params": self.strategy.params,
            "cv_folds": cv_folds,
            "mae": np.mean(scores["mae"]),
            "mse": np.mean(scores["mse"]),
            "r2": np.mean(scores["r2"]),
            "mae_std": np.std(scores["mae"]),
        }

        return self.cv_results

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance from the current strategy."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        if not self.strategy:
            raise ValueError("First define a strategy model")

        return self.strategy.get_feature_importance()

    # def get_shap_values(self, X: pd.DataFrame) -> dict[str, Any]:
    #     """Calculate SHAP values for model interpretability."""
    #     if not self.is_fitted:
    #         raise ValueError("Model must be fitted to calculate SHAP values")
    #
    #     if not self.strategy.supports_shap():
    #         raise ValueError("Current strategy does not support SHAP analysis")
    #
    #     try:
    #         import shap
    #
    #         X_processed, _ = self._preprocess(X, fit_scaler=False)
    #
    #         explainer = shap.Explainer(self.strategy.model, X_processed)
    #         shap_values = explainer(X_processed)
    #
    #         return {
    #             "shap_values": shap_values,
    #             "feature_names": X_processed.columns.tolist(),
    #             "expected_value": explainer.expected_value,
    #         }
    #     except ImportError:
    #         raise ImportError("SHAP not installed. Install with: pip install shap")


# Convenience functions for quick model creation
# def create_modeler(model_type: str, **params) -> TimeSeriesModeler:
#     """Create a TimeSeriesModeler with specified model type."""
#     strategy = ModelFactory.create_strategy(model_type, **params)
#     return TimeSeriesModeler(strategy)


# --- Plot functions ---
def create_forecast_vs_actual(
    df: pd.DataFrame,
    # datetime_col: str,
    actual_col: str,
    predicted_col: str,
    title: str = "Predicted vs Actual",
) -> go.Figure:
    """Plot actual vs predicted values using Plotly.

    Parameters:
    - df: DataFrame containing datetime, actual, and predicted columns
    - datetime_col: Name of the datetime column
    - actual_col: Name of the actual (true) values column
    - predicted_col: Name of the predicted values column
    - title: Title of the chart
    - height: Height of the figure
    - width: Width of the figure

    Returns:
    - Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[actual_col],
            mode="lines",
            name="Actual",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[predicted_col],
            mode="lines",
            name="Predicted",
        ),
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        # legend=dict(x=0.01, y=0.99),
        # height=height,
        # width=width,
        # template="plotly_dark",
    )

    return fig


def plot_feature_importance(
    importance_dict: dict[str, float],
    top_n: int = 20,
    title: str = "Feature Importance",
):
    """Plots feature importance from a dictionary using Plotly.

    Parameters:
    - importance_dict: Dict of feature importances {feature_name: importance_score}.
    - top_n: Number of top features to show.
    - title: Title of the plot.

    Returns:
    - fig: Plotly Figure.
    """
    df = pd.DataFrame(list(importance_dict.items()), columns=["Feature", "Importance"])
    df = df.sort_values("Importance", ascending=False).head(top_n)

    fig = px.bar(
        df[::-1],  # Reverse to show highest on top
        x="Importance",
        y="Feature",
        orientation="h",
        title=title,
        labels={"Importance": "Importance Score", "Feature": "Feature"},
        height=400 + 20 * top_n,
    )
    fig.update_layout(template="plotly_white")
    return fig


if __name__ == "__main__":
    ...
