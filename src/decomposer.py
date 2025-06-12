"""Time Series Decomposer Module for Renewable Energy Forecasting."""

import logging
import pickle
import warnings

# from abc import Protocol, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from statsmodels.tsa.stattools import adfuller

from logging_config import setup_logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = setup_logging(logger_name="Preprocessor", log_level=logging.DEBUG)


# Core Strategy Interface
class DecompositionStrategy(Protocol):
    """Abstract base class for all decomposition strategies."""

    is_fitted: bool
    params: dict[str, Any]
    decomposition_results: dict[str, pd.Series]

    def decompose(self, series: pd.Series, **params) -> dict[str, pd.Series]:
        """Decompose a single time series."""
        ...

    def get_components(self) -> list[str]:
        """Get list of available components for this decomposition method."""
        ...

    def get_default_params(self, granularity: str, energy_type: str) -> dict[str, Any]:
        """Get default parameters for given granularity and energy type."""
        ...

    def validate_series(self, series: pd.Series) -> bool:
        """Validate if series is suitable for this decomposition method."""
        # TODO: move the logic to the child classes
        return len(series) > 0 and not series.isnull().all()

    def supports_multiple_seasonality(self) -> bool:
        """Check if strategy supports multiple seasonal patterns."""
        # TODO: move the logic to the child classes
        return False


# Concrete Strategy Implementations
class STLStrategy(DecompositionStrategy):
    """STL (Seasonal-Trend decomposition using Loess) implementation."""

    def decompose(self, series: pd.Series, **params) -> dict[str, pd.Series]:
        """Decompose using STL method."""
        try:
            from statsmodels.tsa.seasonal import STL

            # Set default parameters
            stl_params = {
                "seasonal": params.get("seasonal", 24),  # Default daily seasonality
                "trend": params.get("trend"),
                "robust": params.get("robust", False),
            }

            # Remove None values
            stl_params = {k: v for k, v in stl_params.items() if v is not None}

            # Perform decomposition
            stl = STL(series, **stl_params)
            logger.debug(f"Fit STL decomposition for {series.name} ({series.shape})")
            result = stl.fit()

        except ImportError:
            raise ImportError(
                "Statsmodels not installed. Install with: pip install statsmodels",
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"STL decomposition failed: {e!s}")

        else:
            return {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "resid": result.resid,
                "observed": series,
            }

    def get_components(self) -> list[str]:
        """Get STL component names."""
        return ["trend", "seasonal", "resid", "observed"]

    def get_default_params(self, granularity: str, energy_type: str) -> dict[str, Any]:
        """Get default STL parameters."""
        seasonal_defaults = {
            "hourly": 24,  # Daily pattern
            "daily": 7,  # Weekly pattern
            "monthly": 12,  # Yearly pattern
        }

        return {
            "seasonal": seasonal_defaults.get(granularity, 24),
            "robust": energy_type in ["wind"],  # Wind can be more volatile
        }

    def validate_series(self, series: pd.Series) -> bool:
        """Validate series for STL decomposition."""
        # Condition 1
        if not super().validate_series(series):
            return False

        # Condition 2: STL needs at least 2 full seasonal periods
        min_length = 2 * self.params.get("seasonal", 24)
        logger.debug(f"{min_length=} | {len(series)}")

        condition_2 = len(series) >= min_length

        if not condition_2:
            print(
                f"Validation fail because: STL needs at least 2 full seasonal periods ({min_length / 2})",
            )
        return condition_2


class MSTLStrategy(DecompositionStrategy):
    """MSTL (Multiple Seasonal-Trend decomposition using Loess) implementation."""

    def decompose(self, series: pd.Series, **params) -> dict[str, pd.Series]:
        """Decompose using MSTL method."""
        try:
            from statsmodels.tsa.seasonal import MSTL

            # Set default parameters
            mstl_params = {
                "periods": params.get("periods", [24, 168]),  # Daily and weekly
                "windows": params.get("windows"),
                "lmbda": params.get("lmbda"),
            }

            # Remove None values
            mstl_params = {k: v for k, v in mstl_params.items() if v is not None}

            # Perform decomposition
            mstl = MSTL(series, **mstl_params)
            logger.debug(f"Fit MSTL decomposition for {series.name} ({series.shape})")
            result = mstl.fit()

        except ImportError:
            raise ImportError(
                "Statsmodels not installed. Install with: pip install statsmodels",
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"MSTL decomposition failed: {e!s}")
        else:
            return {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "resid": result.resid,
                "observed": series,
            }

    def get_components(self) -> list[str]:
        """Get MSTL component names."""
        return ["trend", "seasonal", "resid", "observed"]

    def get_default_params(self, granularity: str, energy_type: str) -> dict[str, Any]:
        """Get default MSTL parameters."""
        period_defaults = {
            "hourly": {
                "solar": [24, 8760],  # Daily, yearly
                "wind": [24, 168, 8760],  # Daily, weekly, yearly
                "demand": [24, 168],  # Daily, weekly
                "default": [24, 168],  # Daily, weekly
            },
            "daily": {
                "solar": [7, 365],  # Weekly, yearly
                "wind": [7, 365],  # Weekly, yearly
                "demand": [7, 365],  # Weekly, yearly
                "default": [7, 365],  # Weekly, yearly
            },
            "monthly": {
                "default": [12],  # Yearly
            },
        }

        granularity_periods = period_defaults.get(
            granularity,
            period_defaults["hourly"],
        )
        periods = granularity_periods.get(energy_type, granularity_periods["default"])

        return {"periods": periods}

    def supports_multiple_seasonality(self) -> bool:
        """MSTL supports multiple seasonal patterns."""
        return True

    def validate_series(self, series: pd.Series) -> bool:
        """Validate series for MSTL decomposition."""
        if not super().validate_series(series):
            return False

        # MSTL needs at least 2 full cycles of the largest period
        periods = self.params.get("periods", [24, 168])
        max_period = max(periods) if periods else 24
        min_length = 2 * max_period

        logger.debug(f"{len(series)=} {max_period=} {periods=}")
        condition_2 = len(series) >= min_length

        if not condition_2:
            print(
                f"Validation fail because: MSTL needs at least 2 full cycles of the larges period ({max_period})",
            )

        return len(series) >= min_length


class MovingAverageStrategy(DecompositionStrategy):
    """Simple moving average decomposition strategy."""

    def decompose(self, series: pd.Series, **params) -> dict[str, pd.Series]:
        """Decompose using moving average method."""
        window = params.get("window", 24)
        center = params.get("center", True)

        # Calculate trend using moving average
        trend = series.rolling(window=window, center=center).mean()

        # Detrended series
        detrended = series - trend

        # Simple seasonal component (average by period)
        period = params.get("seasonal_period", 24)
        seasonal_pattern = detrended.groupby(detrended.index % period).mean()
        seasonal = pd.Series(index=series.index, dtype=float)

        for i, idx in enumerate(series.index):
            seasonal.iloc[i] = seasonal_pattern.iloc[i % len(seasonal_pattern)]

        # Residual
        residual = series - trend - seasonal

        return {
            "trend": trend,
            "seasonal": seasonal,
            "resid": residual,
            "observed": series,
        }

    def get_components(self) -> list[str]:
        """Get moving average component names."""
        return ["trend", "seasonal", "resid", "observed"]

    def get_default_params(self, granularity: str, energy_type: str) -> dict[str, Any]:
        """Get default moving average parameters."""
        window_defaults = {
            "hourly": 24,  # Daily window
            "daily": 7,  # Weekly window
            "monthly": 12,  # Yearly window
        }

        seasonal_defaults = {
            "hourly": 24,  # Daily seasonality
            "daily": 7,  # Weekly seasonality
            "monthly": 12,  # Yearly seasonality
        }

        return {
            "window": window_defaults.get(granularity, 24),
            "seasonal_period": seasonal_defaults.get(granularity, 24),
            "center": True,
        }


class CustomDecompositionStrategy(DecompositionStrategy):
    """Wrapper for custom user-defined decomposition methods."""

    def __init__(self, decompose_func, components: list[str], **kwargs):
        super().__init__(**kwargs)
        self.decompose_func = decompose_func
        self.components = components

    def decompose(self, series: pd.Series, **params) -> dict[str, pd.Series]:
        """Decompose using custom function."""
        try:
            result = self.decompose_func(series, **params)

            # Ensure result is in correct format
            if not isinstance(result, dict):
                raise ValueError(
                    "Custom decomposition function must return a dictionary",
                )

            # Validate components
            for component in self.components:
                if component not in result:
                    raise ValueError(f"Missing component: {component}")

            return result

        except Exception as e:
            raise RuntimeError(f"Custom decomposition failed: {e!s}")

    def get_components(self) -> list[str]:
        """Get custom component names."""
        return self.components

    def get_default_params(self, granularity: str, energy_type: str) -> dict[str, Any]:
        """Get default parameters for custom decomposition."""
        return self.params


# Configuration Classes
@dataclass
class DecompositionConfig:
    """Configuration for time series decomposition."""

    # Core decomposition parameters
    method: str = "MSTL"  # "STL", "MSTL", "moving_average", "custom"
    granularity: str = "hourly"  # "hourly", "daily", "monthly"
    energy_type: str = "solar"  # "solar", "wind", "demand", "default"

    # Data validation and preprocessing
    min_periods_ratio: float = 2.0  # Minimum data length vs largest period
    fill_method: str = "interpolate"  # "interpolate", "forward", "drop"
    validate_alignment: bool = True  # Check temporal alignment

    # Method-specific parameters (will be merged with defaults)
    method_params: dict[str, Any] = None

    def __post_init__(self):
        """Initialize method parameters if not provided."""
        if self.method_params is None:
            self.method_params = {}


# Context Class - Main Interface
class TimeSeriesDecomposer:
    """Main interface for time-series decomposition with multiple strategies.

    Handles coordinated multi-series decomposition and feature extraction.
    """

    def __init__(self, config: DecompositionConfig = None):  # noqa: D107
        self.config = config or DecompositionConfig()
        self.strategy: DecompositionStrategy | None = None
        self.decomposition_results: dict[str, dict[str, pd.Series]] = {}
        self.is_fitted = False

        # Set default strategy based on config
        self._set_default_strategy()

    def _set_default_strategy(self):
        """Set default strategy based on configuration."""
        strategy_map = {
            "STL": STLStrategy,
            "MSTL": MSTLStrategy,
            "moving_average": MovingAverageStrategy,
        }

        if self.config.method.upper() in strategy_map:
            strategy_class = strategy_map[self.config.method.upper()]
            # Get default parameters and merge with user params
            default_params = strategy_class().get_default_params(
                self.config.granularity,
                self.config.energy_type,
            )
            default_params.update(self.config.method_params)

            self.strategy = strategy_class(**default_params)

    def set_strategy(self, strategy: DecompositionStrategy) -> "TimeSeriesDecomposer":
        """Set the decomposition strategy."""
        self.strategy = strategy
        return self

    def decompose_single(
        self,
        series: pd.Series,
        series_name: str = "series",
    ) -> dict[str, pd.Series]:
        """Decompose a single time series."""
        logger.debug(f"Decomposing {series_name} {series.shape}")
        if not self.strategy:
            raise ValueError("Strategy must be set before decomposition")

        # Validate series
        if not self.strategy.validate_series(series):
            logger.debug(f"{self.strategy.__str__}")
            raise ValueError(f"Series '{series_name}' failed validation")

        # Handle missing values
        if series.isnull().any():
            series = self._handle_missing_values(series)

        # Perform decomposition
        try:
            result = self.strategy.decompose(series, **self.strategy.params)
            self.decomposition_results[series_name] = result

        except Exception as e:
            # Create fallback decomposition
            print(f"Warning: Decomposition failed for {series_name}: {e}")
            fallback_result = self._create_fallback_decomposition(series)
            self.decomposition_results[series_name] = fallback_result
            return fallback_result

        else:
            return result

    def decompose_coordinated(
        self,
        data: dict[str, pd.Series],
        primary_series: str = None,
    ) -> dict[str, dict[str, pd.Series]]:
        """Decompose multiple series with coordinated parameters.

        Args:
            data: Dictionary of series to decompose
            primary_series: Key series that drives parameter selection

        Returns:
            Nested dict: {series_name: {component: series}}
        """
        if not data:
            raise ValueError("No data provided for decomposition")

        # Validate series alignment if required
        if self.config.validate_alignment:
            self._validate_series_alignment(data)

        # Set primary series if not specified
        if primary_series is None:
            primary_series = next(iter(data.keys()))

        if primary_series not in data:
            raise ValueError(f"Primary series '{primary_series}' not found in data")

        # Get optimal parameters from primary series
        primary_params = self._get_optimal_params(data[primary_series])

        # Update strategy parameters
        if self.strategy:
            self.strategy.params.update(primary_params)

        # Decompose all series
        results = {}
        for series_name, series in data.items():
            results[series_name] = self.decompose_single(series, series_name)

        self.is_fitted = True
        return results

    def _validate_series_alignment(self, data: dict[str, pd.Series]):
        """Validate temporal alignment of multiple series."""
        if len(data) <= 1:
            return

        # Get reference index from first series
        first_series = next(iter(data.values()))
        reference_index = first_series.index

        for name, series in data.items():
            # Check index alignment
            if not series.index.equals(reference_index):
                raise ValueError(f"Series '{name}' has misaligned index")

            # Check minimum length requirement
            if self.strategy:
                min_length = self._calculate_min_length()
                if len(series) < min_length:
                    raise ValueError(
                        f"Series '{name}' too short: {len(series)} < {min_length}",
                    )

    def _calculate_min_length(self) -> int:
        """Calculate minimum required series length."""
        if not self.strategy:
            return 0

        # Get the largest period from strategy parameters
        params = self.strategy.params

        if "periods" in params:  # MSTL
            max_period = max(params["periods"])
        elif "seasonal" in params:  # STL
            max_period = params["seasonal"]
        elif "window" in params:  # Moving average
            max_period = params["window"]
        else:
            max_period = 24  # Default

        return int(max_period * self.config.min_periods_ratio)

    def _get_optimal_params(self, series: pd.Series) -> dict[str, Any]:
        """Get optimal parameters based on primary series characteristics."""
        # For now, return default parameters
        # Could be enhanced with automatic parameter selection
        return {}

    def _handle_missing_values(self, series: pd.Series) -> pd.Series:
        """Handle missing values according to configuration."""
        if self.config.fill_method == "interpolate":
            return series.interpolate(method="linear")
        elif self.config.fill_method == "forward":
            return series.fillna(method="ffill")
        elif self.config.fill_method == "drop":
            return series.dropna()
        else:
            return series

    def _create_fallback_decomposition(self, series: pd.Series) -> dict[str, pd.Series]:
        """Create simple fallback decomposition when primary method fails."""
        # Simple moving average fallback
        window = min(24, len(series) // 4)  # Adaptive window size

        trend = series.rolling(window=window, center=True).mean()
        residual = series - trend
        seasonal = pd.Series(0, index=series.index)  # No seasonal component

        return {
            "trend": trend,
            "seasonal": seasonal,
            "resid": residual,
            "observed": series,
        }

    def create_feature_matrix(
        self,
        components: list[str] = None,
        include_lags: bool = False,
        lag_periods: list[int] = None,
    ) -> pd.DataFrame:
        """Create aligned feature matrix from decomposed components.

        Args:
            components: Which components to include (default: all except 'observed')
            include_lags: Whether to include lagged features
            lag_periods: Specific lag periods to include

        Returns:
            DataFrame with columns like 'series_trend', 'series_seasonal', etc.
        """
        if not self.decomposition_results:
            raise ValueError("No decomposition results available")

        # Default components
        if components is None:
            components = ["trend", "seasonal", "resid"]

        feature_dfs = []

        for series_name, decomp_result in self.decomposition_results.items():
            for component in components:
                if component in decomp_result:
                    col_name = f"{series_name}_{component}"
                    feature_series = decomp_result[component].rename(col_name)
                    feature_dfs.append(feature_series)

                    # Add lagged features if requested
                    if include_lags and lag_periods:
                        for lag in lag_periods:
                            lag_name = f"{col_name}_lag_{lag}"
                            lag_series = feature_series.shift(lag).rename(lag_name)
                            feature_dfs.append(lag_series)

        if not feature_dfs:
            raise ValueError("No valid features could be created")

        return pd.concat(feature_dfs, axis=1)

    def get_component_summary(self) -> pd.DataFrame:
        """Get summary statistics for all decomposed components."""
        if not self.decomposition_results:
            raise ValueError("No decomposition results available")

        summary_data = []

        for series_name, decomp_result in self.decomposition_results.items():
            for component, component_series in decomp_result.items():
                if component != "observed":  # Skip original series
                    summary_data.append(
                        {
                            "series": series_name,
                            "component": component,
                            "mean": component_series.mean(),
                            "std": component_series.std(),
                            "min": component_series.min(),
                            "max": component_series.max(),
                            "variance_explained": (
                                component_series.var() / decomp_result["observed"].var()
                            )
                            * 100,
                        },
                    )

        return pd.DataFrame(summary_data)


# Factory for easy strategy creation
class DecompositionFactory:
    """Factory class for creating decomposition strategies."""

    @staticmethod
    def create_strategy(
        method: str,
        granularity: str = "hourly",
        energy_type: str = "solar",
        **params,
    ) -> DecompositionStrategy:
        """Create a strategy based on method type."""
        strategies = {
            "stl": STLStrategy,
            "mstl": MSTLStrategy,
            "moving_average": MovingAverageStrategy,
        }

        method_lower = method.lower()
        if method_lower not in strategies:
            available = ", ".join(strategies.keys())
            raise ValueError(f"Unknown method: {method}. Available: {available}")

        strategy_class = strategies[method_lower]

        # Get default parameters and merge with user params
        default_params = strategy_class().get_default_params(granularity, energy_type)
        default_params.update(params)

        return strategy_class(**default_params)

    @staticmethod
    def create_custom_strategy(
        decompose_func,
        components: list[str],
        **params,
    ) -> CustomDecompositionStrategy:
        """Create a custom strategy with user-provided decomposition function."""
        return CustomDecompositionStrategy(decompose_func, components, **params)


# Convenience factory functions (maintaining compatibility)
def create_solar_decomposer(granularity: str = "hourly") -> TimeSeriesDecomposer:
    """Create decomposer optimized for solar energy analysis."""
    config = DecompositionConfig(
        method="MSTL",
        granularity=granularity,
        energy_type="solar",
    )
    return TimeSeriesDecomposer(config)


def create_wind_decomposer(granularity: str = "hourly") -> TimeSeriesDecomposer:
    """Create decomposer optimized for wind energy analysis."""
    config = DecompositionConfig(
        method="MSTL",
        granularity=granularity,
        energy_type="wind",
    )
    return TimeSeriesDecomposer(config)


def create_demand_decomposer(granularity: str = "hourly") -> TimeSeriesDecomposer:
    """Create decomposer optimized for energy demand analysis."""
    config = DecompositionConfig(
        method="MSTL",
        granularity=granularity,
        energy_type="demand",
    )
    return TimeSeriesDecomposer(config)


def check_stationarity(series: pd.Series, significance_level: float = 0.05):
    """Performs the ADF test and prints a summary."""
    result = adfuller(series.dropna())
    p_value = result[1]

    print("--- ADF Test for Stationarity ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value <= significance_level:
        print("✅ Result: Residuals are likely stationary (reject H0).")
    else:
        print("❌ Result: Residuals are likely non-stationary (fail to reject H0).")


# Example usage and testing
if __name__ == "__main__":
    # Paths
    data_path = Path().cwd() / "data" / "2_decomposer"
    input_files = data_path / "input"
    output_files = data_path / "output"

    hourly_generation = pd.read_csv(input_files / "hourly_generation.csv")

    def export_to_pkl(data, file_path):
        """Export data to a .pkl file using pathlib."""
        path = Path(file_path)

        # Ensure the directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:
            pickle.dump(data, f)

        logger.info(f"Data successfully exported to {path.absolute()}")

    def import_from_pkl(file_path):
        """Import data from a .pkl file using pathlib."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"The file {path} does not exist")

        with path.open("rb") as f:
            data = pickle.load(f)  # noqa: S301

        logger.info(f"Data successfully imported from {path.absolute()}")
        return data

    solar_series = (
        hourly_generation["solar"]
        if isinstance(hourly_generation["solar"], pd.Series)
        else None
    )
    wind_series = hourly_generation["eolica"]

    # Create decomposer with MSTL strategy
    solar_decomposer = create_solar_decomposer("hourly")
    solar_result = solar_decomposer.decompose_single(solar_series, "solar")

    solar_items = list(solar_result.values())

    solar_decomposed = pd.concat(solar_items, axis=1)
    solar_decomposed.to_csv(output_files / "solar_decomposed.csv")

    print("Components available:", list(solar_result.keys()))
    for component, series in solar_result.items():
        if component != "observed":
            print(
                f"{component.capitalize()} - Mean: {series.mean():.2f}, Std: {series.std():.2f}",
            )
