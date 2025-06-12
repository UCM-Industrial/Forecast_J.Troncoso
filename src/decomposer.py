import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL, STL

from .logging_config import setup_logging

logger = setup_logging(logger_name="Decomposer", log_level=logging.DEBUG)


class Decomposer(ABC):
    @abstractmethod
    def decompose_coordinated(
        self,
        data: dict[str, pd.Series],
        primary_series: str,
        **kwargs,
    ) -> dict[str, dict[str, pd.Series]]:
        """Decompose the time series into trend, seasonal, residual."""
        ...


@dataclass
class DecompositionConfig:
    """Configuration for coordinated decomposition across multiple series."""

    # Core decomposition parameters
    method: str = "MSTL"  # "STL" or "MSTL"
    periods: list[int] | None = None  # For MSTL: [24, 168] for hourly data
    seasonal: int | None = None  # For STL

    # Granularity and energy-specific settings
    granularity: str = "hourly"
    energy_type: str = "solar"

    # Alignment and validation
    min_periods_ratio: float = 2.0  # Minimum data length vs largest period
    fill_method: str = "interpolate"  # How to handle missing values

    def __post_init__(self) -> None:
        """Set default periods based on granularity if not provided."""
        if self.periods is None:
            self.periods = self._get_default_periods()
            logger.debug(
                f"Set default periods to {self.periods} for granularity={self.granularity}",
            )

    def _get_default_periods(self) -> list[int]:
        """Get default seasonal periods based on granularity."""
        logger.debug(f"Setting default periods for granularity: {self.granularity}")

        defaults = {
            "hourly": [24, 168],  # Daily, weekly
            "daily": [7, 365],  # Weekly, yearly
            "monthly": [12],  # Yearly
        }
        return defaults.get(self.granularity, [24, 168])


class CoordinatedDecomposer(Decomposer):
    """Performs coordinated decomposition of multiple energy series.

    Key features:
    - Ensures consistent decomposition parameters across related series
    - Validates temporal alignment and data quality
    - Provides energy-specific presets
    - Enables synchronized feature extraction for training
    """

    def __init__(self, config: DecompositionConfig) -> None:  # noqa: D107
        self.config = config
        logger.info(
            f"Initializing decomposer with method={config.method}, "
            f"granularity={config.granularity}, energy_type={config.energy_type}",
        )
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.method not in ["STL", "MSTL"]:
            raise ValueError("Method must be 'STL' or 'MSTL'")

        if self.config.method == "STL" and len(self.config.periods) > 1:
            raise ValueError("STL supports only single seasonal period")

    def decompose_coordinated(
        self,
        data: dict[str, pd.Series],
        primary_series: str,
        **kwargs,
    ) -> dict[str, dict[str, pd.Series]]:
        """Decompose multiple series with coordinated parameters.

        Args:
            data: Dictionary of series to decompose
            primary_series: Key series that drives parameter selection
            **kwargs: Additional decomposition parameters

        Returns:
            Nested dict: {series_name: {component: series}}

        """
        logger.info(
            f"Starting coordinated decomposition for {len(data)} series. Primary: {primary_series}",
        )
        # Validate input data alignment
        self._validate_series_alignment(data)

        # Get decomposition parameters from primary series
        primary_params = self._get_optimal_params(data[primary_series], **kwargs)
        logger.debug(f"Using decomposition parameters: {primary_params}")

        # Decompose all series with consistent parameters
        results = {}
        for series_name, series in data.items():
            try:
                logger.debug(f"Decomposing series: {series_name}")
                results[series_name] = self._decompose_single(series, **primary_params)
            except Exception as e:
                logger.warning(
                    f"Failed to decompose {series_name}: {e!s}",
                    exc_info=True,
                )
                print(f"Warning: Failed to decompose {series_name}: {e}")
                results[series_name] = self._create_fallback_decomposition(series)
                logger.info(f"Created fallback decomposition for {series_name}")
        logger.info("Completed coordinated decomposition")
        return results

    def _validate_series_alignment(self, data: dict[str, pd.Series]):
        """Ensure all series have compatible temporal alignment."""
        logger.debug("Validating series alignment")
        if not data:
            logger.error("No data provided for alignment validation")
            raise ValueError("No data provided")

        # Check index alignment
        first_series = next(iter(data.values()))
        reference_index = first_series.index

        for name, series in data.items():
            if not series.index.equals(reference_index):
                logger.error(
                    f"Index misalignment in series '{name}'. "
                    f"Expected index: {reference_index}, found: {series.index}",
                )
                raise ValueError(f"Series '{name}' index misalignment")

            # Check minimum length requirement
            min_length = max(self.config.periods) * self.config.min_periods_ratio
            if len(series) < min_length:
                logger.error(
                    f"Series '{name}' too short. Length={len(series)}, required={min_length}",
                )
                raise ValueError(
                    f"Series '{name}' too short: {len(series)} < {min_length}",
                )
        logger.debug("All series aligned and meet length requirements")

    def _get_optimal_params(self, series: pd.Series, **kwargs) -> dict:
        """Determine optimal decomposition parameters for primary series."""
        base_params = kwargs.copy()

        if self.config.method == "MSTL":
            base_params.setdefault("periods", self.config.periods)
        elif self.config.method == "STL":
            base_params.setdefault(
                "seasonal",
                self.config.seasonal or self.config.periods[0],
            )

        return base_params

    def _decompose_single(self, series: pd.Series, **params) -> dict[str, pd.Series]:
        """Decompose a single series."""
        logger.info(f"Decomposing series using {self.config.method} method")
        # Handle missing values
        if series.isnull().any():
            missing_count = series.isnull().sum()
            logger.warning(
                f"Series contains {missing_count} missing values. Applying {self.config.fill_method}",
            )
            series = self._handle_missing_values(series)

        # Perform decomposition
        if self.config.method == "MSTL":
            logger.debug(f"Running MSTL with params: {params}")
            result = MSTL(series, **params).fit()
        else:  # STL
            logger.debug(f"Running STL with seasonal={params.get('seasonal')}")
            result = STL(series, **params).fit()

        logger.debug("Decomposition completed successfully")
        return {
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
            "observed": series,
        }

    def _handle_missing_values(self, series: pd.Series) -> pd.Series:
        """Handle missing values according to configuration."""
        original_count = len(series)
        if self.config.fill_method == "interpolate":
            return series.interpolate(method="linear")
        elif self.config.fill_method == "forward":
            return series.fillna(method="ffill")
        elif self.config.fill_method == "drop":
            return series.dropna()

        new_count = len(series)
        if new_count != original_count:
            logger.warning(
                f"Series length changed from {original_count} to {new_count} after handling missing values",
            )
        return series

    def _create_fallback_decomposition(self, series: pd.Series) -> dict[str, pd.Series]:
        """Create fallback decomposition when normal decomposition fails."""
        logger.warning("Creating fallback decomposition using rolling mean")
        return {
            "trend": series.rolling(
                window=max(self.config.periods),
                center=True,
            ).mean(),
            "seasonal": pd.Series(0, index=series.index),
            "resid": series
            - series.rolling(window=max(self.config.periods), center=True).mean(),
            "observed": series,
        }

    def create_feature_matrix(
        self,
        decomposed_data: dict[str, dict[str, pd.Series]],
        components: list[str] = ["trend", "seasonal"],
    ) -> pd.DataFrame:
        """Create aligned feature matrix for training.

        Args:
            decomposed_data: Output from decompose_coordinated
            components: Which components to include

        Returns:
            DataFrame with columns like 'solar_trend', 'demand_seasonal', etc.
        """
        logger.info(f"Creating feature matrix with components: {components}")
        feature_dfs = []

        for series_name, components_dict in decomposed_data.items():
            for component in components:
                if component in components_dict:
                    col_name = f"{series_name}_{component}"
                    feature_dfs.append(components_dict[component].rename(col_name))
        logger.debug(f"Created feature matrix with {len(feature_dfs)} columns")
        return pd.concat(feature_dfs, axis=1)


# Convenience factory functions
def create_solar_decomposer(granularity: str = "hourly") -> CoordinatedDecomposer:
    """Create decomposer optimized for solar energy analysis."""
    logger.info(f"Creating solar decomposer for {granularity} granularity")
    config = DecompositionConfig(
        method="MSTL",
        granularity=granularity,
        energy_type="solar",
        periods=[24, 8760] if granularity == "hourly" else [7, 365],
    )
    return CoordinatedDecomposer(config)


def create_wind_decomposer(granularity: str = "hourly") -> CoordinatedDecomposer:
    """Create decomposer optimized for wind energy analysis."""
    logger.info(f"Creating wind decomposer for {granularity} granularity")
    config = DecompositionConfig(
        method="MSTL",
        granularity=granularity,
        energy_type="wind",
        periods=[24, 168, 8760]
        if granularity == "hourly"
        else [7, 365],  # Include yearly for wind
    )
    return CoordinatedDecomposer(config)


if __name__ == "__main__":
    # Paths
    data_path = Path().cwd() / "data" / "2_decomposer"
    input_files = data_path / "input"
    output_files = data_path / "output"

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

    def test_usage():
        # Read data
        # hourly_generation = pd.read_csv(input_files / "hourly_generation.csv")
        #
        decomposer = create_wind_decomposer()
        # data = {
        #     "solar": hourly_generation["solar"],
        #     "eolica": hourly_generation["eolica"],
        # }
        #

        features = decomposer.create_feature_matrix(
            decomposed,
            components=["trend", "seasonal"],
        )

        features.to_csv(output_files / "solar_eolica.csv")

    # Example usage
    def example_usage():
        """Example of how to use the coordinated decomposer."""
        # Create sample data
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="h")
        solar_data = pd.Series(
            np.random.randn(len(dates))
            + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 24),
            index=dates,
            name="solar_generation",
        )
        demand_data = pd.Series(
            np.random.randn(len(dates))
            + 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 24),
            index=dates,
            name="energy_demand",
        )

        # Setup coordinated decomposition
        decomposer = create_solar_decomposer("hourly")

        # Decompose multiple series together
        data = {"solar": solar_data, "demand": demand_data}

        # decomposed = decomposer.decompose_coordinated(data=data, primary_series="solar")
        # export_to_pkl(decomposed, file_path=output_files / "demo_decomposed.pkl")
        decomposed = import_from_pkl(output_files / "demo_decomposed.pkl")

        # Create feature matrix for training
        features = decomposer.create_feature_matrix(
            decomposed,
            components=["trend", "seasonal"],
        )

        return features, decomposed

    example_usage()
