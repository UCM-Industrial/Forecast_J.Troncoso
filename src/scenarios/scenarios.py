from typing import Any

import numpy as np
import pandas as pd


class Scenario:
    """Generates future scenarios with realistic synthetic data for climate variables and capacity."""

    def __init__(self, config: dict[str, Any]) -> None:  # noqa: D107
        self.name = config.get("name", "Unnamed Scenario")
        self.transformations = config.get("transformations", [])
        self._method_map = {
            "trend": self._apply_trend,
            "exponential": self._apply_exponential,
            "climate_shift": self._apply_climate_shift,
            "seasonal_variability": self._apply_seasonal_variability,
            "capacity_steps": self._apply_capacity_steps,
            "synthetic_generation": self._apply_synthetic_generation,
        }

    def _clean_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean duplicate indices by keeping the first occurrence."""
        if df.index.duplicated().any():
            print(
                f"Warning: Found {df.index.duplicated().sum()} duplicate timestamps. Keeping first occurrence.",
            )
            df = df[~df.index.duplicated(keep="first")]
        return df

    def _get_cols(self, params: dict[str, Any]) -> list[str]:
        """Extract column names from parameters, handling both single and multiple columns."""
        cols = params.get("col") or params.get("cols")
        if not cols:
            return []
        return [cols] if isinstance(cols, str) else list(cols)

    def _validate_columns(self, df: pd.DataFrame, cols: list[str]) -> list[str]:
        """Validate that columns exist in the dataframe and return only valid ones."""
        valid_cols = [col for col in cols if col in df.columns]
        if len(valid_cols) != len(cols):
            missing = set(cols) - set(valid_cols)
            print(f"Warning: Columns {missing} not found in DataFrame")
        return valid_cols

    def generate_future(self, data: pd.DataFrame, future_years: int) -> pd.DataFrame:
        """Generates realistic future scenarios using simple statistical models.

        Args:
            data (pd.DataFrame): Historical data with datetime index (America/Santiago timezone)
            future_years (int): Number of years to generate into the future

        Returns:
            pd.DataFrame: Combined historical and synthetic future data
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("The input DataFrame must have a DatetimeIndex.")

        # Clean duplicates first
        clean_data = self._clean_index(data.copy())

        # Infer frequency from clean data
        freq = self._infer_frequency(clean_data)

        # Generate future timeline
        last_timestamp = clean_data.index.max()
        periods = future_years * 365 * 24  # Hourly for specified years

        future_index = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=periods,
            freq="h",
            tz=clean_data.index.tz,  # Preserve timezone
        )

        # Generate synthetic future data
        future_df = self._generate_synthetic_data(clean_data, future_index)

        # Combine historical and future
        combined_df = pd.concat([clean_data, future_df])

        # Apply transformations
        return self.apply(combined_df, is_internal_call=True)

    def _infer_frequency(self, df: pd.DataFrame) -> str:
        """Infer frequency from clean dataframe."""
        try:
            freq = pd.infer_freq(df.index)
            if freq:
                return freq
        except:
            pass

        # Fallback: calculate median time difference
        time_diffs = df.index.to_series().diff().dropna()
        median_diff = time_diffs.median()

        if median_diff <= pd.Timedelta(hours=1):
            return "h"  # Hourly
        elif median_diff <= pd.Timedelta(days=1):
            return "D"  # Daily
        else:
            return "h"  # Default to hourly

    def _generate_synthetic_data(
        self,
        historical_df: pd.DataFrame,
        future_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Generate synthetic future data using simple statistical models."""
        future_df = pd.DataFrame(index=future_index, columns=historical_df.columns)

        for col in historical_df.columns:
            future_df[col] = self._generate_column_data(
                historical_df[col],
                future_index,
            )

        return future_df

    def _generate_column_data(
        self,
        historical_series: pd.Series,
        future_index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Generate synthetic data for a single column using seasonal patterns + noise."""
        # Extract seasonal patterns by hour and day of year
        hist_df = pd.DataFrame(
            {
                "value": historical_series,
                "hour": historical_series.index.hour,
                "dayofyear": historical_series.index.dayofyear,
                "month": historical_series.index.month,
            },
        ).dropna()

        if len(hist_df) == 0:
            return pd.Series(0, index=future_index)

        # Calculate seasonal averages and variability
        hourly_pattern = hist_df.groupby("hour")["value"].agg(["mean", "std"]).fillna(0)
        monthly_pattern = (
            hist_df.groupby("month")["value"].agg(["mean", "std"]).fillna(0)
        )

        # Generate synthetic values
        synthetic_values = []

        for timestamp in future_index:
            hour = timestamp.hour
            month = timestamp.month

            # Base value from hourly pattern
            base_value = hourly_pattern.loc[hour, "mean"]
            base_std = hourly_pattern.loc[hour, "std"]

            # Monthly adjustment
            monthly_factor = (
                monthly_pattern.loc[month, "mean"] / hist_df["value"].mean()
                if hist_df["value"].mean() != 0
                else 1
            )

            # Add realistic noise
            noise = np.random.normal(0, max(base_std * 0.3, abs(base_value) * 0.1))

            synthetic_value = base_value * monthly_factor + noise

            # Ensure non-negative for physical variables (like solar irradiance)
            if historical_series.min() >= 0:
                synthetic_value = max(0, synthetic_value)

            synthetic_values.append(synthetic_value)

        return pd.Series(synthetic_values, index=future_index)

    def apply(self, data: pd.DataFrame, is_internal_call: bool = False) -> pd.DataFrame:
        """Apply all transformation steps defined in the configuration."""
        if not is_internal_call and not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Input DataFrame must have a DatetimeIndex.")

        df = data.copy() if not is_internal_call else data

        for step in self.transformations:
            transform_type = step.get("type")
            if transform_type not in self._method_map:
                print(
                    f"Warning: Unknown transformation type '{transform_type}', skipping.",
                )
                continue

            cols_to_apply = self._get_cols(step)
            if not cols_to_apply:
                print(
                    f"Warning: No columns specified for transformation '{transform_type}', skipping.",
                )
                continue

            valid_cols = self._validate_columns(df, cols_to_apply)
            if not valid_cols:
                continue

            params = step.get("params", {})
            df = self._method_map[transform_type](df, params, valid_cols)

        return df

    def _apply_trend(
        self,
        df: pd.DataFrame,
        params: dict,
        cols: list[str],
    ) -> pd.DataFrame:
        """Apply linear trend to specified columns."""
        rate = params.get("rate", 0)
        if rate == 0:
            return df

        # Calculate years from start
        # TODO: Enable flexibility for different granularities
        years = (df.index - df.index[0]).total_seconds() / (365.25 * 24 * 3600)

        for col in cols:
            df[col] = df[col] * (1 + rate * years)
        return df

    def _apply_exponential(
        self,
        df: pd.DataFrame,
        params: dict,
        cols: list[str],
    ) -> pd.DataFrame:
        """Apply exponential growth/decay to specified columns."""
        rate = params.get("rate", 0)
        if rate == 0:
            return df

        # TODO: Enable flexibility for different granularities
        years = (df.index - df.index[0]).total_seconds() / (365.25 * 24 * 3600)

        for col in cols:
            df[col] = df[col] * ((1 + rate) ** years)
        return df

    def _apply_climate_shift(
        self,
        df: pd.DataFrame,
        params: dict,
        cols: list[str],
    ) -> pd.DataFrame:
        """Apply gradual climate shift over time."""
        total_change = params.get("change", 0)
        start_year = params.get(
            "start_year",
            0,
        )  # Years from beginning when change starts

        if total_change == 0:
            return df

        years = (df.index - df.index[0]).total_seconds() / (365.25 * 24 * 3600)

        # Gradual change starting from start_year
        change_factor = np.where(
            years <= start_year,
            0,
            (years - start_year) * total_change / max(years.max() - start_year, 1),
        )

        for col in cols:
            df[col] = df[col] * (1 + change_factor)
        return df

    def _apply_seasonal_variability(
        self,
        df: pd.DataFrame,
        params: dict,
        cols: list[str],
    ) -> pd.DataFrame:
        """Modify seasonal variability patterns."""
        factor = params.get("factor", 1.0)
        window = params.get("window", 24 * 7)  # Weekly window for hourly data

        for col in cols:
            # Calculate seasonal baseline
            baseline = df[col].rolling(window=window, center=True, min_periods=1).mean()
            deviations = df[col] - baseline
            df[col] = baseline + (deviations * factor)

        return df

    def _apply_capacity_steps(
        self,
        df: pd.DataFrame,
        params: dict,
        cols: list[str],
    ) -> pd.DataFrame:
        """Apply step increases in capacity at specific dates."""
        steps = params.get("steps", [])

        for col in cols:
            for step in steps:
                step_date = step.get("date")
                increase = step.get("absolute_increase", 0)

                if step_date and increase:
                    try:
                        date_threshold = pd.to_datetime(step_date)
                        # Handle timezone if present
                        if df.index.tz and date_threshold.tz is None:
                            date_threshold = date_threshold.tz_localize(df.index.tz)
                        df.loc[df.index >= date_threshold, col] += increase
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid date '{step_date}' in capacity step")

        return df

    def _apply_synthetic_generation(
        self,
        df: pd.DataFrame,
        params: dict,
        cols: list[str],
    ) -> pd.DataFrame:
        """Apply additional synthetic variations to make data more organic."""
        noise_level = params.get("noise_level", 0.05)  # 5% noise by default

        for col in cols:
            col_std = df[col].std()
            noise = np.random.normal(0, col_std * noise_level, len(df))
            df[col] = df[col] + noise

            # Keep non-negative for physical variables
            if df[col].min() < 0 and (df[col] >= 0).all():
                df[col] = df[col].clip(lower=0)

        return df


if __name__ == "__main__":
    ...
