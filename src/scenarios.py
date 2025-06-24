import numpy as np
import pandas as pd
import plotly.graph_objects as go


class Scenario:
    """Simple and flexible scenario generator for renewable energy data."""

    def __init__(
        self,
        name: str = "Unnamed Scenario",
        future_years: int = 2,
        freq: str = "h",
        tz: str = "America/Santiago",
    ) -> None:
        self.name = name
        self.future_years = future_years
        self.freq = freq
        self.configs = []
        self.tz = tz

    def generate_future_date_range(
        self,
        last_date: pd.Timestamp,
        periods: int,
        freq: str,
    ) -> pd.DatetimeIndex:
        """Generate a future date range starting after the last_date.

        Args:
            last_date: Last timestamp of existing data
            periods: Number of periods to generate
            freq: Pandas frequency string (e.g., 'h', 'D')
            tz: Timezone name (optional, defaults to last_date's timezone)

        Returns:
            pd.DatetimeIndex localized to the specified timezone
        """
        # Calculate start date using frequency-aware offset
        start_date = last_date + pd.tseries.frequencies.to_offset(freq)

        return pd.date_range(
            start=start_date,
            periods=periods,
            freq=self.freq,
            tz=self.tz if self.tz else last_date.tzinfo,
        )

    def generate_seasonal_like(
        self,
        series: pd.Series,
        variability_factor=0.02,
    ):
        # Repeat the series and add noise
        config_dict = {
            "h": {
                "seasonal": 24,
            },
            "D": {
                "seasonal": 1,
            },
        }
        base_year = series[-365 * config_dict[self.freq]["seasonal"] :]
        synthetic = pd.concat(
            [
                base_year
                + np.random.normal(
                    0,
                    variability_factor * base_year.std(),
                    len(base_year),
                )
                for _ in range(self.future_years)
            ],
        )
        synthetic.index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=len(synthetic),
            freq=self.freq,
        )
        return synthetic

    def add_config(self, cols: str | list[str], **transformations) -> "Scenario":
        """Add a configuration for specific columns with transformations.

        Args:
            cols: Column name(s) to apply transformations to
            **transformations: Transformation methods and their parameters

        Returns:
            Self for method chaining
        """
        if isinstance(cols, str):
            cols = [cols]

        config = {"cols": cols, "transformations": transformations}
        self.configs.append(config)
        return self

    def config(self, config_dict: dict) -> "Scenario":
        """Override configurations with a dictionary.

        Args:
            config_dict: Dictionary with configuration structure
        """
        self.configs = config_dict.get("configs", [])
        self.name = config_dict.get("name", self.name)
        return self

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all configurations to the dataframe.

        Args:
            df: Input dataframe with datetime index

        Returns:
            Modified dataframe with same index
        """
        result_df = df.copy()

        for config in self.configs:
            cols = config["cols"]
            transformations = config["transformations"]

            # Validate columns exist
            valid_cols = [col for col in cols if col in result_df.columns]
            if len(valid_cols) != len(cols):
                missing = set(cols) - set(valid_cols)
                print(f"Warning: Columns {missing} not found in DataFrame")

            if not valid_cols:
                continue

            # Apply each transformation method
            for method_name, params in transformations.items():
                if hasattr(self, f"_apply_{method_name}"):
                    method = getattr(self, f"_apply_{method_name}")
                    result_df = method(result_df, valid_cols, params)
                else:
                    print(f"Warning: Unknown transformation method '{method_name}'")

        return result_df

    def _apply_climate_shift(
        self,
        df: pd.DataFrame,
        cols: list[str],
        params: dict,
    ) -> pd.DataFrame:
        """Apply gradual climate shift over time."""
        change = params.get("change", 0)
        start_year = params.get("start_year", 0)

        if change == 0:
            return df

        # Calculate years from start of dataframe
        years = (df.index - df.index[0]).total_seconds() / (365.25 * 24 * 3600)

        # Gradual change starting from start_year
        change_factor = np.where(
            years <= start_year,
            0,
            (years - start_year) * change / max(years.max() - start_year, 1),
        )

        for col in cols:
            df.loc[:, col] = df[col] * (1 + change_factor)

        return df

    def _apply_trend(
        self,
        df: pd.DataFrame,
        cols: list[str],
        params: dict,
    ) -> pd.DataFrame:
        """Apply linear trend."""
        rate = params.get("rate", 0)

        if rate == 0:
            return df

        years = (df.index - df.index[0]).total_seconds() / (365.25 * 24 * 3600)

        for col in cols:
            df.loc[:, col] = df[col] * (1 + rate * years)

        return df

    def _apply_exponential(
        self,
        df: pd.DataFrame,
        cols: list[str],
        params: dict,
    ) -> pd.DataFrame:
        """Apply exponential growth/decay."""
        rate = params.get("rate", 0)

        if rate == 0:
            return df

        years = (df.index - df.index[0]).total_seconds() / (365.25 * 24 * 3600)

        for col in cols:
            df.loc[:, col] = df[col] * ((1 + rate) ** years)

        return df

    def _apply_capacity_steps(
        self,
        df: pd.DataFrame,
        cols: list[str],
        params: dict,
    ) -> pd.DataFrame:
        """Apply capacity steps at specific dates."""
        steps = params.get("steps", [])

        for col in cols:
            for step in steps:
                step_date = step.get("date")
                increase = step.get("absolute_increase", 0)

                if step_date and increase:
                    try:
                        date_threshold = pd.to_datetime(step_date)
                        # Handle timezone
                        if df.index.tz and date_threshold.tz is None:
                            date_threshold = date_threshold.tz_localize(df.index.tz)
                        df.loc[df.index >= date_threshold, col] += increase
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid date '{step_date}' in capacity step")

        return df

    def _apply_seasonal_adjustment(
        self,
        df: pd.DataFrame,
        cols: list[str],
        params: dict,
    ) -> pd.DataFrame:
        """Adjust seasonal patterns."""
        month_factors = params.get("month_factors", {})  # {1: 1.1, 7: 0.9, ...}
        hour_factors = params.get("hour_factors", {})  # {12: 1.2, 0: 0.8, ...}

        for col in cols:
            # Apply monthly factors
            for month, factor in month_factors.items():
                mask = df.index.month == month
                df.loc[mask, col] *= factor

            # Apply hourly factors
            for hour, factor in hour_factors.items():
                mask = df.index.hour == hour
                df.loc[mask, col] *= factor

        return df

    def _apply_noise(
        self,
        df: pd.DataFrame,
        cols: list[str],
        params: dict,
    ) -> pd.DataFrame:
        """Add random noise."""
        level = params.get("level", 0.05)  # 5% noise by default
        seed = params.get("seed")

        if seed is not None:
            np.random.seed(seed)

        for col in cols:
            col_std = df[col].std()
            noise = np.random.normal(0, col_std * level, len(df))
            df.loc[:, col] = df[col] + noise

            # Keep non-negative if original data was non-negative
            if (df[col] >= 0).all():
                df.loc[:, col] = df[col].clip(lower=0)

        return df

    def summary(self) -> str:
        """Return a summary of configured transformations."""
        summary = [f"Scenario: {self.name}"]
        summary.append(f"Number of configurations: {len(self.configs)}")

        for i, config in enumerate(self.configs):
            summary.append(f"\nConfig {i + 1}:")
            summary.append(f"  Columns: {config['cols']}")
            for method, params in config["transformations"].items():
                summary.append(f"  {method}: {params}")

        return "\n".join(summary)


# --- Plot functions ---


def create_scenario_plot(
    df: pd.DataFrame,
    cols: list[str],
    second_axis_cols: list[str] | None = None,
) -> go.Figure:
    fig = go.Figure()

    for _, col in enumerate(cols):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))

    if second_axis_cols:
        for _, col in enumerate(second_axis_cols):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=col,
                    yaxis="y2",
                ),
            )

    # Update layout for multiple y-axes
    fig.update_layout(
        yaxis=dict(title="Primary Y-Axis"),
        yaxis2=dict(title="Secondary Y-Axis", overlaying="y", side="right"),
        xaxis=dict(title="X Axis"),
        title="Multi-Axis Scenario Plot",
    )

    return fig


if __name__ == "__main__":
    ...
