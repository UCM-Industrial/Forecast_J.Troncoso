import warnings

import pandas as pd
from statsmodels.tsa.seasonal import STL, DecomposeResult

from .util import _calculate_default_trend_span, _make_odd

# class Span(Enum):
#     """Span class for period to analyze."""
#
#     HOUR = "hourly"
#     DAY = "daily"
#     WEEK = "weekly"
#     MONTH = "monthly"
#     QUARTER = "quarterly"


class Decomposer:
    """Class to perform STL decomposition on a time series."""

    # Define presets for common frequencies
    FREQ_PARAMS = {
        # Keys should be lowercase for case-insensitive matching
        "hourly": {"period": 24, "seasonal": 25, "trend": 169},  # 1 week trend
        "daily": {"period": 7, "seasonal": 9, "trend": 91},  # ~3 months trend
        "weekly": {"period": 52, "seasonal": 53, "trend": 103},  # ~2 years trend
        "monthly": {"period": 12, "seasonal": 13, "trend": 37},  # ~3 years trend
        "quarterly": {"period": 4, "seasonal": 5, "trend": 25},  # ~6 years trend
    }

    def __init__(
        self,
        period: int | None = None,
        seasonal_span: int | None = None,
        trend_span: int | None = None,
        frequency: str | None = None,
        robust: bool = False,
        seasonal_deg: int = 1,
        trend_deg: int = 1,
        low_pass_deg: int = 1,
        verbose: bool = True,
    ) -> None:
        """Initialize the Decomposer.

        Args:
            period: The period of the seasonality. Required if `frequency` is not provided.
            seasonal_span: Span (in lags) of the LOESS window for seasonal extraction.
                           If None, a default is calculated.
            trend_span: Span (in lags) of the LOESS window for trend extraction.
                        If None, a default is calculated.
            frequency: A string representing the time series frequency (e.g., 'daily', 'monthly').
                       If provided, `period`, `seasonal_span`, and `trend_span` will be
                       automatically set based on `FREQ_PARAMS`, overriding any explicitly
                       passed numeric values for these three parameters.
            robust: Flag indicating whether to use a robust version of STL.
            seasonal_deg: Degree of the polynomial for seasonal LOESS smoothing.
            trend_deg: Degree of the polynomial for trend LOESS smoothing.
            low_pass_deg: Degree of the polynomial for low-pass LOESS smoothing.

        Raises:
            ValueError: If neither `frequency` nor `period` is provided.
            ValueError: If an invalid `frequency` string is provided.
        """
        self.frequency = frequency
        self.robust = robust
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.result: DecomposeResult | None = None  # Initialize result attribute
        self.verbose = verbose

        # --- Determine period, seasonal_span, and trend_span ---
        if self.frequency is not None:
            params = self.FREQ_PARAMS[self.frequency]
            resolved_period = params["period"]
            resolved_seasonal = params["seasonal"]
            resolved_trend = params["trend"]

            # Warn if user provided frequency AND numeric parameters (which will be ignored)
            if (
                period is not None
                or seasonal_span is not None
                or trend_span is not None
            ):
                warnings.warn(
                    f"Frequency '{self.frequency}' provided. "
                    "Explicit 'period', 'seasonal_span', and 'trend_span' arguments will be ignored.",
                    UserWarning,
                )

            # Assign values from FREQ_PARAMS
            self.period = resolved_period
            self._config_seasonal_span = resolved_seasonal
            self._config_trend_span = resolved_trend

            if self.verbose:
                print(
                    f"INFO: Using parameters for frequency '{self.frequency}': "
                    f"period={self.period}, seasonal_span={self._config_seasonal_span}, "
                    f"trend_span={self._config_trend_span}",
                )

        elif period is not None:
            # Use explicitly provided parameters if frequency is not given
            self.period = period
            self._config_seasonal_span = seasonal_span
            self._config_trend_span = trend_span
            if verbose:
                print(
                    f"INFO: Using explicit parameters: period={self.period}, "
                    f"seasonal_span={self._config_seasonal_span}, "
                    f"trend_span={self._config_trend_span}",
                )

        else:
            # Raise error if neither frequency nor period is specified
            raise ValueError("Must provide either 'frequency' or 'period'.")

        # --- Final validation on period ---
        if not isinstance(self.period, int) or self.period <= 1:
            raise ValueError("Period must be an integer greater than 1.")

    def _determine_spans(self) -> tuple[int, int]:
        """Determines the seasonal and trend spans to use, applying default if needed."""
        s_window: int
        if self._config_seasonal_span is not None:
            s_window = _make_odd(self._config_seasonal_span)  # Ensure it's odd
        else:
            # Default seasonal span: odd number >= 7 and > period
            s_window = _make_odd(
                max(7, self.period + (self.period % 2 == 0)),
            )  # Slightly better default
            print(f"INFO: `seasonal_span` not provided. Using default: {s_window}")

        t_window: int
        if self._config_trend_span is not None:
            t_window = _make_odd(self._config_trend_span)  # Ensure it's odd
            # Basic check: Trend window should generally be larger than period window
            if t_window < self.period:
                warnings.warn(
                    f"Provided `trend_span` ({t_window}) is less than `period` ({self.period}). "
                    "This might lead to undesired results.",
                    UserWarning,
                )
        else:
            # Calculate default based on period and the determined seasonal span
            t_window = _calculate_default_trend_span(self.period, s_window)
            print(f"INFO: `trend_span` not provided. Using default: {t_window}")

        # Ensure spans are odd (redundant if _make_odd used above, but safe)
        s_window = _make_odd(s_window)
        t_window = _make_odd(t_window)

        return s_window, t_window

    def decompose(self, data: pd.DataFrame, target_column: str) -> DecomposeResult:
        """Perform STL Decomposition on the provided data using the instance's configuration."""
        series_to_decompose = data[target_column]

        s_window, t_window = self._determine_spans()

        if self.verbose:
            print("--- STL Decomposition Parameters ---")
            print(f"Period: {self.period}")
            print(f"Seasonal Span (final): {s_window}")
            print(f"Trend Span (final): {t_window}")
            print(f"Robust: {self.robust}")
            print(
                f"Degrees (Seasonal, Trend, Low-Pass): ({self.seasonal_deg}, {self.trend_deg}, {self.low_pass_deg})",
            )
            print("---------------------------------")

        # --- Perform Decomposition ---
        try:
            stl = STL(
                endog=series_to_decompose,
                period=self.period,
                seasonal=s_window,
                trend=t_window,
                robust=self.robust,
                seasonal_deg=self.seasonal_deg,
                trend_deg=self.trend_deg,
                low_pass_deg=self.low_pass_deg,
            )
            result: DecomposeResult = stl.fit()
            self.result = result

        except Exception as e:
            print(f"Error during STL decomposition: {e}")
            raise
        else:
            return result
