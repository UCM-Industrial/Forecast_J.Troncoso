from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import MSTL, STL, DecomposeResult

# --- Decomposition Techniques ---


def run_mstl(
    series: pd.Series,
    periods: list[int],
    windows: list[int] | None = None,
    trend_window: int | None = None,
    robust: bool = False,
    **params,
) -> pd.DataFrame:
    """Performs MSTL (Multiple Seasonal-Trend decomposition using LOESS) on a given time series and returns the components as a DataFrame.

    Args:
        series (pd.Series): The time series data to decompose. Must have a
                            DatetimeIndex.
        periods (list[int]): The seasonal periods to decompose.
        windows (list[int], optional): A list of odd integers for the seasonal
                                       smoothing windows, corresponding to each
                                       period. If None, they are determined
                                       automatically. Defaults to None.
        trend_window (int, optional): The window size for trend smoothing.
                                      Must be an odd integer. If None, it is
                                      determined automatically. Defaults to None.
        robust (bool): If True, uses a robust version of LOESS. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the original series, trend,
                      residuals, and each seasonal component.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Input series must have a DatetimeIndex.")

    # MSTL requires at least 2 full periods for each seasonality
    for period in periods:
        if len(series) < 2 * period:
            raise ValueError(
                f"Series is too short for period {period}. "
                f"Need at least {2 * period} observations.",
            )

    # --- Parameter Validation ---
    if windows:
        if len(periods) != len(windows):
            raise ValueError("Length of 'periods' and 'windows' must be the same.")
        for w in windows:
            if w % 2 == 0:
                raise ValueError(f"Seasonal window {w} must be odd.")

    # Define stl_kwargs for MSTL
    stl_kwargs = {"robust": robust}
    if trend_window:
        if trend_window % 2 == 0:
            raise ValueError(f"Trend window {trend_window} must be odd.")
        stl_kwargs["trend"] = trend_window

    # --- Decomposition ---
    mstl_result = MSTL(
        series,
        periods=periods,
        windows=windows,
        **params,
        # stl_kwargs=stl_kwargs,
    ).fit()

    # --- Format Output ---
    return _format_decomposition_result(mstl_result, periods)


def run_stl(
    series: pd.Series,
    period: int,
    seasonal_window: int | None = None,
    trend_window: int | None = None,
    robust: bool = False,
    **params,
) -> pd.DataFrame:
    """Performs STL (Seasonal-Trend decomposition using LOESS) on a given time series and returns the components as a DataFrame.

    Args:
        series (pd.Series): The time series data to decompose. Must have a
                            DatetimeIndex.
        period (int): The main seasonal period of the series.
        seasonal_window (int, optional): The window size for seasonal smoothing.
                                         Must be an odd integer. Defaults to None.
        trend_window (int, optional): The window size for trend smoothing.
                                      Must be an odd integer. Defaults to None.
        robust (bool): If True, uses a robust version of LOESS. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the original series, trend,
                      seasonal, and residual components.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Input series must have a DatetimeIndex.")

    # STL requires at least 2 full periods
    if len(series) < 2 * period:
        raise ValueError(
            f"Series is too short for period {period}. "
            f"Need at least {2 * period} observations.",
        )

    # --- Parameter Validation ---
    if seasonal_window and seasonal_window % 2 == 0:
        raise ValueError(f"Seasonal window {seasonal_window} must be odd.")
    if trend_window and trend_window % 2 == 0:
        raise ValueError(f"Trend window {trend_window} must be odd.")

    # --- Decomposition ---
    stl_result = STL(
        series,
        **params,
        period=period,
        seasonal=seasonal_window,
        trend=trend_window,
        robust=robust,
    ).fit()

    # --- Format Output ---
    return _format_decomposition_result(stl_result)


# --- Helper Function ---


def _format_decomposition_result(
    result: DecomposeResult,
    periods: list[int] | None = None,
) -> pd.DataFrame:
    """Converts a DecomposeResult object to a clean DataFrame."""
    df = pd.DataFrame(
        {
            "observed": result.observed,
            "trend": result.trend,
            "resid": result.resid,
        },
    )
    # Handle MSTL vs STL seasonal components
    if hasattr(result, "seasonal") and isinstance(result.seasonal, pd.DataFrame):
        # MSTL case
        seasonal_comps = result.seasonal
        seasonal_comps.columns = [f"seasonal_{p}" for p in periods]
        df = pd.concat([df, seasonal_comps], axis=1)
    else:
        # STL case
        df["seasonal"] = result.seasonal

    return df


def create_decomposition_figure(
    df: pd.DataFrame,
    title_text: str = "Decomposition",
    decomposition: str = "stl",
) -> go.Figure:
    """Create a time series decomposition figure with original data, trend, seasonal, and residual components.

    Args:
        df: DataFrame containing decomposition results with columns:
            - observed: Original time series
            - trend: Trend component
            - seasonal: Seasonal component
            - resid: Residual component
        title_text: Title for the figure
        decomposition: Type of decomposition ('stl' or 'mstl')

    Returns:
        A Plotly Figure object with the decomposition visualization
    """
    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=["Original", "Trend", "Seasonal", "Residual"],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    # Original data
    fig.add_trace(
        go.Scatter(
            x=df.observed.index,
            y=df.observed.values,
            name="Original",
        ),
        row=1,
        col=1,
    )

    # Trend
    fig.add_trace(
        go.Scatter(x=df["trend"].index, y=df["trend"], name="Trend"),
        row=2,
        col=1,
    )

    # Seasonal components
    if decomposition == "mstl":
        seasonal = ["seasonal_"]
        seasonal_cols = [
            col
            for col in df.columns
            if any(pattern in col.lower() for pattern in seasonal)
        ]
        for col in df[seasonal_cols]:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=f"Seasonal {col}",
                ),
                row=3,
                col=1,
            )
    else:  # stl or other types
        fig.add_trace(
            go.Scatter(
                x=df["seasonal"].index,
                y=df["seasonal"],
                name="Seasonal",
            ),
            row=3,
            col=1,
        )

    # Residual
    fig.add_trace(
        go.Scatter(
            x=df["resid"].index,
            y=df["resid"],
            name="Residual",
        ),
        row=4,
        col=1,
    )

    fig.update_layout(height=800, title_text=title_text)
    return fig


def detect_granularity(series):
    """Detect the granularity of a time series with improved accuracy.

    Args:
        series: pandas Series with datetime index

    Returns:
        tuple: (granularity_string, confidence_score, details_dict)
    """
    if not hasattr(series.index, "to_series"):
        return "Non-temporal index", 0.0, {}

    try:
        # Calculate all time deltas
        deltas = series.index.to_series().diff().dropna()

        if len(deltas) == 0:
            return "Insufficient data", 0.0, {}

        # Remove any zero deltas (duplicates)
        deltas = deltas[deltas > pd.Timedelta(0)]

        if len(deltas) == 0:
            return "No valid time differences", 0.0, {}

        # Convert to seconds for easier comparison
        delta_seconds = deltas.dt.total_seconds()

        # Define common granularities in seconds
        granularities = {
            "Millisecond": 0.001,
            "Second": 1,
            "Minute": 60,
            "5-Minute": 300,
            "15-Minute": 900,
            "30-Minute": 1800,
            "Hourly": 3600,
            "2-Hourly": 7200,
            "4-Hourly": 14400,
            "6-Hourly": 21600,
            "12-Hourly": 43200,
            "Daily": 86400,
            "Weekly": 604800,
            "Monthly": 2592000,  # 30 days
            "Quarterly": 7776000,  # 90 days
            "Yearly": 31536000,  # 365 days
        }

        # Get frequency counts
        delta_counts = Counter(delta_seconds)
        total_deltas = len(delta_seconds)

        # Find the most common delta
        most_common_delta_sec, most_common_count = delta_counts.most_common(1)[0]

        # Calculate confidence (percentage of deltas that match the most common)
        confidence = most_common_count / total_deltas

        # Find closest granularity
        best_match = None
        min_diff = float("inf")

        for gran_name, gran_seconds in granularities.items():
            diff = abs(most_common_delta_sec - gran_seconds)
            if diff < min_diff:
                min_diff = diff
                best_match = gran_name

        # Adjust confidence based on how close the match is
        if min_diff > 0:
            # If not exact match, reduce confidence
            tolerance = 0.1  # 10% tolerance
            expected_seconds = granularities[best_match]
            relative_diff = min_diff / expected_seconds
            if relative_diff > tolerance:
                confidence *= max(0.1, 1 - relative_diff)

        # Additional analysis
        details = {
            "most_common_delta_seconds": most_common_delta_sec,
            "most_common_delta_readable": str(
                pd.Timedelta(seconds=most_common_delta_sec),
            ),
            "confidence_score": confidence,
            "total_intervals": total_deltas,
            "matching_intervals": most_common_count,
            "unique_deltas": len(delta_counts),
            "delta_distribution": dict(delta_counts.most_common(5)),
        }

        # Check for irregular patterns
        if len(delta_counts) > total_deltas * 0.5:  # More than 50% unique deltas
            granularity_desc = f"Irregular ({best_match}-like)"
            confidence *= 0.5
        elif confidence < 0.7:
            granularity_desc = f"Mixed ({best_match}-dominant)"
        else:
            granularity_desc = best_match

        return granularity_desc, confidence, details

    except Exception as e:
        return f"Error: {e!s}", 0.0, {}


def format_granularity_info(granularity, confidence, details):
    """Format granularity information for display."""
    if confidence == 0.0:
        return f"⚠️ {granularity}"

    # Color coding based on confidence
    if confidence >= 0.9:
        icon = "🟢"
        conf_desc = "Very High"
    elif confidence >= 0.7:
        icon = "🟡"
        conf_desc = "High"
    elif confidence >= 0.5:
        icon = "🟠"
        conf_desc = "Medium"
    else:
        icon = "🔴"
        conf_desc = "Low"

    info_text = f"{icon} **{granularity}** (Confidence: {conf_desc} - {confidence:.1%})"

    if details:
        info_text += f"\n- Most common interval: {details.get('most_common_delta_readable', 'N/A')}"
        info_text += f"\n- {details.get('matching_intervals', 0)}/{details.get('total_intervals', 0)} intervals match"

        if details.get("unique_deltas", 0) > 1:
            info_text += f"\n- {details.get('unique_deltas', 0)} different interval types detected"

    return info_text


if __name__ == "__main__":
    data_path = Path().cwd().parent / "data"
    test_generation = (
        data_path / "processed" / "public_data" / "generation_historic_tipo.csv"
    )

    input_path = data_path / "2_decomposer" / "input"
    output_path = data_path / "2_decomposer" / "output"

    hourly_demand_path = input_path / "hourly_demand.csv"
    hourly_generation_path = input_path / "hourly_generation.csv"

    hourly_generation = read_csv_with_datetime(hourly_generation_path)

    # print("--- Running MSTL Decomposition ---")
    # try:
    #     mstl_periods = [24, 24 * 365]  # Daily and Yearly
    #     # Let statsmodels determine window sizes automatically
    #     mstl_df = run_mstl(hourly_generation, periods=mstl_periods)
    #     print("MSTL decomposition successful. Result shape:", mstl_df.shape)
    #     print(mstl_df.head())
    # except ValueError as e:
    #     print(f"Error during MSTL: {e}")

    # print("\n" + "=" * 40 + "\n")

    print("--- Running STL Decomposition (Daily) ---")
    try:
        stl_df = run_stl(hourly_generation["solar"], period=24, seasonal_window=169)
        print("STL decomposition successful. Result shape:", stl_df.shape)
        print(stl_df.head())
    except ValueError as e:
        print(f"Error during STL: {e}")
