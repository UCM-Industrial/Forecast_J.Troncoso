import numpy as np
import pandas as pd


def cyclical_encode_datetime(
    df: pd.DataFrame,
    datetime_col: str | None = None,
    features: list[str] = ["hour", "day", "month"],
):
    """Apply cyclical encoding to datetime features."""
    # Create a copy to avoid modifying the original dataframe
    temp_df = df.copy()

    # Determine the datetime series to use
    if datetime_col:
        # Use specified column
        if datetime_col not in temp_df.columns:
            raise ValueError(f"Column '{datetime_col}' not found in DataFrame")
        df_series = temp_df[datetime_col]
    else:
        # Use index as datetime source
        if not isinstance(temp_df.index, pd.DatetimeIndex):
            raise ValueError(
                "When datetime_col is None, DataFrame index must be a DatetimeIndex",
            )
        df_series = temp_df.index.to_series()

    # Ensure the series is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_series):
        try:
            df_series = pd.to_datetime(df_series)
        except Exception as e:
            raise ValueError(f"Cannot convert to datetime: {e}")

    # Mapping of features to their maximum values for cyclical encoding
    max_vals = {
        "hour": 24,
        "day": 31,
        "month": 12,
        "dayofweek": 7,
    }

    for feature in features:
        if feature not in max_vals:
            print(f"Warning: Feature '{feature}' not recognized. Skipping.")
            continue

        # Extract the appropriate datetime component
        if feature == "hour":
            values = df_series.dt.hour
        elif feature == "day":
            values = df_series.dt.day
        elif feature == "month":
            values = df_series.dt.month
        elif feature == "dayofweek":
            values = df_series.dt.dayofweek
        else:
            raise ValueError

        # Apply cyclical encoding: sin and cos transformations
        max_val = max_vals[feature]
        temp_df[f"{feature}_sin"] = np.sin(2 * np.pi * values / max_val)
        temp_df[f"{feature}_cos"] = np.cos(2 * np.pi * values / max_val)

    return temp_df


# NOTE: This function could be unnecesary
def make_date_columns(
    df: pd.DataFrame,
    date_col: str = "Date",
    components: list[str] | None = None,
    use_24_hour: bool = False,
) -> pd.DataFrame:
    """Add date components as columns to the DataFrame."""
    # Validate inputs
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame")  # noqa: TRY003

    # Set default components if not specified
    if components is None:
        components = ["year", "month", "day", "hour"]

    result_df = df.copy()

    # Define available datetime components and their accessor methods
    component_methods = {
        "year": "year",
        "month": "month",
        "day": "day",
        "hour": "hour",
        "minute": "minute",
        "second": "second",
        "dayofweek": "dayofweek",
        "dayofyear": "dayofyear",
        "quarter": "quarter",
        "week": "isocalendar().week",
    }

    # Extract each requested component
    for component in components:
        if component not in component_methods:
            print(
                f"Warning: '{component}' is not a recognized date component and will be skipped",
            )
            continue

        method = component_methods[component]
        try:
            # Special handling for week which uses a different accessor pattern
            if method == "isocalendar().week":
                result_df[component] = result_df[date_col].dt.isocalendar().week
            else:
                result_df[component] = getattr(result_df[date_col].dt, method)

            # Special handling for hour if use_24_hour is True
            if component == "hour" and use_24_hour:
                # Find rows where time is 23:59:xx
                mask = (result_df[component] == 23) & (  # noqa: PLR2004
                    result_df[date_col].dt.minute >= 59  # noqa: PLR2004
                )
                # Convert hour 23 to hour 24 for these rows
                result_df.loc[mask, component] = 24
        except (ValueError, AttributeError) as e:
            print(f"Could not extract '{component}' from {date_col}: {e!s}")

    return result_df


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    date_col: str | None = None,
) -> pd.DataFrame:
    """Filters a DataFrame to include rows within a specified date range.

    date_col: str   - Must be a datetime column
    """
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date)

    if date_col:
        mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)

    mask = (df.index >= start_date) & (df.index <= end_date)

    return df.loc[mask].copy()
