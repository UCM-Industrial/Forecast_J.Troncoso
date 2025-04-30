import pandas as pd


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
                mask = (result_df[component] == 23) & (
                    result_df[date_col].dt.minute >= 59
                )
                # Convert hour 23 to hour 24 for these rows
                result_df.loc[mask, component] = 24
        except (ValueError, AttributeError) as e:
            print(f"Could not extract '{component}' from {date_col}: {e!s}")

    return result_df


def filter_by_date_range(
    df: pd.DataFrame,
    date_col: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Filters a DataFrame to include rows within a specified date range."""
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    return df.loc[mask].copy()
