import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_dataframe(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Ensure the date column is in datetime format.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to prepare
    date_col : str
        The column name containing dates

    Returns:
    --------
    pd.DataFrame
        Copy of DataFrame with properly formatted date column
    """
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    return df_copy


def find_missing_dates(df: pd.DataFrame, date_col: str, freq: str = "D") -> list:
    """Find missing dates in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check
    date_col : str
        The column name containing dates
    freq : str, default "D"
        Frequency to check ('D' for daily, 'H' for hourly, etc.)

    Returns:
    --------
    List
        List of missing dates in datetime format
    """
    # For daily checks, normalize to remove time components
    if freq == "D":
        actual_dates = pd.Series(df[date_col].dt.normalize().unique()).sort_values()
    else:
        actual_dates = pd.Series(df[date_col].unique()).sort_values()

    if actual_dates.empty:
        return []

    # Find min and max dates
    min_date = actual_dates.min()
    max_date = actual_dates.max()

    # Create expected date range
    expected_date_range = pd.date_range(start=min_date, end=max_date, freq=freq)

    # Find missing dates
    missing_dates = sorted(set(expected_date_range) - set(actual_dates))

    return missing_dates


def group_dates_into_clusters(dates: list) -> list[list]:
    """Group dates into clusters where consecutive dates are in the same cluster.

    Parameters:
    -----------
    dates : List
        List of dates to group

    Returns:
    --------
    List[List]
        List of date clusters
    """
    if not dates:
        return []

    dates = sorted(dates)
    clusters = []
    current_cluster = [dates[0]]

    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days

        if gap <= 1:  # If consecutive
            current_cluster.append(dates[i])
        else:
            # End current cluster and start new
            clusters.append(current_cluster)
            current_cluster = [dates[i]]

    # Add the last cluster
    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def report_missing_dates(missing_dates: list) -> None:
    """Print a report of missing dates.

    Parameters:
    -----------
    missing_dates : List
        List of missing dates
    """
    if not missing_dates:
        return

    # Format dates for display
    missing_dates_str = [d.strftime("%Y-%m-%d") for d in missing_dates]

    # Group dates into clusters for clearer reporting
    if len(missing_dates) > 10:
        clusters = group_dates_into_clusters(missing_dates)

        for i, cluster in enumerate(clusters):
            if len(cluster) == 1:
                print(f"  Isolated missing date: {cluster[0].strftime('%Y-%m-%d')}")
            else:
                print(
                    f"  Missing cluster #{i + 1}: {cluster[0].strftime('%Y-%m-%d')} to {cluster[-1].strftime('%Y-%m-%d')} ({len(cluster)} dates)",
                )
    else:
        print(f"  Missing dates: {', '.join(missing_dates_str)}")


def convert_hours_to_numeric(df: pd.DataFrame, hour_col: str) -> pd.Series | None:
    """Convert hour column to numeric values if needed.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the hour column
    hour_col : str
        The column name containing hour information

    Returns:
    --------
    pd.Series or None
        Numeric hour series or None if conversion failed
    """
    if pd.api.types.is_numeric_dtype(df[hour_col]):
        return df[hour_col]

    try:
        return pd.to_numeric(df[hour_col])
    except ValueError:
        print(f"⚠ Warning: Could not convert {hour_col} to numeric. Example values:")
        print(df[hour_col].head(10).tolist())
        return None


def find_invalid_hours(df: pd.DataFrame, date_col: str, hour_col: str) -> list[dict]:
    """Find hours outside the valid range (0-23).

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check
    date_col : str
        The column name containing dates
    hour_col : str
        The column name containing hours

    Returns:
    --------
    List[Dict]
        List of records with invalid hours
    """
    invalid_mask = ~df[hour_col].between(0, 23)
    if not invalid_mask.any():
        return []

    invalid_hours = df[invalid_mask][[date_col, hour_col]].copy()
    invalid_hours["date_str"] = invalid_hours[date_col].dt.strftime("%Y-%m-%d")

    return (
        invalid_hours[["date_str", hour_col]]
        .rename(columns={"date_str": "date"})
        .to_dict("records")
    )


def find_missing_hours_by_date(
    df: pd.DataFrame,
    date_col: str,
    hour_col: str,
) -> list[dict]:
    """Find days with missing hours.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check
    date_col : str
        The column name containing dates
    hour_col : str
        The column name containing hours

    Returns:
    --------
    List[Dict]
        List of days with missing hours and which hours are missing
    """
    df["date_only"] = df[date_col].dt.normalize()
    hours_by_date = df.groupby("date_only")[hour_col]
    hour_counts = hours_by_date.nunique()

    missing_hours_list = []

    for date, unique_hours in hour_counts.items():
        if unique_hours < 24:
            present_hours = set(df[df["date_only"] == date][hour_col])
            missing_hour_values = set(range(24)) - present_hours

            if missing_hour_values:
                missing_hours_list.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "present_hours": len(present_hours),
                        "missing_hours": sorted(missing_hour_values),
                    },
                )

    return missing_hours_list


def find_duplicate_hours(df: pd.DataFrame, date_col: str, hour_col: str) -> list[dict]:
    """Find days with duplicate hour entries.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check
    date_col : str
        The column name containing dates
    hour_col : str
        The column name containing hours

    Returns:
    --------
    List[Dict]
        List of days with duplicate hours
    """
    df["date_only"] = df[date_col].dt.normalize()
    duplicates_list = []

    for date, hours in df.groupby("date_only")[hour_col]:
        hour_value_counts = hours.value_counts()
        duplicates = hour_value_counts[hour_value_counts > 1]

        if not duplicates.empty:
            duplicates_list.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "duplicate_hours": {
                        int(hour): count for hour, count in duplicates.items()
                    },
                },
            )

    return duplicates_list


def check_date_completeness(
    df: pd.DataFrame,
    date_col: str = "fecha_opreal",
    freq: str = "D",
) -> dict:
    """Check for missing dates in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check
    date_col : str, default "fecha_opreal"
        The column name containing dates
    freq : str, default "D"
        Frequency to check ('D' for daily, 'H' for hourly, etc.)

    Returns:
    --------
    Dict
        Dictionary with date validation results
    """
    # Prepare dataframe
    df = prepare_dataframe(df, date_col)

    # Find missing dates
    missing_dates = find_missing_dates(df, date_col, freq)

    # Prepare results
    results = {
        "summary": {
            "date_range": {
                "start": df[date_col].min().strftime("%Y-%m-%d"),
                "end": df[date_col].max().strftime("%Y-%m-%d"),
            },
            "total_records": len(df),
            "missing_dates_count": len(missing_dates),
        },
        "missing_dates": [d.strftime("%Y-%m-%d") for d in missing_dates],
    }

    # Print report
    if missing_dates:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        print(
            f"Found {len(missing_dates)} missing date(s) between {min_date.strftime('%Y-%m-%d')} and {max_date.strftime('%Y-%m-%d')}.",
        )
        report_missing_dates(missing_dates)
    else:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        print(
            f"Complete: No missing dates found in the range {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}.",
        )

    return results


def check_hourly_completeness(
    df: pd.DataFrame,
    date_col: str = "fecha_opreal",
    hour_col: str = "hora_opreal",
) -> dict:
    """Check for hourly completeness and consistency in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check
    date_col : str, default "fecha_opreal"
        The column name containing dates
    hour_col : str, default "hora_opreal"
        The column name containing hours

    Returns:
    --------
    Dict
        Dictionary with hour validation results
    """
    # Prepare dataframe
    df = prepare_dataframe(df, date_col)

    # Convert hours to numeric
    df = df.copy()
    df[hour_col] = convert_hours_to_numeric(df, hour_col)

    if df[hour_col] is None:
        return {"error": "Hour column could not be converted to numeric type"}

    # Validate hours
    invalid_hours = find_invalid_hours(df, date_col, hour_col)
    missing_hours = find_missing_hours_by_date(df, date_col, hour_col)
    duplicate_hours = find_duplicate_hours(df, date_col, hour_col)

    # Prepare results
    hour_issues = {
        "invalid_hours": invalid_hours,
        "days_with_missing_hours": missing_hours,
        "days_with_duplicate_hours": duplicate_hours,
    }

    # Print summary
    has_hour_issues = any(len(v) > 0 for v in hour_issues.values())

    if has_hour_issues:
        print("\nFound hour inconsistencies:")
        for issue_type, issues in hour_issues.items():
            if issues:
                print(f"  - {issue_type}: {len(issues)} instances found")
    else:
        print("\nAll hour validations passed! All days have exactly 24 hours (0-23).")

    return hour_issues


def check_time_series(
    df: pd.DataFrame,
    date_col: str = "fecha_opreal",
    hour_col: str | None = None,
    freq: str = "D",
) -> dict:
    """Main function to check time series completeness.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check
    date_col : str, default "fecha_opreal"
        The column name containing dates
    hour_col : str, default None
        Optional column containing hours. If provided, will check hourly completeness
    freq : str, default "D"
        Frequency to check if hour_col is None ('D' for daily, 'H' for hourly, etc.)

    Returns:
    --------
    Dict
        Dictionary with validation results
    """
    # Check date completeness
    date_results = check_date_completeness(df, date_col, freq)

    results = {
        "summary": date_results["summary"],
        "missing_dates": date_results["missing_dates"],
    }

    # Check hour completeness if hour_col provided
    if hour_col:
        hour_results = check_hourly_completeness(df, date_col, hour_col)
        results["hour_issues"] = hour_results

    return results


def calculate_validation_metrics(combined_df):
    """Calculate various validation metrics between pairs of columns with _df1 and _df2 suffixes."""
    results = []

    # Get base column names (without suffixes)
    base_columns = set()
    for col in combined_df.columns:
        if col.endswith("_df1"):
            base_columns.add(col[:-4])

    # Calculate metrics for each column pair
    for base_col in base_columns:
        col1 = f"{base_col}_df1"
        col2 = f"{base_col}_df2"

        if col1 not in combined_df.columns or col2 not in combined_df.columns:
            continue

        # Skip non-numeric columns
        if not np.issubdtype(combined_df[col1].dtype, np.number) or not np.issubdtype(
            combined_df[col2].dtype,
            np.number,
        ):
            continue

        # Drop NaN values for calculations
        valid_data = combined_df[[col1, col2]].dropna()
        if len(valid_data) == 0:
            continue

        x = valid_data[col1]
        y = valid_data[col2]

        # Calculate metrics
        mse = mean_squared_error(x, y)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(x, y)

        # Handle R² calculation (can be negative if model performs worse than baseline)
        try:
            r2 = r2_score(x, y)
        except:
            r2 = np.nan

        # Calculate correlation
        try:
            corr = x.corr(y)
        except:
            corr = np.nan

        # Calculate percent difference
        mean_abs_percent_diff = np.mean(np.abs((x - y) / ((x + y) / 2))) * 100

        results.append(
            {
                "column": base_col,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "correlation": corr,
                "mean_abs_percent_diff": mean_abs_percent_diff,
                "samples": len(valid_data),
            },
        )

    return pd.DataFrame(results).sort_values("column")
