import logging
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.runtime.uploaded_file_manager import UploadedFile

logger = logging.getLogger(__name__)


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
        data = pickle.load(f)

    logger.info(f"Data successfully imported from {path.absolute()}")
    return data


# INFO: Only supported for Linux
def notify(title: str, message: str):
    subprocess.run(["notify-send", title, message])


def read_csv_with_datetime(
    path: str | Path | UploadedFile,
    datetime_col: str = "datetime",
    input_timezone: str = "America/Santiago",
    output_timezone: str = "America/Santiago",
) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[datetime_col])
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="raise", utc=True)
    df[datetime_col] = df[datetime_col].dt.tz_convert(output_timezone)
    return df.set_index(datetime_col)


def create_local_datetime(
    df: pd.DataFrame,
    timezone: str = "America/Santiago",
    hour_col: str = "hora_opreal",
    date_col: str = "fecha_opreal",
):
    """Convert fecha_opreal + hora_opreal to timezone-aware datetime index."""
    df = df.copy()

    # Create base datetime
    df["datetime"] = pd.to_datetime(df["fecha_opreal"]) + pd.to_timedelta(
        df["hora_opreal"] - 1,
        unit="h",
    )

    # Handle hour 25 (extra hour during DST fall back)
    mask_25 = df["hora_opreal"] == 25  # noqa: PLR2004
    df.loc[mask_25, "datetime"] = pd.to_datetime(
        df.loc[mask_25, "fecha_opreal"],
    ) + pd.Timedelta(hours=24)

    # Split into normal hours and special cases
    normal_mask = df["hora_opreal"] <= 24  # noqa: PLR2004
    df_normal = df[normal_mask].copy()
    df_hour25 = df[~normal_mask].copy()

    # Handle normal hours with proper DST handling
    if len(df_normal) > 0:
        try:
            df_normal["datetime"] = df_normal["datetime"].dt.tz_localize(
                timezone,
                ambiguous="NaT",  # Mark ambiguous times as NaT first
                nonexistent="shift_forward",
            )

            # Handle ambiguous times (fall back - 2 occurrences of same hour)
            ambiguous_mask = df_normal["datetime"].isna()
            if ambiguous_mask.any():
                # For ambiguous times, use first=True for first occurrence, False for second
                ambiguous_dates = df_normal.loc[
                    ambiguous_mask,
                    "fecha_opreal",
                ].unique()

                for date in ambiguous_dates:
                    date_mask = (df_normal["fecha_opreal"] == date) & ambiguous_mask
                    ambiguous_hours = df_normal.loc[date_mask, "hora_opreal"].values

                    for i, (idx, row) in enumerate(df_normal[date_mask].iterrows()):
                        is_first = i < len(ambiguous_hours) // 2
                        dt_base = pd.to_datetime(date) + pd.to_timedelta(
                            row["hora_opreal"] - 1,
                            unit="h",
                        )
                        df_normal.loc[idx, "datetime"] = dt_base.tz_localize(
                            timezone,
                            ambiguous=is_first,
                        )

        except Exception:
            # Fallback: handle each datetime individually
            df_normal["datetime"] = pd.NaT
            for idx, row in df_normal.iterrows():
                dt = pd.to_datetime(row["fecha_opreal"]) + pd.to_timedelta(
                    row["hora_opreal"] - 1,
                    unit="h",
                )
                try:
                    df_normal.loc[idx, "datetime"] = dt.tz_localize(
                        timezone,
                        ambiguous="infer",
                    )
                except:
                    # For truly ambiguous times, assume first occurrence
                    df_normal.loc[idx, "datetime"] = dt.tz_localize(
                        timezone,
                        ambiguous=True,
                    )

    # Handle hour 25 (always second occurrence of repeated hour)
    if len(df_hour25) > 0:
        for idx, row in df_hour25.iterrows():
            dt_base = pd.to_datetime(row["fecha_opreal"]) + pd.Timedelta(
                hours=1,
            )  # Hour 25 = 01:00 next day
            df_hour25.loc[idx, "datetime"] = dt_base.tz_localize(
                timezone,
                ambiguous=False,
            )

    # Combine and sort
    df_combined = pd.concat([df_normal, df_hour25]).sort_values(
        ["fecha_opreal", "hora_opreal"],
    )

    return df_combined.set_index("datetime").drop(
        ["fecha_opreal", "hora_opreal"],
        axis=1,
    )


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    date_col: str | None = None,
) -> pd.DataFrame:
    """Filters a DataFrame to include rows within a specified date range."""
    # Ensure start_date and end_date are pandas Timestamps
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date)

    # Determine which date series to use for filtering
    if date_col:
        if date_col not in df.columns:
            raise KeyError(f"Column '{date_col}' not found in DataFrame.")
        dates_to_filter = df[date_col]
        if not pd.api.types.is_datetime64_any_dtype(dates_to_filter):
            raise ValueError(
                f"Column '{date_col}' ({dates_to_filter.dtype}) is not a datetime type.",
            )
    else:
        dates_to_filter = df.index
        if not isinstance(dates_to_filter, pd.DatetimeIndex):
            raise ValueError(
                f"DataFrame index ({type(dates_to_filter)}) is not a DatetimeIndex.",
            )

    # Make start_date and end_date compatible with the timezone of dates_to_filter
    df_timezone = dates_to_filter.tz

    if df_timezone is not None:  # If DataFrame's dates are timezone-aware
        # If start_date is naive, localize it to the DataFrame's timezone
        if start_date.tz is None:
            start_date = start_date.tz_localize(df_timezone)
        # If start_date is aware but different timezone, convert it
        elif start_date.tz != df_timezone:
            start_date = start_date.tz_convert(df_timezone)

        # Repeat for end_date
        if end_date.tz is None:
            end_date = end_date.tz_localize(df_timezone)
        elif end_date.tz != df_timezone:
            end_date = end_date.tz_convert(df_timezone)
    else:  # If DataFrame's dates are timezone-naive
        # If start_date is aware, make it naive (or raise error, depending on desired behavior)
        if start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        # Repeat for end_date
        if end_date.tz is not None:
            end_date = end_date.tz_localize(None)

    # Perform the filtering
    mask = (dates_to_filter >= start_date) & (dates_to_filter <= end_date)

    return df.loc[mask].copy()


def create_cyclical_features(
    df: pd.DataFrame,
    datetime_col: str | None = None,
    features: list[str] = ["hour", "day", "month"],
    include_year: bool = True,
):
    """Apply cyclical encoding to datetime features."""
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

    if include_year:
        temp_df["year"] = df_series.dt.year

    return temp_df


def truncate_datetime_by_resolution(dt_series: pd.Series, resolution: str) -> pd.Series:
    """Truncate datetime series to specified resolution for joining."""
    if resolution == "exact":
        return dt_series
    elif resolution:
        return dt_series.dt.to_period(resolution).dt.start_time
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")


def group_by_datetime_index(
    df: pd.DataFrame,
    resolution: str,
    agg_functions: str | dict[str, str] | list[str] = "sum",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Group a Dataframe by a specific resolution."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("The dataframe must have a DatetimeIndex as index")

    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()

    df_subset = df[columns].copy()
    truncated_dates = truncate_datetime_by_resolution(df.index.to_series(), resolution)
    df_subset["_temp_date"] = truncated_dates

    grouped = df_subset.groupby("_temp_date").agg(agg_functions)

    grouped.index.name = "datetime"

    return grouped


if __name__ == "__main__":
    ...
