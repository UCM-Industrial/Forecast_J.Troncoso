import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import xarray as xr


class Preprocessor:
    def __init__(self) -> None:  # noqa: D107
        self.timezone = "America/Santiago"
        self.region_mask: gpd.GeoDataFrame

    def mask_regions(
        self,
        da: xr.Dataset,
        gdf: gpd.GeoDataFrame | None,
        **kwargs,
    ):
        mask_gdf = gdf or self.region_mask
        if mask_gdf is None:
            raise ValueError("No region mask provided")

        if mask_gdf.crs != "EPSG:4326":
            mask_gdf = mask_gdf.to_crs("EPSG:4326")

        # Set default values
        kwargs.setdefault("names", "Region")
        kwargs.setdefault("overlap", False)

        regions = regionmask.from_geopandas(
            mask_gdf,
            **kwargs,
        )
        temp_mask = regions.mask(da.longitude, da.latitude)
        # masked_ds = da.where(temp_mask.notnull())

        return temp_mask

    def extract_regional_means(
        self,
        da: xr.Dataset,
        gdf: gpd.GeoDataFrame,
        time_coord: str | None = None,
        chunk_size: dict[str, int] | None = None,
    ) -> pd.DataFrame:
        """Calculate regional means with memory optimization."""
        # Convert to DataArray if needed
        if isinstance(da, xr.Dataset):
            data_vars = list(da.data_vars)
            if len(data_vars) == 1:
                da = da[data_vars[0]]
            else:
                raise ValueError("Dataset must have exactly one data variable")

        # Apply chunking for memory optimization
        if chunk_size:
            da = da.chunk(chunk_size)

        # Handle time coordinates
        da = _standardize_time_coord(da, custom_time=time_coord)

        # Calculate regional means
        mask = self.mask_regions(da, gdf)

        # regions = regionmask.from_geopandas(gdf, names="Region", overlap=False)
        # mask = regions.mask(da.longitude, da.latitude)
        regional_means = da.groupby(mask).mean(dim=["latitude", "longitude"])

        # Convert to DataFrame
        df = regional_means.to_pandas()
        region_names = dict(enumerate(gdf["Region"]))
        df.columns = df.columns.map(region_names)

        df.index = df.index.tz_localize(self.timezone)
        return df.sort_index()

    def create_cyclical_encode(
        self,
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

    def create_local_datetime(
        self,
        df: pd.DataFrame,
        timezone: str = "America/Santiago",
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

    def _standardize_time_coord(
        self,
        da: xr.DataArray,
        custom_time: str | None,
    ) -> xr.DataArray:
        """Standardize time coordinate to 'datetime'."""
        # Custom time coordinate
        if custom_time and custom_time in da.dims:
            return da.rename({custom_time: "datetime"})

        # Forecast data (time + step)
        if "step" in da.dims and "time" in da.dims:
            if "valid_time" not in da.coords:
                raise ValueError("Missing 'valid_time' coordinate for forecast data")

            da_stacked = da.stack(forecast_time=("time", "step"))
            valid_times = da.valid_time.stack(forecast_time=("time", "step"))
            da = da_stacked.assign_coords(datetime=valid_times).swap_dims(
                {"forecast_time": "datetime"},
            )

            # Clean up old coordinates
            drop_coords = ["forecast_time"]

            coords_to_check = ["time", "step"]
            drop_coords.extend(
                [coord for coord in coords_to_check if coord in da.coords],
            )
            da = da.drop_vars(drop_coords)

        # Standard time coordinate
        elif "time" in da.dims:
            da = da.rename({"time": "datetime"})

        # Already standardized
        elif "datetime" in da.dims:
            pass

        else:
            raise ValueError(f"No recognized time coordinate found in {list(da.dims)}")

        return da

    def filter_by_date_range(
        df: pd.DataFrame,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        date_col: str | None = None,
    ) -> pd.DataFrame:
        """Filters a DataFrame to include rows within a specified date range.

        Args:
            df (pd.DataFrame): The DataFrame to filter.
            start_date (str | pd.Timestamp): The start of the date range.
            end_date (str | pd.Timestamp): The end of the date range.
            date_col (str | None, optional): The name of the datetime column in df.
                                             If None, the DataFrame's index is used.
                                             Defaults to None.

        Returns:
            pd.DataFrame: A new DataFrame containing rows within the specified range.

        Raises:
            ValueError: If date_col is specified but not a datetime type,
                        or if date_col is None and the index is not a DatetimeIndex.
            KeyError: If date_col is specified but not found in the DataFrame.
        """
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


if __name__ == "__main__":
    # TODO: Add paths and load the data to test the class.
    test_da = ""
    test_gdf = ""
    test_generation = ""

    preprocessor = Preprocessor()

    # preprocessor.extract_regional_means()
    # preprocessor.create_cyclical_encode()
    # preprocessor.create_local_datetime()
    # preprocessor.filter_by_date_range()
