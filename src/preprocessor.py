import tempfile
from pathlib import Path

import geopandas as gpd
import leafmap.foliumap as leafmap
import numpy as np
import pandas as pd
import regionmask
import xarray as xr


def mask_regions(
    _da: xr.DataArray,
    _gdf: gpd.GeoDataFrame | None,
    **kwargs,
) -> xr.DataArray:
    """Creates a mask for the given regions."""
    mask_gdf = _gdf
    if mask_gdf is None:
        raise ValueError("No region mask provided")

    if mask_gdf.crs != "EPSG:4326":
        mask_gdf = mask_gdf.to_crs("EPSG:4326")

    # Set default values
    # kwargs.setdefault("names", "Region")
    kwargs.setdefault("overlap", False)
    regions = regionmask.from_geopandas(mask_gdf, **kwargs)

    return regions.mask(_da.longitude, _da.latitude)


def extract_regional_means(
    _ds: xr.Dataset,
    _gdf: gpd.GeoDataFrame,
    data_variable: str,
    time_coord: str | None = None,
    chunk_size: dict[str, int] | None = {"latitude": 50, "longitude": 50},
    column_names: str = "Region",
    output_timezone: str = "America/Santiago",
) -> pd.DataFrame:
    """Calculate regional means with memory optimization."""
    if not data_variable or data_variable not in _ds.data_vars:
        raise ValueError(
            f"{data_variable} is not a valid data variable in the dataset.",
        )

    da = _ds[data_variable]

    # Apply chunking for memory optimization
    if chunk_size:
        da = da.chunk(chunk_size)

    # Handle time coordinates
    da = _standardize_time_coord(da, custom_time=time_coord)

    # Calculate regional means
    mask = mask_regions(da, _gdf, names=column_names)
    regional_means = da.groupby(mask).mean(dim=["latitude", "longitude"])

    # NOTE: This is a temporal fix
    if "isobaricInhPa" in regional_means.dims:
        # Choose one:
        # regional_means = regional_means.isel(isobaricInhPa=0)  # select level
        regional_means = regional_means.mean(dim="isobaricInhPa")  # average over levels

    # Convert to DataFrame
    df = regional_means.to_pandas()
    gdf = _gdf.dropna(subset=[column_names])
    region_names = dict(enumerate(_gdf[column_names]))
    df.columns = df.columns.map(region_names)

    df.index = df.index.tz_localize("UTC")

    if output_timezone != "UTC":
        df.index = df.index.tz_convert(output_timezone)

    return df.sort_index()


def create_cyclical_features(
    df: pd.DataFrame,
    datetime_col: str | None = None,
    features: list[str] = ["hour", "day", "month"],
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

    return temp_df


def _standardize_time_coord(
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


# --- Plot functions ---


def create_map(center: list[float] = [-36, -71], zoom: int = 6) -> leafmap.Map:
    """Creates a leafmap.Map object with initial settings."""
    m = leafmap.Map(center=center, zoom=zoom)
    m.add_basemap("CartoDB.DarkMatter")
    return m


def display_mask(m: leafmap.Map, mask_gdf: gpd.GeoDataFrame):
    """Adds a GeoDataFrame mask to the map."""
    m.add_gdf(
        mask_gdf,
        layer_name="Mask",
        style={"color": "white", "weight": 2, "fillOpacity": 0.2},
        info_mode="on_hover",
    )
    # Center the map on the mask's bounds
    m.zoom_to_gdf(mask_gdf)


def display_climate_data(m: leafmap.Map, tif_path: str):
    """Adds a COG layer for climate data to the map."""
    m.add_raster(
        source=tif_path,
        name="Climate Data (COG)",
        palette="viridis",
        rescale="0,5000",  # Rescale pixel values for better visualization
    )


def export_dataarray_to_tif(da: xr.DataArray) -> str:
    """Converts a georeferenced DataArray to a temporary GeoTIFF and returns its path."""
    da.rio.set_spatial_dims(
        x_dim="longitude",
        y_dim="latitude",
        inplace=True,
    )  # or "lon"/"lat" depending on your GRIB
    da.rio.write_crs("EPSG:4326", inplace=True)

    temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    da.rio.to_raster(temp_file.name)
    return temp_file.name


def display_dataarray(m: leafmap.Map, da: xr.DataArray, rescale: str = "0,5000"):
    """Converts and displays an xarray DataArray on the map."""
    tif_path = export_dataarray_to_tif(da)
    m.add_raster(
        source=tif_path,
        name="Climate Data",
        palette="viridis",
        rescale=rescale,
    )


def display_both(m: leafmap.Map, mask_gdf: gpd.GeoDataFrame, tif_path: str):
    """Displays both the mask and the climate data."""
    display_climate_data(m, tif_path)
    display_mask(m, mask_gdf)


# WARNING: Deprecated
def process_dataset(
    ds: xr.Dataset,
    variables: list[str],
    path: str | Path,
) -> None:
    for var in variables:
        temp_df = extract_regional_means(
            _ds=ds,
            _gdf=regions_chile,
            data_variable=var,
            chunk_size={"latitude": 50, "longitude": 50},
        )

        new_path = Path(path)

        filename = new_path / f"{var}_means.csv"
        temp_df.to_csv(filename)


def convert_ssrd_to_watts(
    da: xr.Dataset,
    accumulation_hours: int = 1,
) -> xr.Dataset:
    """Convert SSRD from J/m² to W/m²."""
    # Convert from J/m² to W/m²
    seconds_per_accumulation = accumulation_hours * 3600
    da_watts = da / seconds_per_accumulation

    # Update attributes
    da_watts.attrs = da.attrs.copy()
    da_watts.attrs["units"] = "W m**-2"

    # Update long_name if it exists
    if "long_name" in da_watts.attrs:
        da_watts.attrs["long_name"] = da_watts.attrs["long_name"].replace(
            "J m**-2",
            "W m**-2",
        )

    # Add conversion info to history
    if "history" in da_watts.attrs:
        da_watts.attrs["history"] += (
            f" | Converted from J/m² to W/m² (÷{seconds_per_accumulation}s)"
        )
    else:
        da_watts.attrs["history"] = (
            f"Converted from J/m² to W/m² (÷{seconds_per_accumulation}s)"
        )

    return da_watts


# --- Plotting Functions ---


if __name__ == "__main__":
    # --- Declare Paths ---
    data_path = Path().cwd() / "data"
    test_generation = (
        data_path / "processed" / "public_data" / "generation_historic_tipo.csv"
    )

    input_path = data_path / "1_preprocessing" / "input"
    output_path = data_path / "1_preprocessing" / "output"

    snowmelt_path = input_path / "snowmelt.grib"
    runoff_path = input_path / "runoff.grib"
    regions_chile_path = input_path / "Regiones" / "Regional.shp"

    # --- Define test data ---
    test_da = runoff_path
    test_gdf = regions_chile_path

    regions_chile = gpd.read_file(test_gdf)

    weather = xr.open_dataset(
        test_da,
        engine="cfgrib",
        decode_timedelta=False,
        # filter_by_keys={"shortName": "smlt"},
    )
    print(weather)

    variables_to_extract = ["ro"]
    weather_filtered = weather[variables_to_extract]

    # ssrd_watts = convert_ssrd_to_watts(
    #     weather,
    #     accumulation_hours=1,
    # )

    process_dataset(
        weather_filtered,
        variables=variables_to_extract,
        path=output_path,
    )

    # --- Check datetime format processing ---
    # temp_df = preprocessor.create_local_datetime(generation)
    # temp_df.to_csv("generation_test.csv")
    # logger.info("Datetime formatted done")

    # --- Check cyclical encoding feature engineering ---
    # temp_df = preprocessor.create_cyclical_encode(temp_df)
    # temp_df.to_csv("feature_enginering_test.csv")
    # logger.info("Cyclical encoding done")

    # preprocessor.filter_by_date_range()
