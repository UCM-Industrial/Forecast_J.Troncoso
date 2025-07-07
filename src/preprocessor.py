import gc
import tempfile
from pathlib import Path

import dask
import geopandas as gpd
import leafmap.foliumap as leafmap
import pandas as pd
import regionmask
import xarray as xr


def load_dataset(filepath: str | Path, **kwargs) -> xr.Dataset:
    """Load a dataset from a NetCDF or GRIB file.

    Args:
        filepath: Path to the input file (.nc, .grib, .grb, .grb2).
        **kwargs: Additional keyword arguments for xarray.open_dataset.

    Returns:
        An xarray.Dataset object.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Determine engine based on file extension
    if filepath.suffix == ".nc":
        engine = kwargs.pop("engine", "netcdf4")
    elif filepath.suffix in [".grib", ".grb", ".grb2", ".grib2"]:
        engine = kwargs.pop("engine", "cfgrib")
    else:
        raise ValueError(
            f"Unsupported file extension: '{filepath.suffix}'. "
            "Please provide a .nc or .grib file.",
        )

    return xr.open_dataset(filepath, engine=engine, **kwargs)


def mask_regions(
    _da: xr.DataArray,
    _gdf: gpd.GeoDataFrame | None,
    *,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
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

    return regions.mask(_da[lon_name], _da[lat_name])


def extract_regional_means(
    _ds: xr.Dataset,
    _gdf: gpd.GeoDataFrame,
    data_variable: str,
    *,
    time_coord: str | None = None,
    latitude: str = "latitude",
    longitude: str = "longitude",
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
    mask = mask_regions(
        da,
        _gdf,
        names=column_names,
        lat_name=latitude,
        lon_name=longitude,
    )
    regional_means = da.groupby(mask).mean(dim=[latitude, longitude])

    # NOTE: This is a temporal fix
    if "isobaricInhPa" in regional_means.dims:
        # regional_means = regional_means.isel(isobaricInhPa=0)  # select level
        regional_means = regional_means.mean(dim="isobaricInhPa")  # average over levels

    # Convert to DataFrame
    df = regional_means.to_pandas()

    if hasattr(df.index, "to_datetimeindex"):
        df.index = df.index.to_datetimeindex()

    df.index = df.index.tz_localize("UTC").tz_convert(output_timezone)

    # gdf = _gdf.dropna(subset=[column_names])
    region_names = dict(enumerate(_gdf[column_names]))
    df.columns = df.columns.map(region_names)

    df = df.dropna(how="all")

    return df.sort_index()


def _standardize_time_coord(
    da: xr.DataArray,
    dataset: xr.Dataset | None = None,
    custom_time: str | None = None,
    warn_only: bool = False,
) -> xr.DataArray:
    """Standardize time coordinate to 'datetime', using da or dataset context."""
    # Use dataset context to look for time coordinates if missing in da
    if dataset is not None:
        da_coords = set(da.coords) | set(dataset.coords)
    else:
        da_coords = set(da.coords)

    dims = list(da.dims)

    # Case 1: Custom time coordinate
    if custom_time and custom_time in da_coords:
        if custom_time in da.coords:
            return da.rename({custom_time: "datetime"})
        else:
            return da.assign_coords(datetime=dataset[custom_time])

    # Case 2: Forecast-style
    if "valid_time" in da.coords:
        da = da.rename({"time": "datetime"})
        da = da.assign_coords(datetime=da["valid_time"])
        if "valid_time" in da.coords:
            da = da.drop_vars("valid_time")
        return da

    # Case 3: Forecast-style with time/step/valid_time (stacking)
    if {"time", "step", "valid_time"}.issubset(da_coords):
        if "time" in dims and "step" in dims:
            da_stacked = da.stack(forecast_time=("time", "step"))
            valid_times = (
                dataset["valid_time"].stack(forecast_time=("time", "step"))
                if dataset is not None
                else da["valid_time"].stack(forecast_time=("time", "step"))
            )
            da = da_stacked.assign_coords(datetime=valid_times).swap_dims(
                {"forecast_time": "datetime"},
            )
            drop_coords = ["forecast_time"] + [
                c for c in ("time", "step") if c in da.coords
            ]
            return da.drop_vars([c for c in drop_coords if c in da.coords])

    # Case 4: Simple time dimension
    if "time" in dims:
        return da.rename({"time": "datetime"})

    # Case 5: Already standardized
    if "datetime" in dims or "datetime" in da.coords:
        return da

    # Case 6: Cannot resolve
    msg = (
        f"⛔ No reconocida coordenada de tiempo.\n"
        f"  - dims: {dims}\n"
        f"  - coordenadas: {sorted(da_coords)}"
    )
    if warn_only:
        import warnings

        warnings.warn(msg)
        return da
    raise ValueError(msg)


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


def export_dataarray_to_tif(
    da: xr.DataArray,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> str:
    """Converts a georeferenced DataArray to a temporary GeoTIFF and returns its path."""
    da.rio.set_spatial_dims(
        x_dim=lon_name,
        y_dim=lat_name,
        inplace=True,
    )  # or "lon"/"lat" depending on your GRIB
    da.rio.write_crs("EPSG:4326", inplace=True)

    temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    da.rio.to_raster(temp_file.name)
    return temp_file.name


def display_dataarray(
    m: leafmap.Map,
    da: xr.DataArray,
    rescale: str = "0,5000",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
):
    """Converts and displays an xarray DataArray on the map."""
    tif_path = export_dataarray_to_tif(
        da,
        lat_name=lat_name,
        lon_name=lon_name,
    )
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
            chunk_size={"latitude": 25, "longitude": 25},
            column_names="Comuna",
        )

        new_path = Path(path)

        filename = new_path / f"{var}_means.csv"
        temp_df.to_csv(filename)


def convert_ssrd_to_watts(
    da: xr.DataArray,
    accumulation_hours: int = 1,
) -> xr.DataArray:
    """Convert SSRD from J/m² to W/m² with lazy evaluation."""
    # Convert from J/m² to W/m² using lazy operations
    seconds_per_accumulation = accumulation_hours * 3600

    # Create lazy computation
    da_watts = da / seconds_per_accumulation

    # Update attributes (this doesn't trigger computation)
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


def process_dataset_optimized(
    ds: xr.Dataset,
    variables: list[str],
    path: str | Path,
    regions_gdf: gpd.GeoDataFrame,
    column_names: str = "Comuna",
    batch_size: int = 1,
) -> list[Path] | None:
    """Process dataset with memory optimization and batch processing."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    created_files = []

    # Process variables in batches to manage memory
    for i in range(0, len(variables), batch_size):
        batch_vars = variables[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1}: {batch_vars}")

        for var in batch_vars:
            try:
                # Extract only the needed variable
                var_ds = ds[[var]]

                temp_df = extract_regional_means(
                    _ds=var_ds,
                    _gdf=regions_gdf,
                    data_variable=var,
                    column_names=column_names,
                )

                filename = path / f"{var}_means.csv"
                temp_df.to_csv(filename)
                created_files.append(filename)

                # Clean up
                del temp_df, var_ds
                gc.collect()

            except Exception as e:
                print(f"Error processing {var}: {e}")
                continue

    return created_files


if __name__ == "__main__":
    dask.config.set(
        {
            "array.chunk-size": "128 MB",
            "distributed.worker.memory.target": 0.6,
            "distributed.worker.memory.spill": 0.7,
            "distributed.worker.memory.pause": 0.8,
            "distributed.worker.memory.terminate": 0.9,
        },
    )
    # --- Declare Paths ---
    data_path = Path().cwd() / "data"

    input_path = data_path / "1_preprocessing" / "input"
    output_path = "/home/kyoumas/repos/ts_energy_patterns/data/2_pre-out_dec-in/hourly/"
    ssrd_path = (
        "/home/kyoumas/repos/ts_energy_patterns/data/1_pre-in/hourly/ssrd_hourly.grib"
    )
    regions_chile_path = (
        "/home/kyoumas/repos/ts_energy_patterns/data/1_pre-in/masks/Comunas/comunas.shp"
    )

    # --- Define test data ---
    print("Loading regions...")
    regions_chile = gpd.read_file(regions_chile_path)

    print("Opening dataset with chunking...")
    weather = load_dataset(
        ssrd_path,
        decode_timedelta=False,
        chunks="auto",  # Let xarray determine optimal chunking
    )

    variables_to_extract = ["ssrd"]

    print("Filtering variables...")
    weather_filtered = weather[variables_to_extract]

    print("Converting to Watts (lazy operation)...")
    # Convert the DataArray, not the Dataset
    ssrd_watts_da = convert_ssrd_to_watts(
        weather_filtered["ssrd"],
        accumulation_hours=1,
    )

    # Create new dataset with converted variable
    ssrd_watts = xr.Dataset({"ssrd": ssrd_watts_da})

    print("Processing dataset with optimization...")
    process_dataset_optimized(
        ssrd_watts,
        variables=variables_to_extract,
        path=output_path,
        regions_gdf=regions_chile,
        column_names="Comuna",
        batch_size=1,
    )

    print("Processing complete!")
