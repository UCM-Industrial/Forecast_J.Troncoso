"""RECAST — Geospatial processing.

Extract regional mean values from gridded climate data using
geographic masks (shapefiles).

Adapted from legacy ``preprocessor.py``.
"""

from __future__ import annotations

import gc
from pathlib import Path

import geopandas as gpd
import pandas as pd
import regionmask
import xarray as xr

from src.preprocessing.loader import standardize_time_coord
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("preprocessing.geospatial")


def mask_regions(
    da: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    *,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
    names: str = "Region",
    overlap: bool = False,
) -> xr.DataArray:
    """Create a spatial mask from a GeoDataFrame.

    Args:
        da: DataArray with spatial coordinates.
        gdf: GeoDataFrame defining geographic regions.
        lon_name: Name of the longitude coordinate.
        lat_name: Name of the latitude coordinate.
        names: Column in ``gdf`` to use as region names.
        overlap: Allow overlapping regions.

    Returns:
        Integer mask DataArray.

    Raises:
        ValueError: If ``gdf`` is None.
    """
    if gdf is None:
        msg = "No region mask provided"
        raise ValueError(msg)

    gdf_projected = gdf
    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf_projected = gdf.to_crs("EPSG:4326")

    regions = regionmask.from_geopandas(gdf_projected, names=names, overlap=overlap)
    return regions.mask(da[lon_name], da[lat_name])


def extract_regional_means(
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    data_variable: str,
    *,
    time_coord: str | None = None,
    latitude: str = "latitude",
    longitude: str = "longitude",
    chunk_size: dict[str, int] | None = None,
    column_names: str = "Region",
    output_timezone: str = "America/Santiago",
) -> pd.DataFrame:
    """Calculate regional means for a climate variable.

    Applies a geographic mask to grid data and computes the spatial
    mean for each region at every time step.

    Args:
        ds: Input xarray Dataset.
        gdf: GeoDataFrame of regions.
        data_variable: Variable name in the dataset.
        time_coord: Custom time coordinate name.
        latitude: Latitude coordinate name in the dataset.
        longitude: Longitude coordinate name in the dataset.
        chunk_size: Dask chunk dimensions for memory optimisation.
        column_names: GeoDataFrame column for region names.
        output_timezone: Target timezone for the output index.

    Returns:
        DataFrame with datetime index and one column per region.

    Raises:
        ValueError: If ``data_variable`` is not found.
    """
    if chunk_size is None:
        settings = get_settings()
        chunk_size = settings.geospatial.chunk_size

    if data_variable not in ds.data_vars:
        msg = f"'{data_variable}' is not a variable in the dataset. Available: {list(ds.data_vars)}"
        raise ValueError(msg)

    da = ds[data_variable]

    # Apply chunking for memory optimisation
    if chunk_size:
        da = da.chunk(chunk_size)

    # Standardise time coordinate
    da = standardize_time_coord(da, custom_time=time_coord)

    # Create spatial mask and compute regional means
    mask = mask_regions(
        da,
        gdf,
        names=column_names,
        lat_name=latitude,
        lon_name=longitude,
    )

    logger.info("Computing regional means for '%s'…", data_variable)
    regional_means = da.groupby(mask).mean(dim=[latitude, longitude])

    # Handle extra dimensions (e.g. isobaricInhPa)
    if "isobaricInhPa" in regional_means.dims:
        regional_means = regional_means.mean(dim="isobaricInhPa")

    # Convert to DataFrame
    df = regional_means.to_pandas()
    df.index = df.index.tz_localize("UTC").tz_convert(output_timezone)

    # Map integer region IDs to names
    region_names = dict(enumerate(gdf[column_names]))
    df.columns = df.columns.map(region_names)

    df = df.dropna(how="all").sort_index()
    logger.info(
        "Extracted %d regions × %d timesteps for '%s'",
        len(df.columns),
        len(df),
        data_variable,
    )
    return df


def create_regional_csvs(
    ds: xr.Dataset,
    regions_gdf: gpd.GeoDataFrame,
    variables: list[str],
    output_dir: str | Path,
    *,
    column_names: str = "Region",
) -> list[Path]:
    """Process multiple variables into per-region CSV files.

    Args:
        ds: Input xarray Dataset.
        regions_gdf: GeoDataFrame of geographic regions.
        variables: List of variable names to extract.
        output_dir: Directory to save output CSVs.
        column_names: GeoDataFrame column for region names.

    Returns:
        List of paths to created CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files: list[Path] = []

    for var in variables:
        try:
            logger.info("Processing variable '%s'…", var)
            var_ds = ds[[var]]

            df = extract_regional_means(
                ds=var_ds,
                gdf=regions_gdf,
                data_variable=var,
                column_names=column_names,
            )

            filepath = output_dir / f"{var}_means.csv"
            df.to_csv(filepath)
            created_files.append(filepath)
            logger.info("Saved %s", filepath)

            # Free memory
            del df, var_ds
            gc.collect()

        except Exception:
            logger.exception("Failed to process variable '%s'", var)

    return created_files
