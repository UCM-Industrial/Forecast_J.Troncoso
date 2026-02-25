"""RECAST — Dataset loader.

Load GRIB and NetCDF datasets via xarray, with automatic
time-coordinate standardisation.

Adapted from legacy ``preprocessor.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import xarray as xr

from src.utils.logger import get_logger

logger = get_logger("preprocessing.loader")

# Supported file extensions → xarray engine mapping
_ENGINES: dict[str, str] = {
    ".nc": "netcdf4",
    ".grib": "cfgrib",
    ".grb": "cfgrib",
    ".grb2": "cfgrib",
    ".grib2": "cfgrib",
}


def load_dataset(filepath: str | Path, **kwargs: object) -> xr.Dataset:
    """Load a climate dataset from a GRIB or NetCDF file.

    Automatically selects the appropriate xarray engine based on
    the file extension.

    Args:
        filepath: Path to the input file.
        **kwargs: Forwarded to ``xr.open_dataset``.

    Returns:
        An xarray ``Dataset``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        msg = f"File not found: {filepath}"
        raise FileNotFoundError(msg)

    engine = _ENGINES.get(filepath.suffix)
    if engine is None:
        msg = (
            f"Unsupported file extension: '{filepath.suffix}'. "
            f"Supported: {', '.join(_ENGINES)}"
        )
        raise ValueError(msg)

    # Allow caller to override engine
    engine = kwargs.pop("engine", engine)

    logger.info("Loading dataset from %s (engine=%s)", filepath.name, engine)
    return xr.open_dataset(filepath, engine=engine, **kwargs)


def standardize_time_coord(
    da: xr.DataArray,
    dataset: xr.Dataset | None = None,
    custom_time: str | None = None,
) -> xr.DataArray:
    """Standardise the time coordinate to ``"datetime"``.

    Handles multiple conventions found in ECMWF GRIB files:

    - ``time`` as a simple dimension
    - ``step`` with scalar ``time`` (forecast single-init)
    - ``time`` + ``step`` + ``valid_time`` (forecast multi-init)
    - Already-standardised ``datetime`` dimension

    Args:
        da: DataArray to standardise.
        dataset: Optional parent Dataset for coordinate lookup.
        custom_time: Name of a custom time coordinate to rename.

    Returns:
        DataArray with a ``datetime`` dimension.

    Raises:
        ValueError: If no recognisable time coordinate is found.
    """
    # Combine available coordinates
    da_coords = set(da.coords)
    if dataset is not None:
        da_coords |= set(dataset.coords)

    dims = list(da.dims)

    # Case 1: Custom time coordinate
    if custom_time and custom_time in da_coords:
        if custom_time in da.coords:
            return da.rename({custom_time: "datetime"})
        return da.assign_coords(datetime=dataset[custom_time])

    # Case 2: step dimension with scalar time (e.g. aifs-single)
    if "step" in dims and "time" in da.coords:
        if da["time"].ndim == 0:
            valid_time = da["time"] + da["step"]
            return da.assign_coords(datetime=valid_time).swap_dims(
                {"step": "datetime"},
            )

    # Case 3: Forecast with time, step, and valid_time
    if {"time", "step", "valid_time"}.issubset(da_coords):
        if "time" in dims and "step" in dims:
            da_stacked = da.stack(forecast_time=("time", "step"))
            source = dataset if dataset is not None else da
            valid_times = source["valid_time"].stack(
                forecast_time=("time", "step"),
            )
            da_result = da_stacked.assign_coords(
                datetime=valid_times,
            ).swap_dims({"forecast_time": "datetime"})
            drop_coords = [
                c for c in ("forecast_time", "time", "step") if c in da_result.coords
            ]
            return da_result.drop_vars(drop_coords)

        if "valid_time" in dims:
            return da.rename({"valid_time": "datetime"})
        if "valid_time" in da.coords:
            return da.assign_coords(datetime=da["valid_time"])

    # Case 4: Simple time dimension
    if "time" in dims:
        return da.rename({"time": "datetime"})

    # Case 5: Already standardised
    if "datetime" in dims or "datetime" in da.coords:
        return da

    # Case 6: Unresolvable
    msg = (
        f"Could not find a recognised time coordinate.\n"
        f"  dims: {dims}\n"
        f"  coords: {sorted(da_coords)}"
    )
    raise ValueError(msg)
