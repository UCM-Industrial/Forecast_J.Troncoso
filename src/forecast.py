import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import xarray as xr

from _util import create_cyclical_features, read_csv_with_datetime
from preprocessor import load_dataset, process_dataset_optimized

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATE = datetime.utcnow().strftime("%Y%m%d")
HOUR = "00"  # "00", "06", "12", "18"

BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
VARS = {
    "var_DSWRF": "on",
    "var_UGRD": "on",
    "var_VGRD": "on",
}
LEVELS = {
    "lev_100_m_above_ground": "on",
    "lev_surface": "on",
}
REGION = {
    "toplat": "-15",
    "leftlon": "-76",
    "rightlon": "-66",
    "bottomlat": "-56",
}
OUT_NETCDF = "forecast_14days.nc"

# TODO: Automatize a default temp directory to save the temporal files
MASK_PATHS = {
    "Comuna": "/home/kyoumas/repos/ts_energy_patterns/data/1_pre-in/masks/Comunas/comunas.shp",
    "Regional": "/home/kyoumas/repos/ts_energy_patterns/data/1_pre-in/masks/Regiones/Regional.shp",
    "Provincia": "/home/kyoumas/repos/ts_energy_patterns/data/1_pre-in/masks/Provincias/Provincias.shp",
}


def create_temp_directory(prefix: str = "forecast_temp") -> Path:
    """Create a temporary directory for processing."""
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    return temp_dir


def cleanup_temp_directory(temp_dir: Path) -> None:
    """Clean up temporary directory and all its contents."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def download_gfs_data(
    temp_dir: Path,
    date: str = DATE,
    hour: str = HOUR,
    future_hours: int = 380,
):
    """Request climate data from GSF from today (now) to `future_hours` ahead."""
    downloaded_files = []

    for i in range(1, future_hours):
        step = f"{i:03d}"
        filename = f"gfs.t{hour}z.pgrb2.0p25.f{step}"
        dir_param = f"/gfs.{date}/{hour}/atmos"

        params = {
            "file": filename,
            "dir": dir_param,
            **VARS,
            **LEVELS,
            **REGION,
        }

        dest = temp_dir / filename

        try:
            response = requests.get(BASE_URL, params=params, timeout=5)
            dest.write_bytes(response.content)
            downloaded_files.append(dest)
            print(f"{filename}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e})")
            continue

    return downloaded_files


def create_netcdf_file_optimized(
    temp_dir: Path,
    grib_files: list[Path] | None = None,
    output_path: Path | None = None,
    chunk_size: int = 50,
    compression: dict | None = None,
) -> Path:
    """Create NetCDF file from GRIB files with memory optimization.

    Args:
        temp_dir: Directory containing GRIB files
        grib_files: List of GRIB files to process. If None, searches temp_dir
        output_path: Output NetCDF file path. If None, saves to temp_dir
        chunk_size: Number of files to process in each batch
        compression: Compression settings for NetCDF

    Returns:
        Path to created NetCDF file
    """
    if grib_files is None:
        grib_files = sorted(temp_dir.glob("*.f???"))

    if not grib_files:
        raise ValueError("No GRIB files found to process")

    if output_path is None:
        output_path = temp_dir / "forecast_14days.nc"

    # Default compression settings
    if compression is None:
        compression = {"zlib": True, "complevel": 4, "shuffle": True}

    logger.info(f"Processing {len(grib_files)} GRIB files in batches of {chunk_size}")

    # Process files in chunks to avoid memory issues
    datasets = []
    failed_files = []

    for i in range(0, len(grib_files), chunk_size):
        batch = grib_files[i : i + chunk_size]
        logger.info(
            f"Processing batch {i // chunk_size + 1}: files {i + 1}-{min(i + chunk_size, len(grib_files))}",
        )

        batch_datasets = []
        for file_path in batch:
            try:
                # Use chunks='auto' and lazy loading for memory efficiency
                ds = xr.open_dataset(
                    file_path,
                    engine="cfgrib",
                    backend_kwargs={"indexpath": ""},
                    chunks="auto",  # Enable dask for lazy loading
                )
                batch_datasets.append(ds)

            except Exception as e:
                logger.warning(f"Error reading {file_path.name}: {e}")
                failed_files.append(file_path)
                continue

        if batch_datasets:
            # Concatenate batch and append to main list
            try:
                batch_concat = xr.concat(batch_datasets, dim="time")
                datasets.append(batch_concat)

                # Close individual datasets to free memory
                for ds in batch_datasets:
                    ds.close()

            except Exception as e:
                logger.error(f"Error concatenating batch: {e}")
                continue

    if not datasets:
        raise ValueError("No datasets could be loaded from GRIB files")

    # Final concatenation
    logger.info("Performing final concatenation of all batches")
    try:
        final_dataset = xr.concat(datasets, dim="time")

        # Sort by time to ensure proper ordering
        final_dataset = final_dataset.sortby("time")

        # Configure encoding for compression
        encoding = {}
        for var in final_dataset.data_vars:
            encoding[var] = compression.copy()

        # Save to NetCDF with compression
        logger.info(f"Saving NetCDF file: {output_path}")
        final_dataset.to_netcdf(output_path, encoding=encoding, engine="netcdf4")

        # Clean up
        final_dataset.close()
        for ds in datasets:
            ds.close()

        logger.info("NetCDF file created successfully")

        if failed_files:
            logger.warning(
                f"Failed to process {len(failed_files)} files: {[f.name for f in failed_files]}",
            )

        return output_path

    except Exception as e:
        logger.error(f"Error creating final NetCDF file: {e}")
        raise


def create_regional_csv(
    net_cdf_path: str | Path,
    output_path: Path,
    variables: list[str],
    segment_by: str = "Comuna",
) -> list[Path] | None:
    """Read a NetCDF file and segment by regions."""
    regions_gdf = gpd.read_file(MASK_PATHS[segment_by])
    dataset = load_dataset(
        net_cdf_path,
        decode_timedelta=False,
        chunks="auto",
    )
    csv_path = process_dataset_optimized(
        ds=dataset,
        regions_gdf=regions_gdf,
        variables=variables,
        path=output_path,
        column_names=segment_by,
    )
    dataset.close()

    return csv_path


def prepare_ml_features(
    csv_path: Path,
    technology: str,
    n_generation_plants: int,
) -> pd.DataFrame:
    """Prepare features for ML model ingestion."""
    if technology.lower() == "wind":
        return _prepare_wind_features(csv_path, n_generation_plants)
    elif technology.lower() == "solar":
        return _prepare_solar_features(csv_path, n_generation_plants)
    else:
        raise ValueError(f"Unsupported technology: {technology}")


def _prepare_wind_features(csv_path: Path, n_plants: int) -> pd.DataFrame:
    """Prepare wind-specific features."""
    # Assuming CSV path is a directory containing separate files
    u100_path = csv_path / "u100_means.csv"
    v100_path = csv_path / "v100_means.csv"

    if not u100_path.exists() or not v100_path.exists():
        raise FileNotFoundError(f"Wind component files not found in {csv_path}")

    u100 = read_csv_with_datetime(u100_path)
    v100 = read_csv_with_datetime(v100_path)

    # Join wind components
    wind_data = u100.join(
        v100,
        how="inner",
        lsuffix="_u100_means",
        rsuffix="_v100_means",
    )

    # Add metadata
    wind_data["total_plants"] = n_plants

    return create_cyclical_features(wind_data)


def _prepare_solar_features(csv_path: Path, n_plants: int) -> pd.DataFrame:
    """Prepare solar-specific features."""
    dswrf_path = csv_path / "dswrf.csv"

    if not dswrf_path.exists():
        raise FileNotFoundError(f"Solar radiation file not found: {dswrf_path}")

    dswrf = read_csv_with_datetime(dswrf_path)

    # Add metadata
    dswrf["total_plants"] = n_plants

    return create_cyclical_features(dswrf)


def load_model(): ...


def predict(): ...


if __name__ == "__main__":
    temporal_path = Path().cwd() / "forecast_temp"
    temporal_path.mkdir(exist_ok=True)
    # download_gfs_data(temporal_path)
    # create_netcdf_file_optimized(temporal_path)
    # create_regional_csv(
    #     temporal_path / "forecast_14days.nc",
    #     temporal_path / "wind",
    #     [
    #         "u100",
    #         "v100",
    #     ],
    # )
    # create_regional_csv(
    #     temporal_path / "forecast_14days.nc",
    #     temporal_path / "solar",
    #     [
    #         "sdswrf",
    #     ],
    # )

    df = prepare_ml_features(
        temporal_path / "wind",
        technology="wind",
        n_generation_plants=10,
    )
    df.to_csv(temporal_path / "wind_features.csv")
