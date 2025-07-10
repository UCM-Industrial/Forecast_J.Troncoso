import json
import logging
import pickle
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

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
HOUR = "06"  # "00", "06", "12", "18"

BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
ARCHIVE_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"

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
    "subregion": "on",
    "leftlon": "-76",
    "rightlon": "-66",
    "toplat": "-15",
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
    future_hours: int = 384,
    delay_seconds: float = 1.0,
) -> list[Path]:
    """Request climate data from GFS from today to `future_hours` ahead."""
    downloaded_files = []

    # GFS: hourly until 120, then every 3 hours until 384
    hours = list(range(0, 121, 1)) + list(range(123, future_hours + 1, 3))
    forecast_steps = [f"{h:03d}" for h in hours]

    for step in forecast_steps:
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
            response = requests.get(BASE_URL, params=params, timeout=(10, 2))
            print(response.url)
            if (
                response.status_code == 200 and len(response.content) > 10_000
            ):  # avoid saving HTML error pages
                dest.write_bytes(response.content)
                downloaded_files.append(dest)
                # print(f"Downloaded: {filename}")
            else:
                print(
                    f"Skipped (bad response): {filename} | Status: {response.status_code} | Size: {len(response.content)} bytes",
                )

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            continue
        time.sleep(delay_seconds)

    return downloaded_files


# WARNING: Deprecated because it is unnecesary
def create_netcdf_file_optimized(
    temp_dir: Path,
    grib_files: list[Path] | None = None,
    output_path: Path | None = None,
    chunk_size: int = 100,
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

    # Optimized compression settings
    if compression is None:
        compression = {"zlib": True, "complevel": 1, "shuffle": True}

    logger.info(f"Processing {len(grib_files)} GRIB files in batches of {chunk_size}")

    # Use xarray's open_mfdataset for efficient multi-file handling
    try:
        # Open all files at once with lazy loading - much more efficient
        dataset = xr.open_mfdataset(
            grib_files,
            engine="cfgrib",
            concat_dim="time",
            combine="nested",
            parallel=True,  # Use parallel processing
            chunks={"time": chunk_size},  # Chunk along time dimension
            backend_kwargs={"indexpath": ""},
            preprocess=lambda ds: ds.sortby("time"),  # Sort each file by time
        )

        # Configure encoding for all variables
        encoding = {}
        for var in dataset.data_vars:
            encoding[var] = {
                **compression,
                "chunksizes": tuple(
                    min(chunk_size, dataset.sizes[dim])
                    if dim == "time"
                    else dataset.sizes[dim]
                    for dim in dataset[var].dims
                ),
            }

        # Save directly to NetCDF - no need for intermediate concatenations
        logger.info(f"Saving NetCDF file: {output_path}")
        dataset.to_netcdf(output_path, encoding=encoding, engine="netcdf4")

        # Clean up
        dataset.close()

        logger.info("NetCDF file created successfully")
        return output_path

    except Exception as e:
        logger.error(f"Error creating NetCDF file: {e}")
        logger.info("Falling back to batch processing method...")

        # Fallback to original batch method if open_mfdataset fails
        return _create_netcdf_batch_method(
            grib_files,
            output_path,
            chunk_size,
            compression,
        )


# WARNING: Deprecated because it is unnecesary
def _create_netcdf_batch_method(
    grib_files: list[Path],
    output_path: Path,
    chunk_size: int,
    compression: dict,
) -> Path:
    """Fallback batch processing method."""
    import gc

    all_datasets = []
    failed_files = []

    for i in range(0, len(grib_files), chunk_size):
        batch = grib_files[i : i + chunk_size]
        logger.info(f"Processing batch {i // chunk_size + 1}")

        batch_datasets = []
        for file_path in batch:
            try:
                ds = xr.open_dataset(
                    file_path,
                    engine="cfgrib",
                    backend_kwargs={"indexpath": ""},
                    chunks="auto",
                )
                batch_datasets.append(ds)
            except Exception as e:
                logger.warning(f"Error reading {file_path.name}: {e}")
                failed_files.append(file_path)

        if batch_datasets:
            # Concatenate batch
            batch_concat = xr.concat(batch_datasets, dim="time")
            all_datasets.append(batch_concat)

            # Close individual datasets immediately
            for ds in batch_datasets:
                ds.close()

            # Force garbage collection
            gc.collect()

    if not all_datasets:
        raise ValueError("No datasets could be loaded from GRIB files")

    # Final concatenation and save
    logger.info("Performing final concatenation")
    final_dataset = xr.concat(all_datasets, dim="time").sortby("time")

    # Configure encoding
    encoding = {var: compression.copy() for var in final_dataset.data_vars}

    # Save to NetCDF
    logger.info(f"Saving NetCDF file: {output_path}")
    final_dataset.to_netcdf(output_path, encoding=encoding, engine="netcdf4")

    # Clean up
    final_dataset.close()
    for ds in all_datasets:
        ds.close()

    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files")

    return output_path


def create_netcdf_file(
    temp_dir: Path,
    grib_files: list[Path] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Create NetCDF file from GRIB files.

    Args:
        temp_dir: Directory containing GRIB files
        grib_files: List of GRIB files to process. If None, searches temp_dir
        output_path: Output NetCDF file path. If None, saves to temp_dir

    Returns:
        Path to created NetCDF file
    """
    if grib_files is None:
        grib_files = sorted(temp_dir.glob("*.f???"))

    if not grib_files:
        raise ValueError("No GRIB files found to process")

    if output_path is None:
        output_path = temp_dir / "forecast_14days.nc"

    logger.info(f"Processing {len(grib_files)} GRIB files")

    # Simple approach: open all files and concatenate
    try:
        dataset = xr.open_mfdataset(
            grib_files,
            engine="cfgrib",
            concat_dim="time",
            combine="nested",
            backend_kwargs={"indexpath": ""},
            preprocess=lambda ds: ds.sortby("time")
            if "time" in ds.dims and ds.time.ndim == 1
            else ds,
        )
    except Exception as e:
        logger.warning(f"Failed with sortby, trying without preprocessing: {e}")
        dataset = xr.open_mfdataset(
            grib_files,
            engine="cfgrib",
            concat_dim="time",
            combine="nested",
            backend_kwargs={"indexpath": ""},
        )

    # Save to NetCDF with basic compression
    logger.info(f"Saving NetCDF file: {output_path}")
    dataset.to_netcdf(
        output_path,
        encoding={var: {"zlib": True} for var in dataset.data_vars},
        engine="netcdf4",
    )

    dataset.close()
    logger.info("NetCDF file created successfully")
    return output_path


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
    n_generation_plants: int | None = None,
    cyclical_features: list[str] = ["hour", "month", "day", "dayofweek"],
) -> pd.DataFrame:
    """Prepare features for ML model ingestion."""
    if technology.lower() == "wind":
        return _prepare_wind_features(csv_path, cyclical_features, n_generation_plants)
    elif technology.lower() == "solar":
        return _prepare_solar_features(csv_path, cyclical_features, n_generation_plants)
    else:
        raise ValueError(f"Unsupported technology: {technology}")


def _prepare_wind_features(
    csv_path: Path,
    cyclical_features: list[str],
    n_plants: int | None = None,
) -> pd.DataFrame:
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

    if n_plants:
        wind_data["total_plants"] = n_plants

    return create_cyclical_features(
        wind_data,
        features=cyclical_features,
    )


def _prepare_solar_features(
    csv_path: Path,
    cyclical_features: list[str],
    n_plants: int | None = None,
) -> pd.DataFrame:
    """Prepare solar-specific features."""
    dswrf_path = csv_path / "sdswrf_means.csv"

    if not dswrf_path.exists():
        raise FileNotFoundError(f"Solar radiation file not found: {dswrf_path}")

    dswrf = read_csv_with_datetime(dswrf_path)

    if n_plants:
        dswrf["total_plants"] = n_plants

    return create_cyclical_features(
        dswrf,
        features=cyclical_features,
    )


def load_model(model_path: Path) -> Any:
    """Loads a serialized model from a pickle file."""
    # TODO: add digital signature to load serialized files
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_model_bundle(bundle_path: Path) -> tuple[Any, dict] | tuple[None, None]:
    """Carga un modelo y sus metadatos desde un archivo .zip."""
    if not bundle_path.exists():
        return None, None

    with ZipFile(bundle_path, "r") as zip_file:
        # Cargar metadatos
        metadata_bytes = zip_file.read("metadata.json")
        metadata = json.loads(metadata_bytes.decode("utf-8"))

        # Cargar modelo
        model_bytes = zip_file.read("model.pkl")
        model = pickle.loads(model_bytes)

    return model, metadata


def predict(model: Any, data: pd.DataFrame) -> pd.DataFrame:
    """Runs prediction using a loaded model on the given dataframe."""
    predictions = model.predict(data)
    results = pd.DataFrame(predictions, index=data.index, columns=["prediction"])
    return results


def run_etl_pipeline(temporal_path: Path) -> tuple[Path, Path]:
    download_gfs_data(temporal_path / "download")
    create_netcdf_file(
        temp_dir=temporal_path / "download",
        output_path=temporal_path / "forecast_14days.nc",
    )

    # Create csv's for the ML models
    logger.info("Creating wind regions")
    create_regional_csv(
        temporal_path / "forecast_14days.nc",
        temporal_path / "wind",
        [
            "u100",
            "v100",
        ],
    )

    logger.info("Creating solar regions")
    create_regional_csv(
        temporal_path / "forecast_14days.nc",
        temporal_path / "solar",
        [
            "sdswrf",
        ],
    )
    wind_df = prepare_ml_features(
        temporal_path / "wind",
        technology="wind",
    )
    solar_df = prepare_ml_features(
        temporal_path / "solar",
        technology="solar",
    )
    wind_data_path = temporal_path / "wind_features.csv"
    solar_data_path = temporal_path / "solar_features.csv"

    wind_df.to_csv(wind_data_path)
    solar_df.to_csv(solar_data_path)

    return (wind_data_path, solar_data_path)


if __name__ == "__main__":
    temporal_path = Path().cwd() / "forecast_temp"
    temporal_path.mkdir(exist_ok=True)

    print(download_gfs_data(temporal_path))

    # wind_data_path, solar_data_path = run_etl_pipeline(temporal_path)
    # wind_data_path = (
    #     "/home/kyoumas/repos/ts_energy_patterns/forecast_temp/wind_features.csv"
    # )

    # wind_data = read_csv_with_datetime(wind_data_path)
    # solar_model = load_model(temporal_path / "models" / "solar.pkl")
    # wind_model = load_model(temporal_path / "models" / "eolica.pkl")

    # print(wind_model.predict(wind_data))
