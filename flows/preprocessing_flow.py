"""RECAST — Preprocessing flow (Prefect).

Flow to transform raw GRIB files into ML-ready feature CSVs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import geopandas as gpd
from prefect import flow, task
from prefect.logging import get_run_logger

from src.preprocessing.features import prepare_ml_features
from src.preprocessing.geospatial import create_regional_csvs
from src.preprocessing.loader import load_dataset
from src.utils.config import get_settings
from src.utils.gcs import download_blob, upload_blob
from src.utils.logger import setup_logging


@task(name="download-raw-data")
def download_raw(date: str) -> Path:
    """Download raw GRIB from storage to a temp location."""
    settings = get_settings()
    remote = f"{settings.gcs.prefixes.raw}/{date}/aifs_single_{date}.grib2"
    local = Path(f"tmp/raw/{date}/aifs_single_{date}.grib2")
    download_blob(remote, local)
    return local


@task(name="extract-regional-means")
def extract_means(grib_path: Path, date: str) -> Path:
    """Load GRIB and extract regional means to CSV."""
    logger = get_run_logger()
    settings = get_settings()

    ds = load_dataset(grib_path)
    gdf = gpd.read_file(settings.geospatial.regions_shapefile)

    output_dir = Path(f"tmp/processed/{date}")
    create_regional_csvs(
        ds=ds,
        regions_gdf=gdf,
        variables=settings.ecmwf.variables,
        output_dir=output_dir,
        column_names=settings.geospatial.segment_by,
    )

    logger.info("Regional means saved to %s", output_dir)
    return output_dir


@task(name="build-features")
def build_features(csv_dir: Path, technology: str, date: str) -> Path:
    """Build ML features from regional mean CSVs."""
    logger = get_run_logger()
    settings = get_settings()

    features_df = prepare_ml_features(csv_dir, technology)

    output_path = Path(f"tmp/features/{date}/{technology}_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path)

    # Upload to storage
    remote = f"{settings.gcs.prefixes.processed}/{date}/{technology}_features.csv"
    upload_blob(output_path, remote)

    logger.info("Features for '%s' → %s", technology, remote)
    return output_path


@flow(name="preprocessing", log_prints=True)
def preprocessing_flow(date: str | None = None) -> dict[str, Path]:
    """Transform raw GRIB data into ML-ready features.

    Args:
        date: Date string (``YYYYMMDD``). Defaults to today.

    Returns:
        Dict mapping technology to feature file paths.
    """
    setup_logging()
    logger = get_run_logger()

    date = date or datetime.now().strftime("%Y%m%d")
    settings = get_settings()
    logger.info("=== Preprocessing Flow — date=%s ===", date)

    # Step 1: Download raw data
    grib_path = download_raw(date)

    # Step 2: Extract regional means
    csv_dir = extract_means(grib_path, date)

    # Step 3: Build features for each technology
    results: dict[str, Path] = {}
    for tech in settings.prediction.technologies:
        results[tech] = build_features(csv_dir, tech, date)

    logger.info("=== Preprocessing Flow complete ===")
    return results


if __name__ == "__main__":
    preprocessing_flow()
