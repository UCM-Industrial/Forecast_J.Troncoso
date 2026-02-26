"""RECAST — Preprocessing flow (Prefect).

Flow to transform raw **AIFS-single** GRIB files into
ML-ready feature CSVs for batch **prediction**.

For training data preparation, see ``training_data_flow.py``.
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


@task(name="download-aifs-raw")
def download_raw(date: str) -> Path:
    """Download raw AIFS GRIB from storage to a temp location."""
    settings = get_settings()
    remote = f"{settings.gcs.prefixes.raw_aifs}/{date}/aifs_single_{date}.grib2"
    local = Path(f"tmp/aifs/{date}/aifs_single_{date}.grib2")
    download_blob(remote, local)
    return local


@task(name="extract-regional-means")
def extract_means(grib_path: Path, date: str) -> Path:
    """Load GRIB and extract regional means to CSV."""
    logger = get_run_logger()
    settings = get_settings()

    ds = load_dataset(grib_path)
    gdf = gpd.read_file(settings.geospatial.regions_shapefile)

    output_dir = Path(f"tmp/prediction/{date}")
    create_regional_csvs(
        ds=ds,
        regions_gdf=gdf,
        variables=settings.aifs.variables,
        output_dir=output_dir,
        column_names=settings.geospatial.segment_by,
    )

    logger.info("Regional means saved to %s", output_dir)
    return output_dir


@task(name="build-prediction-features")
def build_features(csv_dir: Path, technology: str, date: str) -> Path:
    """Build ML features from regional mean CSVs for prediction."""
    logger = get_run_logger()
    settings = get_settings()

    features_df = prepare_ml_features(csv_dir, technology)

    output_path = Path(f"tmp/prediction/{date}/{technology}_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path)

    remote = f"{settings.gcs.prefixes.processed_prediction}/{date}/{technology}_features.csv"
    upload_blob(output_path, remote)

    logger.info("Prediction features for '%s' → %s", technology, remote)
    return output_path


@flow(name="prediction-preprocessing", log_prints=True)
def preprocessing_flow(date: str | None = None) -> dict[str, Path]:
    """Transform AIFS-single data into prediction-ready features.

    Args:
        date: Date string (``YYYYMMDD``). Defaults to today.

    Returns:
        Dict mapping technology to feature file paths.
    """
    setup_logging()
    logger = get_run_logger()

    date = date or datetime.now().strftime("%Y%m%d")
    settings = get_settings()
    logger.info("=== Prediction Preprocessing Flow — date=%s ===", date)

    grib_path = download_raw(date)
    csv_dir = extract_means(grib_path, date)

    results: dict[str, Path] = {}
    for tech in settings.prediction.technologies:
        results[tech] = build_features(csv_dir, tech, date)

    logger.info("=== Prediction Preprocessing Flow complete ===")
    return results


if __name__ == "__main__":
    preprocessing_flow()
