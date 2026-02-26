"""RECAST — Training data flow (Prefect).

Flow to build and update the training dataset by:

1. Downloading ERA5 reanalysis data (climate features)
2. Converting GRIB → Parquet
3. Loading CEN generation data (targets)
4. Merging ERA5 + CEN into a training-ready dataset
5. Appending to the historical training Parquet
"""

from __future__ import annotations

from pathlib import Path

from prefect import flow, task
from prefect.logging import get_run_logger

from src.ingestion.cen_loader import load_cen_generation
from src.ingestion.downloader import download_and_store_era5
from src.preprocessing.converter import append_parquet, grib_to_parquet
from src.preprocessing.features import merge_era5_with_cen
from src.utils.config import get_settings
from src.utils.gcs import download_blob, upload_blob
from src.utils.logger import setup_logging


@task(name="download-era5", retries=2, retry_delay_seconds=120)
def download_era5(year: int, month: int) -> str:
    """Download ERA5 data for a given year/month."""
    logger = get_run_logger()
    logger.info("Downloading ERA5 — %d/%02d", year, month)
    return download_and_store_era5(year=year, month=month)


@task(name="convert-grib-to-parquet")
def convert_to_parquet(grib_remote: str, year: int, month: int) -> Path:
    """Convert ERA5 GRIB to Parquet."""
    logger = get_run_logger()
    settings = get_settings()

    # Download GRIB from storage
    local_grib = Path(f"tmp/era5/{year}/era5_{year}_{month:02d}.grib")
    download_blob(grib_remote, local_grib)

    # Convert to Parquet
    parquet_path = Path(f"tmp/era5_parquet/{year}/era5_{year}_{month:02d}.parquet")
    grib_to_parquet(local_grib, parquet_path)

    # Upload Parquet to storage
    remote = f"{settings.gcs.prefixes.processed_era5}/{year}/era5_{year}_{month:02d}.parquet"
    upload_blob(parquet_path, remote)

    logger.info("Converted + uploaded → %s", remote)
    return parquet_path


@task(name="load-cen-data")
def load_cen(technology: str) -> "pd.DataFrame":
    """Load CEN generation data for a technology."""
    import pandas as pd  # noqa: PLC0415

    logger = get_run_logger()
    try:
        df = load_cen_generation(technology=technology)
        logger.info("CEN data loaded: %d rows for '%s'", len(df), technology)
        return df
    except FileNotFoundError:
        logger.warning(
            "CEN data not found for '%s' — returning empty DataFrame. "
            "Place CEN CSV/Excel files in data/cen/ to enable training.",
            technology,
        )
        return pd.DataFrame()


@task(name="merge-and-append")
def merge_and_append(
    era5_parquet: Path,
    cen_df: "pd.DataFrame",
    technology: str,
) -> Path | None:
    """Merge ERA5 features + CEN targets and append to training dataset."""
    import pandas as pd  # noqa: PLC0415

    logger = get_run_logger()
    settings = get_settings()

    if cen_df.empty:
        logger.warning("Skipping merge for '%s' — no CEN data", technology)
        return None

    merged = merge_era5_with_cen(
        era5_parquet,
        cen_df,
        target_col=settings.cen.generation_col,
    )

    # Append to the cumulative training Parquet
    training_path = Path(f"data/training/{technology}_training.parquet")
    append_parquet(merged, training_path)

    # Upload to storage
    remote = f"{settings.gcs.prefixes.processed_training}/{technology}_training.parquet"
    upload_blob(training_path, remote)

    logger.info("Training dataset for '%s' updated: %s", technology, training_path)
    return training_path


@flow(name="training-data-pipeline", log_prints=True)
def training_data_flow(
    year: int,
    month: int,
    technologies: list[str] | None = None,
) -> dict[str, Path | None]:
    """Build/update the training dataset for given ERA5 period.

    Args:
        year: ERA5 data year.
        month: ERA5 data month.
        technologies: Technologies to process.  Defaults to config.

    Returns:
        Dict mapping technology to training dataset path.
    """
    setup_logging()
    logger = get_run_logger()
    settings = get_settings()

    technologies = technologies or settings.prediction.technologies
    logger.info("=== Training Data Flow — %d/%02d ===", year, month)

    # Step 1: Download ERA5
    grib_remote = download_era5(year, month)

    # Step 2: Convert to Parquet
    era5_parquet = convert_to_parquet(grib_remote, year, month)

    # Step 3-4: For each technology, load CEN and merge
    results: dict[str, Path | None] = {}
    for tech in technologies:
        cen_df = load_cen(tech)
        results[tech] = merge_and_append(era5_parquet, cen_df, tech)

    logger.info("=== Training Data Flow complete ===")
    return results


if __name__ == "__main__":
    training_data_flow(year=2024, month=1)
