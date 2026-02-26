"""RECAST — Data downloader and storage.

Orchestrates the download of weather data and its
storage into GCS (or local filesystem).

Two entry-points:

* ``download_and_store_forecast()`` — AIFS-single (prediction)
* ``download_and_store_era5()`` — ERA5 historical (training)
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

from src.utils.config import get_settings
from src.utils.gcs import upload_blob
from src.utils.logger import get_logger

logger = get_logger("ingestion.downloader")


# ── AIFS-single (prediction) ────────────────────────────────


def download_and_store_forecast(
    *,
    date: str | None = None,
    time: int | None = None,
    output_dir: str | Path | None = None,
) -> str:
    """Download an AIFS-single forecast and upload it to storage.

    This is the entry-point for the **prediction** ingestion step.

    Args:
        date: Initialization date (``"YYYYMMDD"``).  Defaults to today.
        time: Initialization hour.  Defaults to config.
        output_dir: Local directory to keep the raw file.

    Returns:
        The remote path (GCS URI or local path) of the stored file.
    """
    from src.ingestion.ecmwf_client import ECMWFClient  # noqa: PLC0415

    settings = get_settings()
    date = date or datetime.now().strftime("%Y%m%d")

    client = ECMWFClient()

    if output_dir is not None:
        local_dir = Path(output_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        local_dir = Path(tempfile.mkdtemp(prefix="recast_aifs_"))
        cleanup = True

    filename = f"aifs_single_{date}.grib2"
    local_path = local_dir / filename

    try:
        client.download_forecast(target=local_path, date=date, time=time)

        remote_path = f"{settings.gcs.prefixes.raw_aifs}/{date}/{filename}"
        uri = upload_blob(local_path, remote_path)

        logger.info("Forecast stored at %s", uri)
        return uri

    finally:
        if cleanup and local_dir.exists():
            import shutil  # noqa: PLC0415

            shutil.rmtree(local_dir, ignore_errors=True)
            logger.debug("Cleaned up temp dir %s", local_dir)


# ── ERA5 (training) ─────────────────────────────────────────


def download_and_store_era5(
    *,
    year: int,
    month: int,
    output_dir: str | Path | None = None,
) -> str:
    """Download ERA5 reanalysis data and upload it to storage.

    This is the entry-point for the **training** ingestion step.

    Args:
        year: Year to download.
        month: Month to download.
        output_dir: Local directory to keep the raw file.

    Returns:
        The remote path (GCS URI or local path) of the stored file.
    """
    from src.ingestion.era5_client import ERA5Client  # noqa: PLC0415

    settings = get_settings()
    client = ERA5Client()

    if output_dir is not None:
        local_dir = Path(output_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        local_dir = Path(tempfile.mkdtemp(prefix="recast_era5_"))
        cleanup = True

    month_str = f"{month:02d}"
    filename = f"era5_{year}_{month_str}.grib"
    local_path = local_dir / filename

    try:
        client.download(target=local_path, year=year, month=month)

        remote_path = f"{settings.gcs.prefixes.raw_era5}/{year}/{filename}"
        uri = upload_blob(local_path, remote_path)

        logger.info("ERA5 stored at %s", uri)
        return uri

    finally:
        if cleanup and local_dir.exists():
            import shutil  # noqa: PLC0415

            shutil.rmtree(local_dir, ignore_errors=True)
