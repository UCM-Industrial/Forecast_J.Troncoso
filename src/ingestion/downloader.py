"""RECAST — Forecast data downloader.

Orchestrates the download of ECMWF forecast data and its
storage into GCS (or local filesystem).
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

from src.ingestion.ecmwf_client import ECMWFClient
from src.utils.config import get_settings
from src.utils.gcs import upload_blob
from src.utils.logger import get_logger

logger = get_logger("ingestion.downloader")


def download_and_store_forecast(
    *,
    date: str | None = None,
    time: int | None = None,
    output_dir: str | Path | None = None,
) -> str:
    """Download an ECMWF forecast and upload it to storage.

    This is the main entry-point for the ingestion step.  It:

    1. Downloads the GRIB file to a temporary directory.
    2. Uploads it to GCS (or local storage) under a date-versioned path.

    Args:
        date: Initialization date (``"YYYYMMDD"``).  Defaults to today.
        time: Initialization hour.  Defaults to config.
        output_dir: Local directory to keep the raw file.  If omitted, a
            temporary directory is used and cleaned up automatically.

    Returns:
        The remote path (GCS URI or local path) of the stored file.
    """
    settings = get_settings()
    date = date or datetime.now().strftime("%Y%m%d")

    client = ECMWFClient()

    # Determine where to save locally
    if output_dir is not None:
        local_dir = Path(output_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        local_dir = Path(tempfile.mkdtemp(prefix="recast_ingestion_"))
        cleanup = True

    filename = f"aifs_single_{date}.grib2"
    local_path = local_dir / filename

    try:
        # Step 1: Download
        client.download_forecast(
            target=local_path,
            date=date,
            time=time,
        )

        # Step 2: Upload to storage
        remote_path = f"{settings.gcs.prefixes.raw}/{date}/{filename}"
        uri = upload_blob(local_path, remote_path)

        logger.info("Forecast stored at %s", uri)
        return uri

    finally:
        if cleanup and local_dir.exists():
            import shutil  # noqa: PLC0415

            shutil.rmtree(local_dir, ignore_errors=True)
            logger.debug("Cleaned up temp dir %s", local_dir)
