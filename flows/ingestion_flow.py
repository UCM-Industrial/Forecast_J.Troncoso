"""RECAST — Ingestion flow (Prefect).

Daily flow to download ECMWF Open Data forecasts and store
them in GCS / local storage.
"""

from __future__ import annotations

from datetime import datetime

from prefect import flow, task
from prefect.logging import get_run_logger

from src.ingestion.downloader import download_and_store_forecast
from src.utils.logger import setup_logging


@task(name="download-ecmwf-forecast", retries=3, retry_delay_seconds=60)
def download_forecast(date: str) -> str:
    """Download and store a single forecast."""
    logger = get_run_logger()
    logger.info("Starting ECMWF download for date=%s", date)
    uri = download_and_store_forecast(date=date)
    logger.info("Download complete: %s", uri)
    return uri


@flow(name="daily-ingestion", log_prints=True)
def ingestion_flow(date: str | None = None) -> str:
    """Orchestrate the daily ECMWF data ingestion.

    Args:
        date: Target date in ``YYYYMMDD`` format.
              Defaults to today.

    Returns:
        URI of the stored forecast file.
    """
    setup_logging()
    logger = get_run_logger()

    date = date or datetime.now().strftime("%Y%m%d")
    logger.info("=== Ingestion Flow — date=%s ===", date)

    uri = download_forecast(date)

    logger.info("=== Ingestion Flow complete ===")
    return uri


if __name__ == "__main__":
    ingestion_flow()
