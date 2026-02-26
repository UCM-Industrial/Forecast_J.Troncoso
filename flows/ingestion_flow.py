"""RECAST — Forecast ingestion flow (Prefect).

Daily flow to download **AIFS-single** forecast data
for the prediction pipeline.
"""

from __future__ import annotations

from datetime import datetime

from prefect import flow, task
from prefect.logging import get_run_logger

from src.ingestion.downloader import download_and_store_forecast
from src.utils.logger import setup_logging


@task(name="download-aifs-forecast", retries=3, retry_delay_seconds=60)
def download_forecast(date: str) -> str:
    """Download and store an AIFS-single forecast."""
    logger = get_run_logger()
    logger.info("Starting AIFS-single download for date=%s", date)
    uri = download_and_store_forecast(date=date)
    logger.info("Download complete: %s", uri)
    return uri


@flow(name="forecast-ingestion", log_prints=True)
def ingestion_flow(date: str | None = None) -> str:
    """Download daily AIFS-single forecast data for predictions.

    Args:
        date: Target date in ``YYYYMMDD`` format.
              Defaults to today.

    Returns:
        URI of the stored forecast file.
    """
    setup_logging()
    logger = get_run_logger()

    date = date or datetime.now().strftime("%Y%m%d")
    logger.info("=== Forecast Ingestion Flow — date=%s ===", date)

    uri = download_forecast(date)

    logger.info("=== Forecast Ingestion Flow complete ===")
    return uri


if __name__ == "__main__":
    ingestion_flow()
