"""RECAST — Prediction flow (Prefect).

Daily flow to generate batch predictions from the latest
model and processed features.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from prefect import flow, task
from prefect.logging import get_run_logger

from src.prediction.batch import run_batch_prediction
from src.utils.config import get_settings
from src.utils.logger import setup_logging


@task(name="predict-technology")
def predict_technology(technology: str, date: str) -> Path:
    """Generate predictions for one technology."""
    logger = get_run_logger()

    output_path = run_batch_prediction(
        date=date,
        technology=technology,
    )

    logger.info("Predictions for '%s' → %s", technology, output_path)
    return output_path


@flow(name="daily-prediction", log_prints=True)
def prediction_flow(date: str | None = None) -> dict[str, Path]:
    """Generate batch predictions for all configured technologies.

    Args:
        date: Date string (``YYYYMMDD``). Defaults to today.

    Returns:
        Dict mapping technology to output file paths.
    """
    setup_logging()
    logger = get_run_logger()

    date = date or datetime.now().strftime("%Y%m%d")
    settings = get_settings()

    logger.info("=== Prediction Flow — date=%s ===", date)

    results: dict[str, Path] = {}
    for tech in settings.prediction.technologies:
        results[tech] = predict_technology(tech, date)

    logger.info("=== Prediction Flow complete ===")
    return results


if __name__ == "__main__":
    prediction_flow()
