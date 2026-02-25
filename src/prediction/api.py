"""RECAST — FastAPI prediction service.

Lightweight REST API for serving predictions stored in GCS.
Designed to be deployed as a container on Cloud Run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.utils.config import get_settings
from src.utils.gcs import blob_exists, list_blobs
from src.utils.logger import get_logger, setup_logging

# Initialise logging
setup_logging()
logger = get_logger("prediction.api")

app = FastAPI(
    title="RECAST Prediction API",
    description=(
        "REST API for renewable energy generation forecasts. "
        "Serves batch predictions stored in Google Cloud Storage."
    ),
    version="2.0.0",
)


# ── Response models ──────────────────────────────────────────


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    storage_backend: str


class PredictionResponse(BaseModel):
    """Prediction result response model."""

    technology: str
    date: str
    predictions: list[dict[str, Any]]
    count: int


class AvailableDatesResponse(BaseModel):
    """Available prediction dates response model."""

    technology: str
    dates: list[str]
    count: int


# ── Endpoints ────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        storage_backend=settings.storage_backend,
    )


@app.get("/predictions/latest", response_model=PredictionResponse)
async def get_latest_predictions(
    technology: str = Query(
        default="wind",
        description="Energy technology: 'wind' or 'solar'",
    ),
) -> PredictionResponse:
    """Return the most recent predictions for a technology."""
    _validate_technology(technology)

    settings = get_settings()
    prefix = f"{settings.gcs.prefixes.predictions}/{technology}/"
    blobs = sorted(list_blobs(prefix))

    if not blobs:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for technology='{technology}'",
        )

    latest_blob = blobs[-1]
    return _load_prediction(latest_blob, technology)


@app.get("/predictions/{date}", response_model=PredictionResponse)
async def get_predictions_by_date(
    date: str,
    technology: str = Query(
        default="wind",
        description="Energy technology: 'wind' or 'solar'",
    ),
) -> PredictionResponse:
    """Return predictions for a specific date."""
    _validate_technology(technology)
    _validate_date(date)

    settings = get_settings()
    remote_path = f"{settings.gcs.prefixes.predictions}/{technology}/{date}.csv"

    if not blob_exists(remote_path):
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for {technology}/{date}",
        )

    return _load_prediction(remote_path, technology)


@app.get("/predictions/dates/available", response_model=AvailableDatesResponse)
async def get_available_dates(
    technology: str = Query(
        default="wind",
        description="Energy technology: 'wind' or 'solar'",
    ),
) -> AvailableDatesResponse:
    """List available prediction dates."""
    _validate_technology(technology)

    settings = get_settings()
    prefix = f"{settings.gcs.prefixes.predictions}/{technology}/"
    blobs = list_blobs(prefix)

    dates = sorted(
        [b.split("/")[-1].replace(".csv", "") for b in blobs if b.endswith(".csv")],
    )

    return AvailableDatesResponse(
        technology=technology,
        dates=dates,
        count=len(dates),
    )


# ── Helpers ──────────────────────────────────────────────────


def _validate_technology(technology: str) -> None:
    allowed = get_settings().prediction.technologies
    if technology not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid technology '{technology}'. Allowed: {allowed}",
        )


def _validate_date(date: str) -> None:
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format '{date}'. Expected YYYYMMDD.",
        ) from None


def _load_prediction(
    remote_path: str,
    technology: str,
) -> PredictionResponse:
    """Download and parse a prediction CSV from storage."""
    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from src.utils.gcs import download_blob  # noqa: PLC0415

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        download_blob(remote_path, tmp_path)
        df = pd.read_csv(tmp_path, parse_dates=["datetime"])
        records = df.to_dict(orient="records")
        date_str = remote_path.split("/")[-1].replace(".csv", "")

        return PredictionResponse(
            technology=technology,
            date=date_str,
            predictions=records,
            count=len(records),
        )
    finally:
        tmp_path.unlink(missing_ok=True)
