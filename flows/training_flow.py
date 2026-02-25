"""RECAST — Training flow (Prefect).

Weekly flow to retrain the XGBoost forecaster with
accumulated historical data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from prefect import flow, task
from prefect.logging import get_run_logger

from src.training.trainer import train_pipeline
from src.utils.config import get_settings
from src.utils.gcs import upload_blob
from src.utils.logger import setup_logging


@task(name="train-model")
def train_model(
    features_path: str | Path,
    target_col: str,
    technology: str,
    date: str,
) -> dict:
    """Run the training pipeline for one technology."""
    logger = get_run_logger()

    model_output = Path(f"tmp/models/{technology}/{date}/model")

    results = train_pipeline(
        features_path=features_path,
        target_col=target_col,
        model_output_path=model_output,
    )

    # Upload model and metadata to storage
    settings = get_settings()
    model_file = Path(results["model_path"])
    meta_file = model_file.with_suffix(".meta.json")

    remote_model = f"{settings.gcs.prefixes.models}/{technology}/{date}/model.joblib"
    remote_meta = f"{settings.gcs.prefixes.models}/{technology}/{date}/model.meta.json"

    upload_blob(model_file, remote_model)
    if meta_file.exists():
        upload_blob(meta_file, remote_meta)

    logger.info(
        "Model for '%s' trained and uploaded — R²=%.4f, MAE=%.4f",
        technology,
        results["test_metrics"]["r2"],
        results["test_metrics"]["mae"],
    )
    return results


@flow(name="weekly-training", log_prints=True)
def training_flow(
    wind_features_path: str | Path | None = None,
    solar_features_path: str | Path | None = None,
    wind_target: str = "generation_mwh",
    solar_target: str = "generation_mwh",
) -> dict[str, dict]:
    """Retrain XGBoost models for wind and solar.

    Args:
        wind_features_path: Path to wind features CSV.
        solar_features_path: Path to solar features CSV.
        wind_target: Target column name for wind.
        solar_target: Target column name for solar.

    Returns:
        Dict with training results for each technology.
    """
    setup_logging()
    logger = get_run_logger()

    date = datetime.now().strftime("%Y%m%d")
    logger.info("=== Training Flow — date=%s ===", date)

    results: dict[str, dict] = {}

    if wind_features_path:
        results["wind"] = train_model(
            features_path=wind_features_path,
            target_col=wind_target,
            technology="wind",
            date=date,
        )

    if solar_features_path:
        results["solar"] = train_model(
            features_path=solar_features_path,
            target_col=solar_target,
            technology="solar",
            date=date,
        )

    logger.info("=== Training Flow complete ===")
    return results


if __name__ == "__main__":
    training_flow()
