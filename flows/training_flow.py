"""RECAST — Training flow (Prefect).

Weekly flow to retrain XGBoost models using the
accumulated ERA5+CEN training dataset.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from prefect import flow, task
from prefect.logging import get_run_logger

from src.training.trainer import train_pipeline
from src.utils.config import get_settings
from src.utils.gcs import download_blob, upload_blob
from src.utils.logger import setup_logging


@task(name="train-model")
def train_model(technology: str, date: str) -> dict:
    """Train an XGBoost model for one technology.

    Uses the accumulated training dataset (Parquet).
    """
    logger = get_run_logger()
    settings = get_settings()

    # Download training dataset from storage
    remote_training = f"{settings.gcs.prefixes.processed_training}/{technology}_training.parquet"
    local_training = Path(f"tmp/training/{technology}_training.parquet")

    try:
        download_blob(remote_training, local_training)
    except FileNotFoundError:
        logger.warning(
            "No training dataset found for '%s' at %s. Run training_data_flow first.",
            technology,
            remote_training,
        )
        return {"status": "skipped", "reason": "no training data"}

    model_output = Path(f"tmp/models/{technology}/{date}/model")

    results = train_pipeline(
        features_path=local_training,
        target_col=settings.cen.generation_col,
        model_output_path=model_output,
    )

    # Upload model and metadata to storage
    model_file = Path(results["model_path"])
    meta_file = model_file.with_suffix(".meta.json")

    remote_model = f"{settings.gcs.prefixes.models}/{technology}/{date}/model.joblib"
    remote_meta = f"{settings.gcs.prefixes.models}/{technology}/{date}/model.meta.json"

    upload_blob(model_file, remote_model)
    if meta_file.exists():
        upload_blob(meta_file, remote_meta)

    logger.info(
        "Model for '%s' trained — R2=%.4f, MAE=%.4f",
        technology,
        results["test_metrics"]["r2"],
        results["test_metrics"]["mae"],
    )
    return results


@flow(name="weekly-training", log_prints=True)
def training_flow(
    technologies: list[str] | None = None,
) -> dict[str, dict]:
    """Retrain XGBoost models using ERA5+CEN training data.

    Args:
        technologies: Technologies to train.  Defaults to config.

    Returns:
        Dict with training results per technology.
    """
    setup_logging()
    logger = get_run_logger()
    settings = get_settings()

    date = datetime.now().strftime("%Y%m%d")
    technologies = technologies or settings.prediction.technologies
    logger.info("=== Training Flow — date=%s ===", date)

    results: dict[str, dict] = {}
    for tech in technologies:
        results[tech] = train_model(tech, date)

    logger.info("=== Training Flow complete ===")
    return results


if __name__ == "__main__":
    training_flow()
