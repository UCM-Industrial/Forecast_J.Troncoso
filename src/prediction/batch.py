"""RECAST — Batch prediction.

Loads the latest trained model and generates predictions for
new data, saving results to storage.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.training.model import XGBoostForecaster
from src.utils.config import get_settings
from src.utils.gcs import download_blob, list_blobs, upload_blob
from src.utils.logger import get_logger

logger = get_logger("prediction.batch")


def get_latest_model_path(
    models_prefix: str | None = None,
) -> str:
    """Find the most recent model in storage.

    Args:
        models_prefix: GCS prefix to search. Defaults to config.

    Returns:
        Remote path of the latest ``.joblib`` model file.

    Raises:
        FileNotFoundError: If no models are found.
    """
    settings = get_settings()
    prefix = models_prefix or settings.gcs.prefixes.models

    blobs = list_blobs(prefix)
    model_files = sorted([b for b in blobs if b.endswith(".joblib")])

    if not model_files:
        msg = f"No trained models found under '{prefix}'"
        raise FileNotFoundError(msg)

    latest = model_files[-1]  # lexicographic sort → latest date
    logger.info("Latest model: %s", latest)
    return latest


def run_batch_prediction(
    *,
    features_path: str | Path | None = None,
    date: str | None = None,
    technology: str = "wind",
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Generate predictions for a set of features and save them.

    Args:
        features_path: Path to the feature CSV.  If ``None``, downloads
            from storage based on ``date``.
        date: Date string (``YYYYMMDD``).  Defaults to today.
        technology: ``"wind"`` or ``"solar"``.
        model_path: Path to a specific model.  If ``None``, uses the
            latest model from storage.
        output_dir: Local directory for output.  If ``None``, uses a
            temp directory and uploads to storage.

    Returns:
        Path to the predictions file.
    """
    settings = get_settings()
    date = date or datetime.now().strftime("%Y%m%d")

    # ── Load model ───────────────────────────────────────────
    if model_path is not None:
        model = XGBoostForecaster.load(model_path)
    else:
        remote_model = get_latest_model_path()
        local_model = Path(f"tmp/models/{Path(remote_model).name}")
        download_blob(remote_model, local_model)
        # Also download metadata
        remote_meta = remote_model.replace(".joblib", ".meta.json")
        local_meta = local_model.with_suffix(".meta.json")
        try:
            download_blob(remote_meta, local_meta)
        except FileNotFoundError:
            logger.warning("No metadata file for model")
        model = XGBoostForecaster.load(local_model)

    # ── Load features ────────────────────────────────────────
    if features_path is not None:
        features_path = Path(features_path)
    else:
        remote_features = (
            f"{settings.gcs.prefixes.processed}/{date}/{technology}_features.csv"
        )
        features_path = Path(f"tmp/features/{technology}_{date}.csv")
        download_blob(remote_features, features_path)

    from src.preprocessing.features import read_csv_with_datetime  # noqa: PLC0415

    df = read_csv_with_datetime(features_path)
    logger.info("Loaded %d rows of features for %s prediction", len(df), technology)

    # ── Predict ──────────────────────────────────────────────
    # Use only the features the model was trained on
    if model.feature_names:
        available = [f for f in model.feature_names if f in df.columns]
        missing = set(model.feature_names) - set(available)
        if missing:
            logger.warning("Missing features (using available): %s", missing)
        x = df[available]
    else:
        x = df.select_dtypes("number")

    predictions = model.predict(x)

    # ── Save results ─────────────────────────────────────────
    results = pd.DataFrame(
        {"prediction": predictions},
        index=df.index,
    )
    results.index.name = "datetime"

    if output_dir is not None:
        out = Path(output_dir)
    else:
        out = Path(f"tmp/predictions/{technology}")
    out.mkdir(parents=True, exist_ok=True)

    output_file = out / f"{technology}_{date}.csv"
    results.to_csv(output_file)
    logger.info("Predictions saved → %s (%d rows)", output_file, len(results))

    # Upload to storage
    remote_path = f"{settings.gcs.prefixes.predictions}/{technology}/{date}.csv"
    upload_blob(output_file, remote_path)

    return output_file
