"""RECAST — Configuration management.

Loads settings from config/settings.yaml and allows
environment variable overrides via Pydantic Settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ── YAML Sub-models ──────────────────────────────────────────


class ECMWFConfig(BaseModel):
    """ECMWF Open Data source configuration."""

    model: str = "aifs-single"
    type: str = "fc"
    variables: list[str] = Field(default_factory=lambda: ["100v", "100u", "ssrd"])
    steps: list[int] = Field(
        default_factory=lambda: list(range(0, 360, 6)),
    )
    time: int = 0


class GCSPrefixes(BaseModel):
    """GCS path prefixes for data organisation."""

    raw: str = "raw/ecmwf"
    processed: str = "processed"
    models: str = "models"
    predictions: str = "predictions"


class GCSConfig(BaseModel):
    """Google Cloud Storage configuration."""

    bucket: str = "recast-energy-forecast"
    prefixes: GCSPrefixes = Field(default_factory=GCSPrefixes)


class GeoConfig(BaseModel):
    """Geospatial processing configuration."""

    regions_shapefile: str = "data/masks/Regional.shp"
    segment_by: str = "Region"
    output_timezone: str = "America/Santiago"
    chunk_size: dict[str, int] = Field(
        default_factory=lambda: {"latitude": 50, "longitude": 50},
    )


class XGBoostParams(BaseModel):
    """XGBoost hyperparameters."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    objective: str = "reg:squarederror"


class TrainingConfig(BaseModel):
    """Training pipeline configuration."""

    test_size: float = 0.2
    cv_folds: int = 5
    random_seed: int = 42
    xgboost: XGBoostParams = Field(default_factory=XGBoostParams)


class PredictionConfig(BaseModel):
    """Prediction output configuration."""

    output_format: str = "csv"
    technologies: list[str] = Field(
        default_factory=lambda: ["solar", "wind"],
    )


# ── Main Settings ────────────────────────────────────────────


class Settings(BaseSettings):
    """Application settings with env-var overrides.

    Priority: environment variables > .env file > settings.yaml > defaults.
    """

    # Fields populated from YAML (with defaults if YAML is missing)
    ecmwf: ECMWFConfig = Field(default_factory=ECMWFConfig)
    gcs: GCSConfig = Field(default_factory=GCSConfig)
    geospatial: GeoConfig = Field(default_factory=GeoConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)

    # Fields from environment variables
    ecmwf_api_key: str = Field(default="", alias="ECMWF_API_KEY")
    gcs_bucket_name: str = Field(default="", alias="GCS_BUCKET_NAME")
    storage_backend: str = Field(default="local", alias="STORAGE_BACKEND")
    local_data_dir: str = Field(default="./data", alias="LOCAL_DATA_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (where config/ lives)."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "config").is_dir():
            return current
        current = current.parent
    return Path.cwd()


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from YAML file and merge with env vars.

    Args:
        config_path: Path to settings.yaml.  If ``None``, auto-detected
            from the project root.

    Returns:
        Fully validated ``Settings`` instance.
    """
    if config_path is None:
        config_path = _find_project_root() / "config" / "settings.yaml"
    else:
        config_path = Path(config_path)

    yaml_data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            yaml_data = yaml.safe_load(fh) or {}

    settings = Settings(**yaml_data)

    # Override GCS bucket from env if provided
    if settings.gcs_bucket_name:
        settings.gcs.bucket = settings.gcs_bucket_name

    return settings


# Module-level singleton (lazy)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached ``Settings`` singleton."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = load_settings()
    return _settings
