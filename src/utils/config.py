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


class ERA5Config(BaseModel):
    """ERA5 reanalysis data configuration (training features)."""

    dataset: str = "reanalysis-era5-single-levels"
    product_type: str = "reanalysis"
    variables: list[str] = Field(
        default_factory=lambda: [
            "100m_v_component_of_wind",
            "100m_u_component_of_wind",
            "surface_solar_radiation_downwards",
        ],
    )
    area: list[float] = Field(default_factory=lambda: [-17, -76, -56, -66])
    format: str = "grib"


class CENConfig(BaseModel):
    """CEN energy generation data configuration (training targets)."""

    source_dir: str = "data/cen"
    generation_col: str = "generation_mwh"
    datetime_col: str = "datetime"


class AIFSConfig(BaseModel):
    """AIFS-single forecast configuration (prediction features)."""

    model: str = "aifs-single"
    type: str = "fc"
    variables: list[str] = Field(default_factory=lambda: ["100v", "100u", "ssrd"])
    steps: list[int] = Field(
        default_factory=lambda: list(range(0, 360, 6)),
    )
    time: int = 0


class GCSPrefixes(BaseModel):
    """GCS path prefixes for data organisation."""

    raw_era5: str = "raw/era5"
    raw_aifs: str = "raw/aifs"
    raw_cen: str = "raw/cen"
    processed_era5: str = "processed/era5_parquet"
    processed_training: str = "processed/training"
    processed_prediction: str = "processed/prediction"
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

    # Data source configs (from YAML)
    era5: ERA5Config = Field(default_factory=ERA5Config)
    cen: CENConfig = Field(default_factory=CENConfig)
    aifs: AIFSConfig = Field(default_factory=AIFSConfig)

    # Infrastructure configs (from YAML)
    gcs: GCSConfig = Field(default_factory=GCSConfig)
    geospatial: GeoConfig = Field(default_factory=GeoConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)

    # Fields from environment variables
    ecmwf_api_key: str = Field(default="", alias="ECMWF_API_KEY")
    cds_api_key: str = Field(default="", alias="CDS_API_KEY")
    cds_api_url: str = Field(
        default="https://cds.climate.copernicus.eu/api",
        alias="CDS_API_URL",
    )
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
