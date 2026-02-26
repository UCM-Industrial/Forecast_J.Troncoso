"""RECAST — XGBoost forecaster.

Simplified model wrapper focused exclusively on XGBoost
for renewable energy generation forecasting.

Replaces the generic Strategy-pattern machinery from the legacy
``modeling.py`` with a single, well-documented class.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("training.model")


class XGBoostForecaster:
    """XGBoost regressor tailored for energy-generation forecasting.

    Key improvements over the legacy implementation:

    * Single responsibility — only XGBoost, no factory/strategy.
    * ``joblib`` serialisation instead of raw ``pickle``.
    * Model metadata (features, metrics, timestamp) saved alongside.
    * Reproducibility via fixed random seed from config.

    Args:
        params: XGBoost hyperparameters.  Merged with config defaults.
        random_seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        random_seed: int | None = None,
    ) -> None:
        settings = get_settings()
        seed = random_seed or settings.training.random_seed

        # Merge config defaults with caller overrides
        default_params = settings.training.xgboost.model_dump()
        if params:
            default_params.update(params)
        default_params["random_state"] = seed

        self.params = default_params
        self.model = XGBRegressor(**self.params)
        self.is_fitted: bool = False
        self.feature_names: list[str] | None = None
        self._metadata: dict[str, Any] = {}

    def fit(self, x: pd.DataFrame, y: pd.Series) -> XGBoostForecaster:
        """Train the model.

        Args:
            x: Feature DataFrame.
            y: Target Series.

        Returns:
            Self, for method chaining.
        """
        self.feature_names = x.columns.tolist()
        self.model.fit(x, y)
        self.is_fitted = True

        self._metadata = {
            "trained_at": datetime.now(tz=UTC).isoformat(),
            "features": self.feature_names,
            "n_samples": len(x),
            "params": self.params,
        }

        logger.info(
            "Model trained on %d samples x %d features",
            len(x),
            len(self.feature_names),
        )
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            x: Feature DataFrame (must match training features).

        Returns:
            Array of predicted values.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            msg = "Model must be fitted before calling predict()"
            raise RuntimeError(msg)
        return self.model.predict(x)

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores.

        Returns:
            Dict mapping feature names to importance values.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted or not self.feature_names:
            msg = "Model must be fitted before getting feature importance"
            raise RuntimeError(msg)

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist(), strict=True))

    # ── Serialisation ────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Save model and metadata to disk.

        Creates two files:
        - ``<path>.joblib`` — serialised model
        - ``<path>.meta.json`` — metadata (features, params, metrics)

        Args:
            path: Base path (without extension).

        Returns:
            Path to the saved model file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_path = path.with_suffix(".joblib")
        meta_path = path.with_suffix(".meta.json")

        joblib.dump(self.model, model_path)

        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, indent=2, default=str)

        logger.info("Model saved → %s", model_path)
        return model_path

    @classmethod
    def load(cls, path: str | Path) -> XGBoostForecaster:
        """Load a previously saved model.

        Args:
            path: Base path (without extension) or path to ``.joblib`` file.

        Returns:
            A fitted ``XGBoostForecaster`` instance.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        path = Path(path)

        # Accept either base path or full .joblib path
        model_path = path.with_suffix(".joblib") if path.suffix != ".joblib" else path
        meta_path = model_path.with_suffix(".meta.json")

        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        instance = cls.__new__(cls)
        instance.model = joblib.load(model_path)
        instance.is_fitted = True
        instance.params = {}

        # Load metadata if available
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as fh:
                instance._metadata = json.load(fh)
            instance.feature_names = instance._metadata.get("features")
            instance.params = instance._metadata.get("params", {})
        else:
            instance._metadata = {}
            instance.feature_names = None
            logger.warning("No metadata file found at %s", meta_path)

        logger.info("Model loaded ← %s", model_path)
        return instance

    @property
    def metadata(self) -> dict[str, Any]:
        """Access model metadata (read-only)."""
        return dict(self._metadata)
