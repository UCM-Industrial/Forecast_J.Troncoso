"""Tests for ``src.training.model`` (XGBoostForecaster)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.training.model import XGBoostForecaster


class TestXGBoostForecaster:
    """Tests for the XGBoost model wrapper."""

    def test_fit_and_predict(
        self,
        sample_training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Model should fit and produce predictions of correct shape."""
        x, y = sample_training_data
        model = XGBoostForecaster(
            params={"n_estimators": 10, "max_depth": 3},
        )

        model.fit(x, y)

        assert model.is_fitted
        preds = model.predict(x)
        assert preds.shape == (len(x),)

    def test_predict_before_fit_raises(self) -> None:
        """Calling predict before fit should raise RuntimeError."""
        model = XGBoostForecaster(
            params={"n_estimators": 10, "max_depth": 3},
        )
        df = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(df)

    def test_feature_importance(
        self,
        sample_training_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """Feature importance should return valid scores."""
        x, y = sample_training_data
        model = XGBoostForecaster(
            params={"n_estimators": 10, "max_depth": 3},
        )
        model.fit(x, y)

        importance = model.get_feature_importance()
        assert set(importance.keys()) == set(x.columns)
        assert all(isinstance(v, float) for v in importance.values())

    def test_save_and_load(
        self,
        sample_training_data: tuple[pd.DataFrame, pd.Series],
        tmp_path: Path,
    ) -> None:
        """Save/load should preserve predictions."""
        x, y = sample_training_data
        model = XGBoostForecaster(
            params={"n_estimators": 10, "max_depth": 3},
        )
        model.fit(x, y)

        preds_before = model.predict(x)

        model_path = tmp_path / "test_model"
        model.save(model_path)

        # Verify files created
        assert (tmp_path / "test_model.joblib").exists()
        assert (tmp_path / "test_model.meta.json").exists()

        # Load and compare
        loaded = XGBoostForecaster.load(model_path)
        preds_after = loaded.predict(x)

        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_load_preserves_metadata(
        self,
        sample_training_data: tuple[pd.DataFrame, pd.Series],
        tmp_path: Path,
    ) -> None:
        """Loaded model should have correct metadata."""
        x, y = sample_training_data
        model = XGBoostForecaster(
            params={"n_estimators": 10, "max_depth": 3},
        )
        model.fit(x, y)
        model.save(tmp_path / "test_model")

        loaded = XGBoostForecaster.load(tmp_path / "test_model")

        assert loaded.feature_names == x.columns.tolist()
        assert loaded.is_fitted
        assert "trained_at" in loaded.metadata

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        """Loading from a non-existent path should raise."""
        with pytest.raises(FileNotFoundError):
            XGBoostForecaster.load(tmp_path / "nonexistent")
