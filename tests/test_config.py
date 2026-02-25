"""Tests for ``src.utils.config``."""

from __future__ import annotations

from pathlib import Path

from src.utils.config import Settings, load_settings


class TestLoadSettings:
    """Tests for configuration loading."""

    def test_load_from_yaml(self, tmp_config: Path) -> None:
        """Should correctly parse settings.yaml."""
        settings = load_settings(tmp_config)

        assert settings.ecmwf.model == "aifs-single"
        assert "100v" in settings.ecmwf.variables
        assert settings.training.random_seed == 42
        assert settings.training.xgboost.n_estimators == 10

    def test_defaults_without_yaml(self, tmp_path: Path) -> None:
        """Should use defaults when YAML is missing."""
        fake_path = tmp_path / "nonexistent.yaml"
        settings = load_settings(fake_path)

        assert settings.ecmwf.model == "aifs-single"
        assert settings.training.xgboost.n_estimators == 300  # default

    def test_gcs_bucket_override(self, tmp_config: Path, monkeypatch) -> None:
        """GCS_BUCKET_NAME env var should override YAML value."""
        monkeypatch.setenv("GCS_BUCKET_NAME", "my-override-bucket")
        settings = load_settings(tmp_config)

        assert settings.gcs.bucket == "my-override-bucket"

    def test_storage_backend_default(self, tmp_config: Path) -> None:
        """Default storage backend should be 'local'."""
        settings = load_settings(tmp_config)
        assert settings.storage_backend == "local"
