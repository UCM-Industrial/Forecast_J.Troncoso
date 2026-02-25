"""RECAST test configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_datetime_df() -> pd.DataFrame:
    """DataFrame with a DatetimeIndex for feature engineering tests."""
    dates = pd.date_range("2025-01-01", periods=48, freq="h", tz="America/Santiago")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "value_a": rng.random(48) * 100,
            "value_b": rng.random(48) * 50,
        },
        index=dates,
    )


@pytest.fixture
def sample_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic feature matrix and target for model tests."""
    rng = np.random.default_rng(42)
    n = 200
    x = pd.DataFrame(
        {
            "hour_sin": np.sin(np.linspace(0, 4 * np.pi, n)),
            "hour_cos": np.cos(np.linspace(0, 4 * np.pi, n)),
            "month_sin": rng.random(n),
            "month_cos": rng.random(n),
            "wind_speed": rng.random(n) * 15,
            "temperature": rng.random(n) * 30,
        },
    )
    y = pd.Series(
        x["wind_speed"] * 10 + rng.normal(0, 5, n),
        name="generation_mwh",
    )
    return x, y


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Create a minimal settings.yaml in a temp directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    yaml_content = """\
ecmwf:
  model: "aifs-single"
  variables: ["100v", "100u", "ssrd"]
  steps: [0, 6, 12]
  time: 0

gcs:
  bucket: "test-bucket"

training:
  test_size: 0.2
  cv_folds: 3
  random_seed: 42
  xgboost:
    n_estimators: 10
    max_depth: 3
    learning_rate: 0.1

prediction:
  technologies:
    - "wind"
    - "solar"
"""

    (config_dir / "settings.yaml").write_text(yaml_content, encoding="utf-8")
    return config_dir / "settings.yaml"
