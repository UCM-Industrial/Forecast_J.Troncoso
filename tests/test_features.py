"""Tests for ``src.preprocessing.features``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.features import create_cyclical_features


class TestCreateCyclicalFeatures:
    """Tests for the cyclical encoding function."""

    def test_default_features(self, sample_datetime_df: pd.DataFrame) -> None:
        """Default call adds sin/cos for hour, day, month + year."""
        result = create_cyclical_features(sample_datetime_df)

        expected_cols = {
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
            "year",
        }
        assert expected_cols.issubset(result.columns)

    def test_sin_cos_range(self, sample_datetime_df: pd.DataFrame) -> None:
        """Sin and cos values must be in [-1, 1]."""
        result = create_cyclical_features(sample_datetime_df)

        for col in result.columns:
            if col.endswith("_sin") or col.endswith("_cos"):
                assert result[col].min() >= -1.0, f"{col} has value < -1"
                assert result[col].max() <= 1.0, f"{col} has value > 1"

    def test_original_columns_preserved(self, sample_datetime_df: pd.DataFrame) -> None:
        """Original columns should not be removed."""
        result = create_cyclical_features(sample_datetime_df)
        assert "value_a" in result.columns
        assert "value_b" in result.columns

    def test_custom_features(self, sample_datetime_df: pd.DataFrame) -> None:
        """Specifying features= should only add those."""
        result = create_cyclical_features(
            sample_datetime_df,
            features=["hour"],
            include_year=False,
        )

        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "month_sin" not in result.columns
        assert "year" not in result.columns

    def test_no_datetime_raises(self) -> None:
        """Should raise if no DatetimeIndex and no datetime_col."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            create_cyclical_features(df)

    def test_does_not_mutate_input(self, sample_datetime_df: pd.DataFrame) -> None:
        """Input DataFrame should not be modified."""
        original_cols = list(sample_datetime_df.columns)
        create_cyclical_features(sample_datetime_df)
        assert list(sample_datetime_df.columns) == original_cols
