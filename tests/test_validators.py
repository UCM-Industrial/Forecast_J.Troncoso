"""Tests for ``src.utils.validators``."""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils.validators import (
    check_no_nulls,
    find_missing_dates,
    validate_dataframe_schema,
    validate_feature_ranges,
)


class TestValidateDataframeSchema:
    """Tests for schema validation."""

    def test_valid_schema(self) -> None:
        """Should pass when all required columns are present."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        validate_dataframe_schema(df, ["a", "b"])  # Should not raise

    def test_missing_columns(self) -> None:
        """Should raise ValueError for missing columns."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="missing columns"):
            validate_dataframe_schema(df, ["a", "b", "c"])


class TestFindMissingDates:
    """Tests for missing date detection."""

    def test_no_missing(self) -> None:
        """Complete date range should return empty list."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        df = pd.DataFrame({"val": range(5)}, index=dates)
        assert find_missing_dates(df) == []

    def test_one_missing(self) -> None:
        """Should detect a single missing day."""
        dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-04"])
        df = pd.DataFrame({"val": range(3)}, index=dates)
        missing = find_missing_dates(df)
        assert len(missing) == 1
        assert missing[0] == pd.Timestamp("2025-01-03")

    def test_with_date_column(self) -> None:
        """Should work when date_col is specified."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-03"]),
                "val": [1, 2],
            },
        )
        missing = find_missing_dates(df, date_col="date")
        assert len(missing) == 1


class TestCheckNoNulls:
    """Tests for null detection."""

    def test_no_nulls(self) -> None:
        """Clean data should return empty dict."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert check_no_nulls(df) == {}

    def test_with_nulls(self) -> None:
        """Should report columns with null counts."""
        df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
        result = check_no_nulls(df)
        assert result == {"a": 1}


class TestValidateFeatureRanges:
    """Tests for range validation."""

    def test_all_in_range(self) -> None:
        """All values in range should return empty dict."""
        df = pd.DataFrame({"temp": [10, 20, 30]})
        assert validate_feature_ranges(df, {"temp": (0, 50)}) == {}

    def test_out_of_range(self) -> None:
        """Should detect values outside the bounds."""
        df = pd.DataFrame({"temp": [10, 20, 100]})
        result = validate_feature_ranges(df, {"temp": (0, 50)})
        assert result == {"temp": 1}
