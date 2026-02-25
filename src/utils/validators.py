"""RECAST — Data validators.

Functions for checking time-series data completeness and quality.
Adapted from the legacy ``scripts/data_integrity.py``.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("validators")


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: list[str],
    *,
    name: str = "DataFrame",
) -> None:
    """Verify that a DataFrame contains all required columns.

    Args:
        df: Input DataFrame to validate.
        required_columns: List of column names that must be present.
        name: Human-readable name for error messages.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        msg = f"{name} is missing columns: {sorted(missing)}"
        raise ValueError(msg)
    logger.debug("Schema validation passed for %s", name)


def find_missing_dates(
    df: pd.DataFrame,
    date_col: str | None = None,
    freq: str = "D",
) -> list[pd.Timestamp]:
    """Find missing dates in a time series.

    Args:
        df: DataFrame to check. If ``date_col`` is None, uses the index.
        date_col: Column name containing dates, or ``None`` for index.
        freq: Expected frequency (``"D"``, ``"h"``, etc.).

    Returns:
        Sorted list of missing timestamps.
    """
    if date_col is not None:
        dates = pd.to_datetime(df[date_col])
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
    else:
        msg = "DataFrame must have a DatetimeIndex or specify date_col"
        raise ValueError(msg)

    if len(dates) == 0:
        return []

    full_range = pd.date_range(dates.min(), dates.max(), freq=freq)
    missing = sorted(set(full_range) - set(dates))

    if missing:
        logger.warning("Found %d missing dates in range", len(missing))

    return missing


def check_no_nulls(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    name: str = "DataFrame",
) -> dict[str, int]:
    """Check for null values in specified columns.

    Args:
        df: Input DataFrame.
        columns: Columns to check. If ``None``, checks all columns.
        name: Human-readable name for logging.

    Returns:
        Dictionary mapping column names to their null counts (only
        includes columns that have nulls).
    """
    cols = columns or df.columns.tolist()
    null_counts = {
        col: int(df[col].isna().sum()) for col in cols if df[col].isna().any()
    }

    if null_counts:
        logger.warning("%s has nulls: %s", name, null_counts)
    else:
        logger.debug("No nulls found in %s", name)

    return null_counts


def validate_feature_ranges(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    *,
    name: str = "features",
) -> dict[str, int]:
    """Check that feature values fall within expected ranges.

    Args:
        df: DataFrame to validate.
        bounds: Dict mapping column names to ``(min, max)`` tuples.
        name: Human-readable name for logging.

    Returns:
        Dict mapping column names to count of out-of-range values.
    """
    violations: dict[str, int] = {}
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        out_of_range = ~df[col].between(lo, hi)
        count = int(out_of_range.sum())
        if count > 0:
            violations[col] = count
            logger.warning(
                "%s: %d values in '%s' outside [%.2f, %.2f]",
                name,
                count,
                col,
                lo,
                hi,
            )
    return violations
