"""RECAST — Feature engineering.

Functions for creating ML-ready features from processed
climate data (regional CSVs).

Adapted from legacy ``_util.py`` and ``forecast.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("preprocessing.features")


# ── Datetime helpers ─────────────────────────────────────────


def read_csv_with_datetime(
    path: str | Path,
    datetime_col: str = "datetime",
    output_timezone: str = "America/Santiago",
) -> pd.DataFrame:
    """Read a CSV and set a timezone-aware datetime index.

    Args:
        path: Path to the CSV file.
        datetime_col: Column to parse as datetime.
        output_timezone: Target timezone for the index.

    Returns:
        DataFrame with a ``DatetimeIndex`` in the target timezone.
    """
    df = pd.read_csv(path, parse_dates=[datetime_col])
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="raise", utc=True)
    df[datetime_col] = df[datetime_col].dt.tz_convert(output_timezone)
    return df.set_index(datetime_col)


# ── Cyclical encoding ────────────────────────────────────────


_PERIOD_MAX: dict[str, int] = {
    "hour": 24,
    "day": 31,
    "month": 12,
    "dayofweek": 7,
}


def create_cyclical_features(
    df: pd.DataFrame,
    datetime_col: str | None = None,
    features: list[str] | None = None,
    *,
    include_year: bool = True,
) -> pd.DataFrame:
    """Apply sin/cos cyclical encoding to datetime components.

    Args:
        df: Input DataFrame.
        datetime_col: Column name with datetime values.
            If ``None``, uses the DatetimeIndex.
        features: Which datetime parts to encode.
            Defaults to ``["hour", "day", "month"]``.
        include_year: Whether to add a plain ``year`` column.

    Returns:
        Copy of the DataFrame with added cyclical feature columns.

    Raises:
        ValueError: If no datetime source is found.
    """
    if features is None:
        features = ["hour", "day", "month"]

    temp_df = df.copy()

    # Resolve datetime source
    if datetime_col:
        if datetime_col not in temp_df.columns:
            msg = f"Column '{datetime_col}' not found in DataFrame"
            raise ValueError(msg)
        dt_series = temp_df[datetime_col]
    elif isinstance(temp_df.index, pd.DatetimeIndex):
        dt_series = temp_df.index.to_series()
    else:
        msg = "DataFrame must have a DatetimeIndex or specify datetime_col"
        raise ValueError(msg)

    if not pd.api.types.is_datetime64_any_dtype(dt_series):
        dt_series = pd.to_datetime(dt_series)

    # Generate sin/cos features
    _EXTRACTORS: dict[str, pd.Series] = {
        "hour": dt_series.dt.hour,
        "day": dt_series.dt.day,
        "month": dt_series.dt.month,
        "dayofweek": dt_series.dt.dayofweek,
    }

    for feat in features:
        if feat not in _PERIOD_MAX:
            logger.warning("Unknown feature '%s' — skipping", feat)
            continue

        values = _EXTRACTORS[feat]
        max_val = _PERIOD_MAX[feat]
        temp_df[f"{feat}_sin"] = np.sin(2 * np.pi * values / max_val)
        temp_df[f"{feat}_cos"] = np.cos(2 * np.pi * values / max_val)

    if include_year:
        temp_df["year"] = dt_series.dt.year

    logger.debug("Created cyclical features: %s", features)
    return temp_df


# ── Technology-specific feature preparation ──────────────────


def prepare_ml_features(
    csv_dir: str | Path,
    technology: str,
    *,
    n_generation_plants: int | None = None,
    cyclical_features: list[str] | None = None,
) -> pd.DataFrame:
    """Build ML-ready feature DataFrame for a given energy technology.

    Args:
        csv_dir: Directory containing the regional-mean CSV files.
        technology: ``"wind"`` or ``"solar"``.
        n_generation_plants: Optional plant count feature.
        cyclical_features: Datetime features to encode.
            Defaults to ``["hour", "month", "day", "dayofweek"]``.

    Returns:
        Feature DataFrame ready for model training or prediction.

    Raises:
        ValueError: If technology is not supported.
    """
    if cyclical_features is None:
        cyclical_features = ["hour", "month", "day", "dayofweek"]

    tech = technology.lower()

    if tech == "wind":
        return _prepare_wind_features(csv_dir, cyclical_features, n_generation_plants)
    if tech == "solar":
        return _prepare_solar_features(csv_dir, cyclical_features, n_generation_plants)

    msg = f"Unsupported technology: '{technology}'. Use 'wind' or 'solar'."
    raise ValueError(msg)


def _prepare_wind_features(
    csv_dir: str | Path,
    cyclical_features: list[str],
    n_plants: int | None,
) -> pd.DataFrame:
    """Build wind features from u100 and v100 regional means."""
    csv_dir = Path(csv_dir)
    u100_path = csv_dir / "u100_means.csv"
    v100_path = csv_dir / "v100_means.csv"

    if not u100_path.exists() or not v100_path.exists():
        msg = f"Wind component files not found in {csv_dir}"
        raise FileNotFoundError(msg)

    u100 = read_csv_with_datetime(u100_path)
    v100 = read_csv_with_datetime(v100_path)

    wind_data = u100.join(
        v100,
        how="inner",
        lsuffix="_u100_means",
        rsuffix="_v100_means",
    )

    if n_plants is not None:
        wind_data["total_plants"] = n_plants

    logger.info("Prepared wind features: %d rows × %d cols", *wind_data.shape)
    return create_cyclical_features(wind_data, features=cyclical_features)


def _prepare_solar_features(
    csv_dir: str | Path,
    cyclical_features: list[str],
    n_plants: int | None,
) -> pd.DataFrame:
    """Build solar features from SSRD regional means."""
    csv_dir = Path(csv_dir)
    ssrd_path = csv_dir / "ssrd_means.csv"

    if not ssrd_path.exists():
        msg = f"Solar radiation file not found: {ssrd_path}"
        raise FileNotFoundError(msg)

    ssrd = read_csv_with_datetime(ssrd_path)

    # Zero-out duplicate rows (day/night cycle artefact)
    mask = ssrd.duplicated(keep="first")
    ssrd.loc[mask] = 0

    if n_plants is not None:
        ssrd["total_plants"] = n_plants

    logger.info("Prepared solar features: %d rows × %d cols", *ssrd.shape)
    return create_cyclical_features(ssrd, features=cyclical_features)
