"""RECAST — GRIB to Parquet converter.

Converts large ERA5 GRIB files to Parquet format for
efficient storage and fast loading during training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.loader import load_dataset
from src.utils.logger import get_logger

logger = get_logger("preprocessing.converter")


def grib_to_parquet(
    grib_path: str | Path,
    output_path: str | Path,
    *,
    variables: list[str] | None = None,
) -> Path:
    """Convert a GRIB file to Parquet format.

    Reads the GRIB via xarray, flattens spatial dimensions into
    columns, and writes an efficient Parquet file with a datetime
    index.

    Args:
        grib_path: Path to the input GRIB file.
        output_path: Path for the output Parquet file.
        variables: Specific variables to extract.  If ``None``,
            extracts all data variables.

    Returns:
        Path to the created Parquet file.
    """
    grib_path = Path(grib_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting %s → Parquet", grib_path.name)

    ds = load_dataset(grib_path)

    if variables:
        available = [v for v in variables if v in ds.data_vars]
        if available:
            ds = ds[available]
        else:
            logger.warning(
                "None of %s found in dataset. Available: %s",
                variables,
                list(ds.data_vars),
            )

    # Convert to DataFrame (flattens spatial dims)
    df = ds.to_dataframe().reset_index()

    # Try to standardise the time column name to "datetime"
    time_candidates = ["time", "valid_time", "datetime"]
    for col in time_candidates:
        if col in df.columns:
            df = df.rename(columns={col: "datetime"})
            break

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()

    df.to_parquet(output_path, engine="pyarrow")

    size_mb = output_path.stat().st_size / 1e6
    logger.info(
        "Parquet saved → %s (%.1f MB, %d rows × %d cols)",
        output_path.name,
        size_mb,
        len(df),
        len(df.columns),
    )
    return output_path


def append_parquet(
    new_data: str | Path | pd.DataFrame,
    existing_path: str | Path,
) -> Path:
    """Append new data to an existing Parquet file.

    If the target file does not exist, creates it.
    Deduplicates on index after concatenation.

    Args:
        new_data: Path to a Parquet file or a DataFrame.
        existing_path: Path to the target Parquet file.

    Returns:
        Path to the updated Parquet file.
    """
    existing_path = Path(existing_path)

    if isinstance(new_data, (str, Path)):
        new_df = pd.read_parquet(new_data)
    else:
        new_df = new_data

    if existing_path.exists():
        existing_df = pd.read_parquet(existing_path)
        combined = pd.concat([existing_df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        logger.info(
            "Appended %d new rows → %d total in %s",
            len(new_df),
            len(combined),
            existing_path.name,
        )
    else:
        combined = new_df.sort_index()
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Created %s with %d rows", existing_path.name, len(combined))

    combined.to_parquet(existing_path, engine="pyarrow")
    return existing_path
