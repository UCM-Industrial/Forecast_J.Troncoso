"""RECAST — CEN energy generation data loader.

Loads historical energy generation data (the training target)
from local CSV or Excel files published by the Chilean
Coordinador Eléctrico Nacional (CEN).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("ingestion.cen")


def load_cen_generation(
    source: str | Path | None = None,
    *,
    technology: str | None = None,
    datetime_col: str | None = None,
    generation_col: str | None = None,
) -> pd.DataFrame:
    """Load CEN generation data from local files.

    Supports CSV (``.csv``) and Excel (``.xlsx`` / ``.xls``) files.
    If ``source`` is a directory, all matching files inside it are
    concatenated.

    Args:
        source: Path to a file or directory containing CEN data.
            Defaults to ``cen.source_dir`` from config.
        technology: Optional filter (e.g. ``"solar"``, ``"wind"``).  If
            the file/column names include the technology, only matching
            data is returned.
        datetime_col: Name of the datetime column.  Defaults to config.
        generation_col: Name of the generation column.  Defaults to config.

    Returns:
        DataFrame with DatetimeIndex and at minimum the generation column.

    Raises:
        FileNotFoundError: If no matching files are found.
    """
    settings = get_settings()
    source = Path(source or settings.cen.source_dir)
    datetime_col = datetime_col or settings.cen.datetime_col
    generation_col = generation_col or settings.cen.generation_col

    # Collect files
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(
            [f for f in source.iterdir() if f.suffix in (".csv", ".xlsx", ".xls")],
        )
    else:
        msg = f"CEN source not found: {source}"
        raise FileNotFoundError(msg)

    if not files:
        msg = f"No CSV/Excel files found in {source}"
        raise FileNotFoundError(msg)

    logger.info("Loading CEN data from %d file(s) in %s", len(files), source)

    # Read and concatenate
    frames: list[pd.DataFrame] = []
    for f in files:
        if technology and technology.lower() not in f.stem.lower():
            continue

        if f.suffix == ".csv":
            df = pd.read_csv(f, parse_dates=[datetime_col])
        else:
            df = pd.read_excel(f, parse_dates=[datetime_col])

        frames.append(df)

    if not frames:
        msg = f"No files matched technology='{technology}' in {source}"
        raise FileNotFoundError(msg)

    result = pd.concat(frames, ignore_index=True)

    # Set datetime index
    if datetime_col in result.columns:
        result = result.set_index(datetime_col)
    result = result.sort_index()

    logger.info(
        "CEN data loaded: %d rows, columns=%s",
        len(result),
        list(result.columns),
    )
    return result
