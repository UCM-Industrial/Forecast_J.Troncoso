"""RECAST — ERA5 reanalysis data client.

Downloads historical climate data from the Copernicus Climate
Data Store (CDS) for model training.

.. note::

    This client is **mock-ready**: when ``CDS_API_KEY`` is not
    configured, it logs a warning and returns a placeholder path.
    Replace the mock with real ``cdsapi`` calls once credentials
    are set up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("ingestion.era5")


class ERA5Client:
    """Client for downloading ERA5 reanalysis data via CDS API.

    Args:
        dataset: CDS dataset name.  Defaults to config.
    """

    def __init__(self, dataset: str | None = None) -> None:
        settings = get_settings()
        self._dataset = dataset or settings.era5.dataset
        self._variables = settings.era5.variables
        self._area = settings.era5.area
        self._product_type = settings.era5.product_type
        self._format = settings.era5.format
        self._api_key = settings.cds_api_key
        self._api_url = settings.cds_api_url

        if not self._api_key:
            logger.warning(
                "CDS_API_KEY not configured — ERA5 downloads will use mock mode. "
                "Set CDS_API_KEY in .env to enable real downloads.",
            )

        logger.info("ERA5Client initialised — dataset=%s", self._dataset)

    def download(
        self,
        target: str | Path,
        *,
        year: str | int,
        month: str | int,
        days: list[str] | None = None,
        hours: list[str] | None = None,
        variables: list[str] | None = None,
    ) -> Path:
        """Download ERA5 reanalysis data for a given year/month.

        Args:
            target: Destination file path.
            year: Year to download (e.g. ``2024``).
            month: Month to download (e.g. ``1`` or ``"01"``).
            days: List of day strings.  Defaults to all days.
            hours: List of hour strings.  Defaults to ``["00:00", ..., "23:00"]``.
            variables: Override variables from config.

        Returns:
            Path to the downloaded GRIB file.
        """
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        year_str = str(year)
        month_str = str(month).zfill(2)

        if days is None:
            days = [str(d).zfill(2) for d in range(1, 32)]
        if hours is None:
            hours = [f"{h:02d}:00" for h in range(24)]

        request: dict[str, Any] = {
            "product_type": [self._product_type],
            "variable": variables or self._variables,
            "year": [year_str],
            "month": [month_str],
            "day": days,
            "time": hours,
            "area": self._area,
            "data_format": self._format,
        }

        logger.info(
            "Requesting ERA5 data — year=%s, month=%s, %d variables",
            year_str,
            month_str,
            len(request["variable"]),
        )

        if not self._api_key:
            return self._mock_download(target, year_str, month_str)

        return self._real_download(request, target)

    def _real_download(self, request: dict[str, Any], target: Path) -> Path:
        """Download via the CDS API."""
        import cdsapi  # noqa: PLC0415

        client = cdsapi.Client(url=self._api_url, key=self._api_key)
        client.retrieve(self._dataset, request, str(target))

        logger.info("ERA5 download complete → %s (%.1f MB)", target, target.stat().st_size / 1e6)
        return target

    def _mock_download(self, target: Path, year: str, month: str) -> Path:
        """Create a placeholder file when API key is not available."""
        target.write_text(
            f"MOCK ERA5 GRIB — {self._dataset} — {year}/{month}\n"
            f"Variables: {self._variables}\n"
            f"Area: {self._area}\n"
            "Replace this with real data by setting CDS_API_KEY.\n",
            encoding="utf-8",
        )
        logger.warning("Created mock ERA5 file → %s (no CDS_API_KEY)", target)
        return target
