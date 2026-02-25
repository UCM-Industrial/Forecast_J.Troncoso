"""RECAST — ECMWF Open Data client wrapper.

Thin wrapper around ``ecmwf.opendata.Client`` with automatic retries
and structured logging.  Only supports the **aifs-single** model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger("ingestion.ecmwf")


class ECMWFClient:
    """Client for downloading ECMWF Open Data forecasts.

    Wraps ``ecmwf.opendata.Client`` with retries, config-driven
    defaults, and structured logging.

    Args:
        model: ECMWF model name.  Defaults to config value.
        source: Data source (``"ecmwf"``).
    """

    def __init__(
        self,
        model: str | None = None,
        source: str = "ecmwf",
    ) -> None:
        from ecmwf.opendata import Client  # noqa: PLC0415

        settings = get_settings()
        self._model = model or settings.ecmwf.model
        self._client = Client(source=source, model=self._model)
        self._default_variables = settings.ecmwf.variables
        self._default_steps = settings.ecmwf.steps
        self._default_time = settings.ecmwf.time
        self._default_type = settings.ecmwf.type

        logger.info(
            "ECMWFClient initialised — model=%s, source=%s",
            self._model,
            source,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=lambda retry_state: get_logger("ingestion.ecmwf").warning(
            "Retry %d for ECMWF download…",
            retry_state.attempt_number,
        ),
    )
    def download_forecast(
        self,
        target: str | Path,
        *,
        date: str | int | None = None,
        time: int | None = None,
        steps: list[int] | None = None,
        variables: list[str] | None = None,
    ) -> Path:
        """Download a forecast dataset from ECMWF Open Data.

        Args:
            target: Destination file path for the GRIB download.
            date: Initialization date (``"YYYYMMDD"`` or ``int``).
                  If ``None``, uses the latest available date.
            time: Initialization hour (0, 6, 12, 18).  Defaults to config.
            steps: Forecast steps in hours.  Defaults to config.
            variables: Parameter short names.  Defaults to config.

        Returns:
            Path to the downloaded file.

        Raises:
            ConnectionError: After exhausting all retries.
        """
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        request: dict[str, Any] = {
            "type": self._default_type,
            "param": variables or self._default_variables,
            "step": steps or self._default_steps,
            "time": time if time is not None else self._default_time,
            "target": str(target),
        }

        if date is not None:
            request["date"] = date

        logger.info(
            "Downloading ECMWF forecast — model=%s, date=%s, time=%s, %d steps",
            self._model,
            date or "latest",
            request["time"],
            len(request["step"]),
        )

        self._client.retrieve(**request)

        logger.info(
            "Download complete → %s (%.1f MB)", target, target.stat().st_size / 1e6
        )
        return target
