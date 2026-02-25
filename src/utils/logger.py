"""RECAST — Structured logging setup.

Provides coloured console output and file logging,
configured from ``config/logging.yaml``.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml


class ColoredFormatter(logging.Formatter):
    """ANSI-coloured log formatter for terminal output."""

    COLORS: dict[str, str] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    GREY = "\033[90m"
    RESET = "\033[0m"

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        """Wrap the timestamp in grey."""
        asctime = super().formatTime(record, datefmt)
        return f"{self.GREY}{asctime}{self.RESET}"

    def format(self, record: logging.LogRecord) -> str:
        """Colour the level name."""
        original = record.levelname
        colour = self.COLORS.get(record.levelname, "")
        if colour:
            record.levelname = f"{colour}{record.levelname}{self.RESET}"
        formatted = super().format(record)
        record.levelname = original
        return formatted


def _find_project_root() -> Path:
    """Walk up from this file to find the project root."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "config").is_dir():
            return current
        current = current.parent
    return Path.cwd()


def setup_logging(
    config_path: str | Path | None = None,
    log_level: str | None = None,
) -> logging.Logger:
    """Initialise logging from YAML config.

    Args:
        config_path: Path to ``logging.yaml``.  Auto-detected if omitted.
        log_level: Override the root log level (e.g. ``"DEBUG"``).

    Returns:
        The ``recast`` logger instance ready for use.
    """
    if config_path is None:
        config_path = _find_project_root() / "config" / "logging.yaml"
    else:
        config_path = Path(config_path)

    # Ensure log directory exists
    log_dir = _find_project_root() / "logs"
    log_dir.mkdir(exist_ok=True)

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            log_cfg = yaml.safe_load(fh)
        # Resolve relative log file path
        if "handlers" in log_cfg and "file" in log_cfg["handlers"]:
            log_file = log_cfg["handlers"]["file"].get("filename", "logs/recast.log")
            log_cfg["handlers"]["file"]["filename"] = str(
                _find_project_root() / log_file,
            )
        logging.config.dictConfig(log_cfg)
    else:
        # Fallback: basic config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    logger = logging.getLogger("recast")

    if log_level:
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``recast`` namespace.

    Args:
        name: Module or component name (e.g. ``"ingestion"``).

    Returns:
        Logger instance ``recast.<name>``.
    """
    return logging.getLogger(f"recast.{name}")
