import logging
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI colors for different log levels."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "TIME": "\033[90m",  # Grey
    }
    RESET = "\033[0m"

    # NOTE: override the original method to apply the color
    def formatTime(self, record, datefmt=None):  # noqa: N802
        """Override to add color to the timestamp."""
        asctime = super().formatTime(record, datefmt)
        return f"{self.COLORS['TIME']}{asctime}{self.RESET}"

    def format(self, record):
        """Override to add color to the log level name."""
        # Store original levelname to restore it later
        levelname_original = record.levelname

        # Apply color to the levelname
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            record.levelname = f"{color}{record.levelname}{self.RESET}"

        formatted_log = super().format(record)

        # Restore original levelname
        record.levelname = levelname_original

        return formatted_log


def setup_logging(
    logger_name: str | None = None,
    log_level: int = logging.INFO,
    log_dir: str | Path = "logs",
    console_output: bool = True,
) -> logging.Logger:
    """Setup logging configuration for a dedicated application logger."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent logs from propagating to the root logger

    # Clear existing handlers to prevent duplicate output
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    log_filepath = log_path / f"{datetime.now().strftime('%Y%m%d')}.log"

    # Create formatters (same as your original code)
    console_formatter = ColoredFormatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Add file handler
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized for '{logger_name}' - File: {log_filepath}")

    return logger


if __name__ == "__main__":
    app_logger = setup_logging(logger_name="logger_config", log_level=logging.INFO)
