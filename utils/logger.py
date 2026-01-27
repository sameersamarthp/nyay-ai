"""
Logging configuration for Nyay AI India.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from config.settings import settings


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(settings.LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        settings.ensure_directories()
        log_file = settings.LOGS_DIR / f"nyay_ai_{datetime.now():%Y%m%d}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(settings.LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


def setup_root_logger() -> None:
    """Configure the root logger for the application."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(settings.LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    settings.ensure_directories()
    log_file = settings.LOGS_DIR / f"nyay_ai_{datetime.now():%Y%m%d}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(settings.LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_script_logger(name: str) -> logging.Logger:
    """Get a logger configured for CLI scripts with clean console output.

    Like get_logger(), but console output has no timestamps for clean
    user-facing display. File output still includes full timestamps.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance with clean console output.
    """
    logger = logging.getLogger(f"{name}.script")

    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

        # Console handler - clean output without timestamps
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")  # No timestamp
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler - full timestamps
        settings.ensure_directories()
        log_file = settings.LOGS_DIR / f"nyay_ai_{datetime.now():%Y%m%d}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(settings.LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.propagate = False

    return logger
