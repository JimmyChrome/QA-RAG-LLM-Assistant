"""Application logging configuration.

This module provides one reusable logging setup for the entire backend.

Usage:
    from app.core.logging import get_logger, setup_logging

    setup_logging()
    logger = get_logger(__name__)
    logger.info("Application started")
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Final

from app.core.config import settings


_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)

_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def _build_logging_config() -> dict:
    """Build the dictionary used by Python's logging configuration."""

    log_file: Path = settings.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": _LOG_FORMAT,
                "datefmt": _DATE_FORMAT,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.log_level,
                "formatter": "standard",
                "filename": str(log_file),
                "maxBytes": 5_000_000,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["console", "file"],
        },
        "loggers": {
            "uvicorn": {
                "level": settings.log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "uvicorn.error": {
                "level": settings.log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": settings.log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
    }


def setup_logging() -> None:
    """Configure console and rotating-file logging for the backend."""
    logging.config.dictConfig(_build_logging_config())


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a module."""
    return logging.getLogger(name)
