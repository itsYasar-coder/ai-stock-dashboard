"""Application-wide logging setup."""

from __future__ import annotations

import logging
import sys

from config.settings import SETTINGS

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once per process (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logging.basicConfig(level=getattr(logging, (level or SETTINGS.log_level).upper(), logging.INFO), handlers=[handler])
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring logging is configured first."""
    setup_logging()
    return logging.getLogger(name)
