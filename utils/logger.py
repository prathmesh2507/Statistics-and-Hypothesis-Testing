"""
utils/logger.py
───────────────
Centralized logging for EVA.

Uses Python's standard logging wired up with a Rich handler for
beautiful terminal output, plus a rotating file handler for persistence.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("EVA started")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)
_initialized: set[str] = set()


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """
    Return a module-level logger.
    First call configures handlers; subsequent calls return the cached logger.
    """
    logger = logging.getLogger(name)

    if name in _initialized:
        return logger

    _initialized.add(name)

    # Resolve log level (env → arg → default)
    import os
    effective_level = level or os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, effective_level.upper(), logging.INFO)

    logger.setLevel(numeric_level)
    logger.propagate = False  # Don't bubble up to root logger

    # ── Rich (terminal) handler ────────────────────────────────
    rich_handler = RichHandler(
        console=_console,
        show_time=True,
        show_path=True,
        rich_tracebacks=True,
        markup=True,
    )
    rich_handler.setLevel(numeric_level)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    # ── File handler ───────────────────────────────────────────
    log_file = Path(os.getenv("LOG_FILE", "./logs/eva.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    return logger


def get_root_logger() -> logging.Logger:
    """Return the top-level 'eva' logger (used in main.py)."""
    return get_logger("eva")
