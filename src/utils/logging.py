"""Standardized logging: console + per-logger file handler under logs/.

Use ``get_logger(__name__)`` at the top of any module/script. Following the
"no silent failures / no silent drops" principle, log row counts and filter
reasons explicitly through these loggers.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.io import project_root

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger writing to the console and ``logs/<name>.log``.

    Idempotent: repeated calls with the same name do not add duplicate handlers.
    The ``logs/`` directory is created if it does not exist.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:  # already configured
        return logger

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    logs_dir = project_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("/", "_").replace("\\", "_")
    file_handler = logging.FileHandler(logs_dir / f"{safe_name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
