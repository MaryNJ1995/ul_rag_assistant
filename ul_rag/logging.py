# ul_rag/logging.py
import logging
import os
from pathlib import Path

from .config import get_settings


def get_logger(name: str) -> logging.Logger:
    settings = get_settings()
    level_str = os.getenv("LOG_LEVEL", settings.log_level).upper()
    level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    log_file: Path = settings.log_dir / "ul_rag.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger
