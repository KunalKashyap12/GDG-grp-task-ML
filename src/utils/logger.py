"""
Logging Infrastructure for Experiments and Diagnostics.
Provides consistent timestamps, level filtering, and optional console highlighting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "crop_dl", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Constructs and configures a standard logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already initialized
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
