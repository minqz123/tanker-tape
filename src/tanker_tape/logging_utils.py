"""Structured stage logging.

The brief requires row counts and data-quality stats at *every* pipeline stage, so
stages report through :func:`log_stage` rather than ad-hoc print statements.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Install a single stream handler. Safe to call more than once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for a module."""
    configure_logging()
    return logging.getLogger(name)


def log_stage(
    logger: logging.Logger,
    stage: str,
    frame: pd.DataFrame | None = None,
    **stats: Any,
) -> None:
    """Log the outcome of a pipeline stage.

    Args:
        logger: Module logger.
        stage: Short stage name, e.g. ``"portwatch.chokepoints"``.
        frame: Optional resulting frame; its row count and date span are logged.
        **stats: Extra key/value stats (dropped rows, flag counts, ...).
    """
    parts = [f"stage={stage}"]
    if frame is not None:
        parts.append(f"rows={len(frame)}")
        for column in ("date", "timestamp"):
            if column in frame.columns and not frame.empty:
                parts.append(f"{column}_min={frame[column].min()}")
                parts.append(f"{column}_max={frame[column].max()}")
                break
    parts.extend(f"{key}={value}" for key, value in stats.items())
    logger.info(" ".join(parts))


def log_quality(logger: logging.Logger, stage: str, flags: pd.DataFrame) -> None:
    """Log per-flag counts for a data-quality frame.

    Flag columns are boolean columns prefixed ``flag_``. Nothing is ever dropped here;
    this only reports what was marked.
    """
    flag_columns = [column for column in flags.columns if column.startswith("flag_")]
    if not flag_columns:
        logger.info("stage=%s quality=no_flag_columns rows=%d", stage, len(flags))
        return
    counts = {column: int(flags[column].sum()) for column in flag_columns}
    total = len(flags)
    summary = " ".join(f"{name}={count}" for name, count in counts.items())
    logger.info("stage=%s quality rows=%d %s", stage, total, summary)
    for name, count in counts.items():
        if total and count / total > 0.10:
            logger.warning(
                "stage=%s %s affects %.1f%% of rows - investigate before using this data",
                stage,
                name,
                100.0 * count / total,
            )
