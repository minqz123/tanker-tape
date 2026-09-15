"""Parquet + DuckDB storage with vintage (as-of) support.

PortWatch revises published history, so "what does the table say today" and "what did
the table say on 2026-03-09" are different questions. Every raw pull is written under a
``retrieved_at=`` partition and read back with :func:`read_vintage`, which resolves the
newest vintage that existed *on or before* a given date. Analyses that silently read
"latest" are using data that was not available at the time and will overstate results.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import duckdb
import pandas as pd

from .config import get_settings
from .logging_utils import get_logger

logger = get_logger(__name__)

_VINTAGE_DIR_RE = re.compile(r"^retrieved_at=(?P<stamp>\d{8}T\d{6}Z)$")
_VINTAGE_FORMAT = "%Y%m%dT%H%M%SZ"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def encode_vintage(retrieved_at: dt.datetime) -> str:
    """Render a UTC timestamp as a partition directory name."""
    if retrieved_at.tzinfo is None:
        retrieved_at = retrieved_at.replace(tzinfo=dt.UTC)
    return retrieved_at.astimezone(dt.UTC).strftime(_VINTAGE_FORMAT)


def decode_vintage(stamp: str) -> dt.datetime:
    """Parse a partition directory name back into a UTC timestamp."""
    return dt.datetime.strptime(stamp, _VINTAGE_FORMAT).replace(tzinfo=dt.UTC)


def raw_dir(dataset: str) -> Path:
    """Directory holding all vintages of a raw dataset."""
    return get_settings().data_dir / "raw" / dataset


def processed_path(name: str) -> Path:
    """Path for a processed table."""
    return get_settings().data_dir / "processed" / f"{name}.parquet"


def write_vintage(
    frame: pd.DataFrame,
    dataset: str,
    retrieved_at: dt.datetime | None = None,
) -> Path:
    """Write a raw pull as a new immutable vintage.

    Args:
        frame: The pulled data. A ``retrieved_at`` column is added if absent.
        dataset: Dataset name, e.g. ``"portwatch_chokepoints"``.
        retrieved_at: Pull timestamp; defaults to now (UTC).

    Returns:
        Path of the written parquet file.
    """
    stamp = retrieved_at or _utc_now()
    target_dir = raw_dir(dataset) / f"retrieved_at={encode_vintage(stamp)}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "data.parquet"

    payload = frame.copy()
    if "retrieved_at" not in payload.columns:
        payload["retrieved_at"] = pd.Timestamp(stamp).tz_convert("UTC")
    payload.to_parquet(target, index=False)
    logger.info(
        "stage=storage.write_vintage dataset=%s rows=%d vintage=%s path=%s",
        dataset,
        len(payload),
        encode_vintage(stamp),
        target,
    )
    return target


def list_vintages(dataset: str) -> list[dt.datetime]:
    """All vintages available for a dataset, oldest first."""
    directory = raw_dir(dataset)
    if not directory.exists():
        return []
    stamps = []
    for child in directory.iterdir():
        match = _VINTAGE_DIR_RE.match(child.name)
        if match and (child / "data.parquet").exists():
            stamps.append(decode_vintage(match.group("stamp")))
    return sorted(stamps)


def resolve_vintage(dataset: str, as_of: dt.datetime | dt.date | None = None) -> dt.datetime:
    """Find the newest vintage retrieved on or before ``as_of``.

    Args:
        dataset: Dataset name.
        as_of: Point-in-time cutoff. ``None`` means "latest available", which is only
            appropriate for live monitoring, never for backtests.

    Raises:
        FileNotFoundError: If the dataset has no vintages, or none old enough.
    """
    available = list_vintages(dataset)
    if not available:
        raise FileNotFoundError(f"no vintages stored for dataset {dataset!r}; run the ingest first")
    if as_of is None:
        return available[-1]

    cutoff = as_of
    if isinstance(cutoff, dt.date) and not isinstance(cutoff, dt.datetime):
        cutoff = dt.datetime.combine(cutoff, dt.time.max)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=dt.UTC)

    eligible = [stamp for stamp in available if stamp <= cutoff]
    if not eligible:
        raise FileNotFoundError(
            f"dataset {dataset!r} has no vintage retrieved on or before {cutoff.isoformat()}; "
            f"earliest is {available[0].isoformat()}. Backfilling from a later vintage would "
            "leak revised data into the past."
        )
    return eligible[-1]


def read_vintage(
    dataset: str,
    as_of: dt.datetime | dt.date | None = None,
) -> pd.DataFrame:
    """Read a raw dataset as it stood on ``as_of``.

    Args:
        dataset: Dataset name.
        as_of: Point-in-time cutoff; ``None`` reads the latest vintage.

    Returns:
        The stored frame for the resolved vintage.
    """
    stamp = resolve_vintage(dataset, as_of)
    path = raw_dir(dataset) / f"retrieved_at={encode_vintage(stamp)}" / "data.parquet"
    frame = pd.read_parquet(path)
    logger.info(
        "stage=storage.read_vintage dataset=%s rows=%d vintage=%s as_of=%s",
        dataset,
        len(frame),
        encode_vintage(stamp),
        as_of,
    )
    return frame


def write_processed(frame: pd.DataFrame, name: str) -> Path:
    """Write a processed table, overwriting any previous version."""
    target = processed_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    logger.info("stage=storage.write_processed table=%s rows=%d path=%s", name, len(frame), target)
    return target


def read_processed(name: str) -> pd.DataFrame:
    """Read a processed table."""
    target = processed_path(name)
    if not target.exists():
        raise FileNotFoundError(f"processed table {name!r} not found at {target}")
    return pd.read_parquet(target)


def append_partition(frame: pd.DataFrame, dataset: str, partition: str) -> Path:
    """Append-only write used by the live AIS collector.

    Args:
        frame: Rows to persist.
        dataset: Dataset name, e.g. ``"ais_positions"``.
        partition: Partition label, typically ``YYYY-MM-DD/HH``.

    Returns:
        Path of the written parquet file.
    """
    target_dir = get_settings().data_dir / "raw" / dataset / partition
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"part-{_utc_now().strftime('%Y%m%dT%H%M%S%f')}.parquet"
    frame.to_parquet(target, index=False)
    return target


def read_partitions(
    dataset: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Read the hourly partitions the live collector writes.

    Layout is ``raw/<dataset>/<YYYY-MM-DD>/<HH>/part-*.parquet``, matching
    :func:`append_partition`. Dates are filtered from the directory names rather
    than by reading every file, so a narrow window stays cheap on a lake holding
    months of collection.

    Args:
        dataset: Dataset name, e.g. ``"ais_positions"``.
        start: Inclusive first date to read.
        end: Inclusive last date to read.

    Returns:
        The concatenated frame, or an empty frame when nothing matches.
    """
    root = raw_dir(dataset)
    if not root.exists():
        logger.warning("no partitions for dataset %r at %s", dataset, root)
        return pd.DataFrame()

    selected = []
    skipped = 0
    for parquet in sorted(root.glob("*/*/*.parquet")):
        day = parquet.parent.parent.name
        try:
            parsed = dt.date.fromisoformat(day)
        except ValueError:
            skipped += 1
            continue
        if start is not None and parsed < start:
            continue
        if end is not None and parsed > end:
            continue
        selected.append(parquet)

    if skipped:
        logger.warning(
            "ignored %d file(s) under %s whose parent directory is not a YYYY-MM-DD date",
            skipped,
            root,
        )
    if not selected:
        logger.warning("dataset %r has no partitions between %s and %s", dataset, start, end)
        return pd.DataFrame()

    frame = pd.concat([pd.read_parquet(path) for path in selected], ignore_index=True)
    logger.info(
        "stage=storage.read_partitions dataset=%s files=%d rows=%d start=%s end=%s",
        dataset,
        len(selected),
        len(frame),
        start,
        end,
    )
    return frame


def duckdb_connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB catalogue, creating the file and its directory if needed."""
    path = get_settings().duckdb_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def register_processed_views(connection: duckdb.DuckDBPyConnection) -> list[str]:
    """Expose every processed parquet table as a DuckDB view.

    Returns:
        The view names created.
    """
    directory = get_settings().data_dir / "processed"
    if not directory.exists():
        return []
    created = []
    for parquet in sorted(directory.glob("*.parquet")):
        view = parquet.stem
        connection.execute(
            f'CREATE OR REPLACE VIEW "{view}" AS SELECT * FROM read_parquet(?)',
            [str(parquet)],
        )
        created.append(view)
    logger.info("stage=storage.register_views views=%s", ",".join(created) or "none")
    return created
