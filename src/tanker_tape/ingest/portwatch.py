"""IMF PortWatch ingestion via ArcGIS FeatureServer, with vintage capture.

PortWatch publishes weekly (Tuesdays 09:00 ET) and **revises** history as sampling and
spoofing checks change. Every pull here is stored as an immutable vintage
(:func:`tanker_tape.storage.write_vintage`) so analyses can reconstruct what was knowable
on a given date instead of quietly consuming revised numbers.

Schema tolerance is deliberate. The FeatureServer could not be reached from the build
container (see the endpoint verification log in ``CLAUDE.md``), so column names are
*discovered* from the live response against candidate lists rather than assumed. When
discovery fails the error names the columns that were actually returned, which is the
fastest path to fixing it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from typing import Any

import httpx
import pandas as pd

from ..logging_utils import get_logger, log_stage

logger = get_logger(__name__)

ARCGIS_ROOT = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"

# UNVERIFIED: see the endpoint verification log in CLAUDE.md. Override per call if the
# live "API" tab on portwatch.imf.org disagrees.
CHOKEPOINTS_LAYER = f"{ARCGIS_ROOT}/Daily_Chokepoints_Data/FeatureServer/0"
PORTS_LAYER = f"{ARCGIS_ROOT}/Daily_Ports_Data/FeatureServer/0"

# ArcGIS caps a single response; PortWatch's observed cap is 1000 records. The real
# cap is read from the layer metadata where it is published.
PAGE_SIZE = 1000

# Guard rail against an accidental full-layer pull. The daily ports layer held more
# than 1.6 million rows when this was measured on 2026-09-16 and was still paging, so
# an unfiltered pull is never the right thing to do.
MAX_RECORDS = 500_000

_DATE_COLUMN_CANDIDATES = ("date", "Date", "DATE", "date_", "day_date")
_ID_COLUMN_CANDIDATES = ("portid", "PORTID", "chokepoint_id", "chokepointid", "objectid_ref")
_NAME_COLUMN_CANDIDATES = ("portname", "PORTNAME", "chokepoint", "chokepoint_name", "name")


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    """Return the first candidate column present in ``frame``.

    Raises:
        KeyError: If none match, naming the columns that were actually returned.
    """
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(
        f"could not find the {role} column. Tried {list(candidates)}; "
        f"the layer returned {sorted(frame.columns)}. Update the candidate list in "
        "ingest/portwatch.py and note the deviation in CLAUDE.md."
    )


def fetch_layer_metadata(layer_url: str, timeout: float = 60.0) -> dict[str, Any]:
    """Fetch an ArcGIS layer's metadata (field names, types, record limits)."""
    with httpx.Client(timeout=timeout) as client:
        response = client.get(layer_url, params={"f": "json"})
        response.raise_for_status()
        return response.json()


def _date_field_names(metadata: dict[str, Any]) -> set[str]:
    """Names of fields the layer declares as dates (returned as epoch milliseconds)."""
    return {
        field["name"]
        for field in metadata.get("fields", [])
        if field.get("type") == "esriFieldTypeDate"
    }


def iter_features(
    layer_url: str,
    where: str = "1=1",
    out_fields: str = "*",
    page_size: int = PAGE_SIZE,
    order_by: str = "ObjectId ASC",
    timeout: float = 120.0,
    max_records: int | None = MAX_RECORDS,
) -> Iterator[dict[str, Any]]:
    """Page through a FeatureServer layer, yielding attribute dicts.

    Paging uses a stable ``ObjectId`` sort so that pages do not overlap or skip rows
    if the service reorders results between requests.

    ``max_records`` is a guard rail, not a limit to tune. The ports layer holds several
    million rows, and an unfiltered pull of it runs for hours before anything notices —
    a scheduled job just dies on its timeout with nothing written. Hitting this cap
    means the query was too broad; narrow the ``where`` clause rather than raising it.

    Args:
        layer_url: Layer URL without the trailing ``/query``.
        where: SQL-ish filter, e.g. ``"date >= DATE '2024-01-01'"``.
        out_fields: Comma-separated fields, or ``"*"``.
        page_size: Records per request.
        order_by: ArcGIS ``orderByFields`` value.
        timeout: Per-request timeout in seconds.
        max_records: Abort after this many rows; ``None`` disables the guard.

    Yields:
        One dict of attributes per feature.

    Raises:
        RuntimeError: If the service errors, or the result exceeds ``max_records``.
    """
    offset = 0
    with httpx.Client(timeout=timeout) as client:
        while True:
            params = {
                "where": where,
                "outFields": out_fields,
                "returnGeometry": "false",
                "orderByFields": order_by,
                "resultOffset": offset,
                "resultRecordCount": page_size,
                "f": "json",
            }
            response = client.get(f"{layer_url}/query", params=params)
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(f"ArcGIS error from {layer_url}: {payload['error']}")

            features = payload.get("features", [])
            for feature in features:
                yield feature.get("attributes", {})

            offset_after = offset + len(features)
            if max_records is not None and offset_after >= max_records:
                raise RuntimeError(
                    f"query returned more than {max_records:,} rows from {layer_url} and was "
                    f"stopped (where={where!r}). The full ports layer is several million rows; "
                    "filter it server-side - ingest_ports() does this by resolving the ports in "
                    "config/ports.yaml to IDs first."
                )

            logger.debug("portwatch page offset=%d returned=%d", offset, len(features))
            # `exceededTransferLimit` is the authoritative "there is more" signal; the
            # short-page check covers services that omit it.
            if not features or not payload.get("exceededTransferLimit", len(features) == page_size):
                break
            offset += len(features)


def fetch_layer(
    layer_url: str,
    where: str = "1=1",
    page_size: int = PAGE_SIZE,
    max_records: int | None = MAX_RECORDS,
) -> pd.DataFrame:
    """Fetch an entire layer into a DataFrame, converting ArcGIS date fields.

    Args:
        layer_url: Layer URL without the trailing ``/query``.
        where: SQL-ish filter.
        page_size: Records per request.

    Returns:
        The layer's attributes, with declared date fields converted to UTC timestamps.
    """
    metadata = fetch_layer_metadata(layer_url)
    date_fields = _date_field_names(metadata)
    logger.info(
        "stage=portwatch.metadata layer=%s name=%s date_fields=%s max_record_count=%s",
        layer_url,
        metadata.get("name"),
        sorted(date_fields) or "none",
        metadata.get("maxRecordCount"),
    )

    # The service publishes its own page cap; using it cuts the number of round trips.
    published_cap = metadata.get("maxRecordCount")
    if isinstance(published_cap, int) and published_cap > page_size:
        logger.info("using the layer's published maxRecordCount=%d for paging", published_cap)
        page_size = published_cap

    frame = pd.DataFrame(
        list(iter_features(layer_url, where=where, page_size=page_size, max_records=max_records))
    )
    if frame.empty:
        logger.warning("layer %s returned no rows for where=%r", layer_url, where)
        return frame

    for field in date_fields & set(frame.columns):
        # ArcGIS serialises dates as epoch milliseconds (UTC) under f=json.
        frame[field] = pd.to_datetime(frame[field], unit="ms", utc=True)

    log_stage(logger, "portwatch.fetch_layer", frame, layer=metadata.get("name"))
    return frame


def normalise_daily_table(frame: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Standardise a PortWatch daily table's key columns.

    Args:
        frame: Raw layer output.
        entity: ``"chokepoint"`` or ``"port"``; used for the emitted column names.

    Returns:
        The frame with ``date`` (a ``datetime.date``), ``{entity}_id`` and
        ``{entity}_name`` columns added, sorted by entity then date. All original
        columns are preserved — measure columns are passed through untouched because
        their names are unverified.
    """
    if frame.empty:
        return frame

    date_column = _pick_column(frame, _DATE_COLUMN_CANDIDATES, "date")
    id_column = _pick_column(frame, _ID_COLUMN_CANDIDATES, f"{entity} id")
    name_column = _pick_column(frame, _NAME_COLUMN_CANDIDATES, f"{entity} name")

    out = frame.copy()
    dates = out[date_column]
    if pd.api.types.is_numeric_dtype(dates):
        # Defensive: the layer metadata did not declare this as a date field.
        dates = pd.to_datetime(dates, unit="ms", utc=True)
    else:
        dates = pd.to_datetime(dates, utc=True, errors="coerce")

    unparsed = int(dates.isna().sum())
    out["date"] = dates.dt.date
    out[f"{entity}_id"] = out[id_column].astype(str)
    out[f"{entity}_name"] = out[name_column].astype(str)
    out = out.sort_values([f"{entity}_id", "date"]).reset_index(drop=True)

    log_stage(
        logger,
        f"portwatch.normalise.{entity}",
        out,
        date_column=date_column,
        id_column=id_column,
        name_column=name_column,
        unparsed_dates=unparsed,
        entities=out[f"{entity}_id"].nunique(),
    )
    if unparsed:
        logger.warning(
            "%d rows had an unparseable date and are flagged, not dropped (column=%s)",
            unparsed,
            date_column,
        )
    return out


def resolve_entity_ids(frame: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Build the distinct id/name lookup for a normalised daily table.

    Use this to confirm configured IDs (for example whether Hormuz really is
    ``chokepoint6``) instead of trusting the brief.

    Returns:
        Columns ``{entity}_id``, ``{entity}_name``, ``n_days``, ``first_date``, ``last_date``.
    """
    id_column, name_column = f"{entity}_id", f"{entity}_name"
    lookup = (
        frame.groupby([id_column, name_column], as_index=False)
        .agg(n_days=("date", "size"), first_date=("date", "min"), last_date=("date", "max"))
        .sort_values(name_column)
        .reset_index(drop=True)
    )
    return lookup


def ingest_chokepoints(
    where: str = "1=1",
    layer_url: str = CHOKEPOINTS_LAYER,
    retrieved_at: dt.datetime | None = None,
    store: bool = True,
) -> pd.DataFrame:
    """Pull the daily chokepoint table and store it as a vintage.

    Args:
        where: Optional server-side filter.
        layer_url: Override the layer URL.
        retrieved_at: Pull timestamp; defaults to now.
        store: Whether to persist the vintage.

    Returns:
        The normalised frame.
    """
    from ..storage import write_vintage

    frame = normalise_daily_table(fetch_layer(layer_url, where=where), "chokepoint")
    if store and not frame.empty:
        write_vintage(frame, "portwatch_chokepoints", retrieved_at=retrieved_at)
    return frame


def fetch_distinct(
    layer_url: str,
    fields: str,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """Fetch the distinct combinations of a few fields.

    Used to pull the ~2,000-row port directory without touching the millions of daily
    rows behind it.

    Args:
        layer_url: Layer URL without the trailing ``/query``.
        fields: Comma-separated field names.
        timeout: Request timeout in seconds.

    Returns:
        One row per distinct combination.

    Raises:
        RuntimeError: If the service rejects the distinct query.
    """
    params = {
        "where": "1=1",
        "outFields": fields,
        "returnGeometry": "false",
        "returnDistinctValues": "true",
        "resultRecordCount": 10000,
        "f": "json",
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.get(f"{layer_url}/query", params=params)
        response.raise_for_status()
        payload = response.json()
    if "error" in payload:
        raise RuntimeError(
            f"distinct query failed on {layer_url}: {payload['error']}. Pin explicit "
            "portwatch_id values in config/ports.yaml so resolution does not need it."
        )

    frame = pd.DataFrame([feature.get("attributes", {}) for feature in payload.get("features", [])])
    log_stage(logger, "portwatch.fetch_distinct", frame, fields=fields)
    return frame


def resolve_port_ids(
    layer_url: str = PORTS_LAYER,
    ports: dict[str, Any] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Resolve the ports in ``config/ports.yaml`` to live PortWatch IDs.

    This is what makes the daily pull tractable: with IDs in hand the query filters
    server-side to the handful of terminals actually being tracked, instead of
    downloading every port on earth and discarding 99% of it.

    Args:
        layer_url: Ports layer URL.
        ports: Port config; loaded from ``config/ports.yaml`` when omitted.

    Returns:
        ``(resolved, unresolved)`` mapping each config key to its matching IDs, and
        listing the keys that matched nothing.
    """
    from ..config import load_ports

    configured = ports if ports is not None else load_ports()
    directory = fetch_distinct(layer_url, "portid,portname")
    if directory.empty:
        raise RuntimeError(f"the port directory came back empty from {layer_url}")

    id_column = _pick_column(directory, _ID_COLUMN_CANDIDATES, "port id")
    name_column = _pick_column(directory, _NAME_COLUMN_CANDIDATES, "port name")
    lowered = directory[name_column].astype(str).str.lower()

    resolved: dict[str, list[str]] = {}
    unresolved: list[str] = []
    for key, port in configured.items():
        if port.portwatch_id and (directory[id_column] == port.portwatch_id).any():
            resolved[key] = [port.portwatch_id]
            continue
        matches = directory.loc[
            lowered.str.contains(port.match_name.lower(), regex=False, na=False), id_column
        ]
        if matches.empty:
            unresolved.append(key)
            continue
        resolved[key] = [str(value) for value in matches]

    logger.info(
        "stage=portwatch.resolve_port_ids resolved=%d unresolved=%d directory_size=%d",
        len(resolved),
        len(unresolved),
        len(directory),
    )
    if unresolved:
        logger.warning(
            "could not resolve %d configured port(s): %s. They are EXCLUDED from the pull, "
            "not silently zeroed.",
            len(unresolved),
            ", ".join(unresolved),
        )
    return resolved, unresolved


def ingest_ports(
    where: str | None = None,
    layer_url: str = PORTS_LAYER,
    retrieved_at: dt.datetime | None = None,
    store: bool = True,
    all_ports: bool = False,
) -> pd.DataFrame:
    """Pull the daily port activity table and store it as a vintage.

    By default this pulls **only the ports configured in ``config/ports.yaml``**, by
    resolving them to IDs and filtering server-side. That is not an optimisation: the
    full layer was over 1.6 million rows and still paging after 30 minutes when it was
    measured, so an unfiltered pull cannot finish inside any reasonable job timeout.

    Args:
        where: Explicit filter, bypassing port resolution entirely.
        layer_url: Override the layer URL.
        retrieved_at: Pull timestamp; defaults to now.
        store: Whether to persist the vintage.
        all_ports: Pull every port. Expect hours, and raise ``max_records`` first.

    Returns:
        The normalised frame.

    Raises:
        RuntimeError: If no configured port could be resolved.
    """
    from ..storage import write_vintage

    if where is None and not all_ports:
        resolved, unresolved = resolve_port_ids(layer_url)
        if not resolved:
            raise RuntimeError(
                "none of the ports in config/ports.yaml matched the live PortWatch "
                f"directory (tried {len(unresolved)}). Run `tanker-tape resolve-ids port` "
                "to see the real names, then fix match_name or pin portwatch_id."
            )
        identifiers = sorted({value for values in resolved.values() for value in values})
        quoted = ", ".join(f"'{value}'" for value in identifiers)
        where = f"portid IN ({quoted})"
        logger.info(
            "stage=portwatch.ingest_ports filtering to %d port id(s) from %d configured entries",
            len(identifiers),
            len(resolved),
        )
    elif all_ports:
        where = where or "1=1"
        logger.warning(
            "pulling EVERY port. This layer had >1.6M rows when last measured and will "
            "likely exceed max_records or the job timeout."
        )

    frame = normalise_daily_table(fetch_layer(layer_url, where=where), "port")
    if store and not frame.empty:
        write_vintage(frame, "portwatch_ports", retrieved_at=retrieved_at)
    return frame
