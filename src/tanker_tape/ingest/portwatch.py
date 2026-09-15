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

# ArcGIS caps a single response; PortWatch's documented cap is 1000 records.
PAGE_SIZE = 1000

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
) -> Iterator[dict[str, Any]]:
    """Page through a FeatureServer layer, yielding attribute dicts.

    Paging uses a stable ``ObjectId`` sort so that pages do not overlap or skip rows
    if the service reorders results between requests.

    Args:
        layer_url: Layer URL without the trailing ``/query``.
        where: SQL-ish filter, e.g. ``"date >= DATE '2024-01-01'"``.
        out_fields: Comma-separated fields, or ``"*"``.
        page_size: Records per request.
        order_by: ArcGIS ``orderByFields`` value.
        timeout: Per-request timeout in seconds.

    Yields:
        One dict of attributes per feature.

    Raises:
        RuntimeError: If the service returns an error payload.
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

    frame = pd.DataFrame(list(iter_features(layer_url, where=where, page_size=page_size)))
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


def ingest_ports(
    where: str = "1=1",
    layer_url: str = PORTS_LAYER,
    retrieved_at: dt.datetime | None = None,
    store: bool = True,
) -> pd.DataFrame:
    """Pull the daily port activity table and store it as a vintage.

    Returns:
        The normalised frame.
    """
    from ..storage import write_vintage

    frame = normalise_daily_table(fetch_layer(layer_url, where=where), "port")
    if store and not frame.empty:
        write_vintage(frame, "portwatch_ports", retrieved_at=retrieved_at)
    return frame
