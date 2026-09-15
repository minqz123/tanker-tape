"""Live AIS collector for aisstream.io.

Protocol notes confirmed against the published AsyncAPI spec (see CLAUDE.md):

* Single websocket at ``wss://stream.aisstream.io/v0/stream``.
* A JSON subscription must be sent **within 3 seconds** of the connection opening or the
  server closes it.
* Bounding boxes are ``[[lat, lon], [lat, lon]]`` — the reverse of this project's internal
  ``(lon, lat)`` ordering. The flip lives in :meth:`tanker_tape.config.Zone.aisstream_bounding_box`.
* Messages arrive as ``{MessageType, MetaData, Message: {<MessageType>: {...}}}``.

Operational notes:

* The provider drops messages for slow consumers, so the socket reader does nothing but
  decode and enqueue. Parquet writing happens in a separate task. A full queue drops the
  *oldest* buffered rows and increments a counter that is logged — dropping is visible,
  never silent.
* Reconnects use exponential backoff with jitter.
* There is no history: every day not collected is gone. Run this under systemd/cron from
  day one.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import random
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..config import Zone, get_settings, live_zones
from ..logging_utils import get_logger

logger = get_logger(__name__)

AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"
SUBSCRIBE_DEADLINE_SECONDS = 3.0

# ITU-R M.1371 ship type codes.
TANKER_TYPE_RANGE = range(80, 90)
CARGO_TYPE_RANGE = range(70, 80)

POSITION_MESSAGE_TYPES = ("PositionReport", "StandardClassBPositionReport")
STATIC_MESSAGE_TYPES = ("ShipStaticData", "StaticDataReport")


def ship_type_category(ship_type: int | None) -> str:
    """Map an ITU-R M.1371 ship type code to a coarse category."""
    if ship_type is None:
        return "unknown"
    if ship_type in TANKER_TYPE_RANGE:
        return "tanker"
    if ship_type in CARGO_TYPE_RANGE:
        return "cargo"
    return "other"


def zone_for_point(lon: float, lat: float, zones: dict[str, Zone]) -> str | None:
    """Return the key of the first zone whose polygon contains the point.

    Bounding-box subscriptions are rectangles, so a message can arrive from inside the
    box but outside the actual zone polygon. Those are tagged ``None`` and kept.
    """
    from shapely.geometry import Point

    point = Point(lon, lat)
    for key, zone in zones.items():
        if zone.polygon.contains(point):
            return key
    return None


def _parse_timestamp(metadata: dict[str, Any]) -> pd.Timestamp:
    """Parse the message's UTC timestamp, falling back to arrival time."""
    raw = metadata.get("time_utc")
    if raw:
        # aisstream sends e.g. "2026-09-15 07:21:33.123456789 +0000 UTC"; pandas parses
        # the prefix once the Go timezone suffix is removed.
        cleaned = str(raw).replace(" +0000 UTC", "").strip()
        parsed = pd.to_datetime(cleaned, utc=True, errors="coerce")
        if pd.notna(parsed):
            return parsed
    return pd.Timestamp.now(tz="UTC")


def parse_position(envelope: dict[str, Any], zones: dict[str, Zone]) -> dict[str, Any] | None:
    """Decode a position-report envelope into a flat row.

    Returns:
        A row dict, or ``None`` if the envelope has no usable position.
    """
    message_type = envelope.get("MessageType")
    body = (envelope.get("Message") or {}).get(message_type) or {}
    metadata = envelope.get("MetaData") or {}

    lat = body.get("Latitude", metadata.get("latitude"))
    lon = body.get("Longitude", metadata.get("longitude"))
    mmsi = body.get("UserID") or metadata.get("MMSI")
    if lat is None or lon is None or mmsi is None:
        return None

    return {
        "mmsi": int(mmsi),
        "timestamp": _parse_timestamp(metadata),
        "lat": float(lat),
        "lon": float(lon),
        "sog": body.get("Sog"),
        "cog": body.get("Cog"),
        "heading": body.get("TrueHeading"),
        "nav_status": body.get("NavigationalStatus"),
        "message_type": message_type,
        "zone_key": zone_for_point(float(lon), float(lat), zones),
        "ship_name": (metadata.get("ShipName") or "").strip() or None,
    }


def parse_static(envelope: dict[str, Any]) -> dict[str, Any] | None:
    """Decode a static-data envelope into a flat row.

    ``MaximumStaticDraught`` is manually entered by the crew and frequently stale; it is
    carried through with its observation time so downstream code can age it rather than
    trusting it.
    """
    message_type = envelope.get("MessageType")
    body = (envelope.get("Message") or {}).get(message_type) or {}
    metadata = envelope.get("MetaData") or {}

    mmsi = body.get("UserID") or metadata.get("MMSI")
    if mmsi is None:
        return None

    dimension = body.get("Dimension") or {}
    bow, stern = dimension.get("A"), dimension.get("B")
    length = (
        (bow + stern) if isinstance(bow, int | float) and isinstance(stern, int | float) else None
    )
    ship_type = body.get("Type")

    return {
        "mmsi": int(mmsi),
        "timestamp": _parse_timestamp(metadata),
        "imo": body.get("ImoNumber") or None,
        "name": (body.get("Name") or "").strip() or None,
        "call_sign": (body.get("CallSign") or "").strip() or None,
        "ship_type": ship_type,
        "ship_category": ship_type_category(ship_type),
        "draught": body.get("MaximumStaticDraught"),
        "length_m": length,
        "destination": (body.get("Destination") or "").strip() or None,
        "message_type": message_type,
    }


@dataclass
class CollectorStats:
    """Running counters for the collector, logged on every flush."""

    received: int = 0
    positions: int = 0
    static: int = 0
    unparsed: int = 0
    dropped_queue_full: int = 0
    reconnects: int = 0
    flushes: int = 0
    connected_since: dt.datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "received": self.received,
            "positions": self.positions,
            "static": self.static,
            "unparsed": self.unparsed,
            "dropped_queue_full": self.dropped_queue_full,
            "reconnects": self.reconnects,
            "flushes": self.flushes,
        }


@dataclass
class AisCollector:
    """Long-running aisstream collector writing hourly Parquet partitions.

    Args:
        zones: Zones to subscribe to; defaults to those flagged ``collect_live`` in config.
        api_key: aisstream key; defaults to ``AISSTREAM_API_KEY``.
        flush_seconds: Maximum time between Parquet flushes.
        max_queue: Queue depth before the oldest rows are dropped (and counted).
        max_backoff_seconds: Cap on reconnect backoff.
    """

    zones: dict[str, Zone] = field(default_factory=live_zones)
    api_key: str | None = None
    flush_seconds: float = 300.0
    max_queue: int = 200_000
    max_backoff_seconds: float = 60.0

    stats: CollectorStats = field(default_factory=CollectorStats)
    _positions: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _static: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def subscription_message(self) -> dict[str, Any]:
        """Build the JSON subscription payload.

        Raises:
            RuntimeError: If no zones are flagged for live collection.
        """
        if not self.zones:
            raise RuntimeError(
                "no zones have collect_live: true in config/zones.yaml; nothing to subscribe to"
            )
        key = self.api_key or get_settings().require("aisstream_api_key")
        return {
            "APIKey": key,
            "BoundingBoxes": [zone.aisstream_bounding_box() for zone in self.zones.values()],
            "FilterMessageTypes": list(POSITION_MESSAGE_TYPES + STATIC_MESSAGE_TYPES),
        }

    def handle_envelope(self, envelope: dict[str, Any]) -> None:
        """Route one decoded envelope into the position or static buffer."""
        self.stats.received += 1
        message_type = envelope.get("MessageType")

        if message_type in POSITION_MESSAGE_TYPES:
            row = parse_position(envelope, self.zones)
            if row is None:
                self.stats.unparsed += 1
                return
            self._positions.append(row)
            self.stats.positions += 1
        elif message_type in STATIC_MESSAGE_TYPES:
            row = parse_static(envelope)
            if row is None:
                self.stats.unparsed += 1
                return
            self._static.append(row)
            self.stats.static += 1
        else:
            self.stats.unparsed += 1

        self._enforce_buffer_cap()

    def _enforce_buffer_cap(self) -> None:
        """Drop the oldest buffered rows if the buffer exceeds ``max_queue``.

        Dropping is counted and logged; it is never silent.
        """
        for buffer_name, buffer in (("positions", self._positions), ("static", self._static)):
            overflow = len(buffer) - self.max_queue
            if overflow > 0:
                del buffer[:overflow]
                self.stats.dropped_queue_full += overflow
                logger.error(
                    "collector buffer %s overflowed; dropped %d oldest rows "
                    "(total dropped=%d). Lower flush_seconds or raise max_queue.",
                    buffer_name,
                    overflow,
                    self.stats.dropped_queue_full,
                )

    def flush(self, now: dt.datetime | None = None) -> dict[str, int]:
        """Write buffered rows to hourly Parquet partitions and clear the buffers.

        Returns:
            Rows written per dataset.
        """
        from ..storage import append_partition

        stamp = now or dt.datetime.now(dt.UTC)
        partition = f"{stamp:%Y-%m-%d}/{stamp:%H}"
        written = {"positions": 0, "static": 0}

        if self._positions:
            frame = pd.DataFrame(self._positions)
            append_partition(frame, "ais_positions", partition)
            written["positions"] = len(frame)
            self._positions = []

        if self._static:
            frame = pd.DataFrame(self._static)
            append_partition(frame, "ais_static", partition)
            written["static"] = len(frame)
            self._static = []

        self.stats.flushes += 1
        logger.info(
            "stage=aisstream.flush partition=%s positions=%d static=%d %s",
            partition,
            written["positions"],
            written["static"],
            " ".join(f"{key}={value}" for key, value in self.stats.as_dict().items()),
        )
        return written

    async def _read_forever(self, websocket: Any) -> None:
        """Decode messages as fast as they arrive; do no other work here."""
        async for raw in websocket:
            try:
                envelope = json.loads(raw)
            except (TypeError, ValueError):
                self.stats.unparsed += 1
                continue
            if envelope.get("error"):
                raise RuntimeError(f"aisstream returned an error: {envelope['error']}")
            self.handle_envelope(envelope)

    async def _flush_forever(self) -> None:
        """Flush on a fixed cadence for as long as the collector runs."""
        while True:
            await asyncio.sleep(self.flush_seconds)
            self.flush()

    async def run(self, max_reconnects: int | None = None) -> None:
        """Connect, subscribe, and stream until cancelled.

        Args:
            max_reconnects: Stop after this many reconnect attempts (tests use a small
                value); ``None`` reconnects forever.
        """
        import websockets

        subscription = self.subscription_message()
        redacted = {**subscription, "APIKey": "***"}
        logger.info("stage=aisstream.subscribe payload=%s", json.dumps(redacted))

        attempt = 0
        flusher = asyncio.create_task(self._flush_forever())
        try:
            while max_reconnects is None or attempt <= max_reconnects:
                try:
                    async with websockets.connect(
                        AISSTREAM_URL,
                        # permessage-deflate is negotiated by default; keepalive pings
                        # surface a dead link instead of hanging forever.
                        ping_interval=20,
                        ping_timeout=20,
                        max_queue=None,
                    ) as websocket:
                        await asyncio.wait_for(
                            websocket.send(json.dumps(subscription)),
                            timeout=SUBSCRIBE_DEADLINE_SECONDS,
                        )
                        self.stats.connected_since = dt.datetime.now(dt.UTC)
                        attempt = 0
                        logger.info("stage=aisstream.connected zones=%s", ",".join(self.zones))
                        await self._read_forever(websocket)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - reconnect on anything transient
                    self.stats.reconnects += 1
                    backoff = min(2.0**attempt, self.max_backoff_seconds)
                    backoff *= 0.5 + random.random() / 2.0  # jitter to avoid lockstep retries
                    logger.warning(
                        "aisstream connection lost (%s); reconnecting in %.1fs (attempt %d)",
                        exc,
                        backoff,
                        attempt + 1,
                    )
                    self.flush()
                    attempt += 1
                    await asyncio.sleep(backoff)
        finally:
            flusher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await flusher
            self.flush()


async def collect(max_reconnects: int | None = None) -> None:
    """Entry point used by the CLI."""
    await AisCollector().run(max_reconnects=max_reconnects)
