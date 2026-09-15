"""Gate-line transit detection.

A transit is two consecutive position reports for one vessel that fall on opposite sides
of a chokepoint's gate line, within ``max_gap_hours`` of each other. The time limit
matters: with a 14-hour gap between fixes there is no evidence the vessel crossed *this*
gate rather than taking a different route, so those pairs are reported separately as
``ambiguous`` rather than counted or silently discarded.

Vessels that loiter on the gate produce a burst of real crossings that are not real
transits. :func:`deduplicate_transits` collapses those, keying on IMO where known so a
mid-voyage MMSI change does not split one vessel into two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from ..config import Zone
from ..logging_utils import get_logger, log_stage

logger = get_logger(__name__)

REQUIRED_POSITION_COLUMNS = ("mmsi", "timestamp", "lat", "lon")


def signed_side(
    gate_start: tuple[float, float], gate_end: tuple[float, float], point: tuple[float, float]
) -> float:
    """Signed area of the triangle (gate_start, gate_end, point).

    The sign says which half-plane the point is in: positive to the left of the directed
    gate, negative to the right, zero exactly on the line.

    Args:
        gate_start: Gate segment start as ``(lon, lat)``.
        gate_end: Gate segment end as ``(lon, lat)``.
        point: Point as ``(lon, lat)``.

    Returns:
        The signed cross product.
    """
    (ax, ay), (bx, by), (px, py) = gate_start, gate_end, point
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _gate_segments(gate: LineString) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    coordinates = list(gate.coords)
    return list(zip(coordinates[:-1], coordinates[1:], strict=True))


def _crossed_segment(
    gate: LineString, previous: tuple[float, float], current: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the gate segment that the movement segment crosses, if any."""
    movement = LineString([previous, current])
    if not movement.intersects(gate):
        return None
    for start, end in _gate_segments(gate):
        if movement.intersects(LineString([start, end])):
            return start, end
    return None


def detect_transits(
    positions: pd.DataFrame,
    zone: Zone,
    max_gap_hours: float = 6.0,
) -> pd.DataFrame:
    """Detect gate crossings for one zone.

    Args:
        positions: Columns ``mmsi``, ``timestamp`` (UTC), ``lat``, ``lon``. Extra columns
            are ignored. Rows need not be sorted.
        zone: The chokepoint definition supplying the gate and direction labels.
        max_gap_hours: Maximum time between the two fixes bracketing a crossing.

    Returns:
        One row per crossing with columns ``mmsi``, ``zone_key``, ``crossing_time``,
        ``direction``, ``gap_hours``, ``crossing_lon``, ``crossing_lat`` and
        ``ambiguous`` (True when the bracketing gap exceeded ``max_gap_hours``).

    Raises:
        KeyError: If a required column is missing.
    """
    missing = [column for column in REQUIRED_POSITION_COLUMNS if column not in positions.columns]
    if missing:
        raise KeyError(f"positions is missing required columns: {missing}")

    if positions.empty:
        return _empty_transits()

    frame = positions.loc[:, list(REQUIRED_POSITION_COLUMNS)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.dropna(subset=["lat", "lon", "timestamp"])
    frame = frame.sort_values(["mmsi", "timestamp"])

    gate = zone.line
    records: list[dict[str, object]] = []

    for mmsi, group in frame.groupby("mmsi", sort=False):
        if len(group) < 2:
            continue
        lons = group["lon"].to_numpy()
        lats = group["lat"].to_numpy()
        times = group["timestamp"].to_numpy()

        for index in range(1, len(group)):
            previous = (float(lons[index - 1]), float(lats[index - 1]))
            current = (float(lons[index]), float(lats[index]))

            segment = _crossed_segment(gate, previous, current)
            if segment is None:
                continue

            start, end = segment
            side_before = signed_side(start, end, previous)
            side_after = signed_side(start, end, current)
            if (
                side_before == 0.0
                or side_after == 0.0
                or np.sign(side_before) == np.sign(side_after)
            ):
                # Touching or running along the gate is not a crossing.
                continue

            gap_hours = float((times[index] - times[index - 1]) / np.timedelta64(1, "h"))
            intersection = LineString([previous, current]).intersection(gate)
            crossing_point = (
                intersection
                if isinstance(intersection, Point)
                else Point(intersection.coords[0])
                if hasattr(intersection, "coords") and len(intersection.coords) > 0
                else Point(current)
            )

            # Interpolate the crossing time along the movement segment.
            travelled = LineString([previous, current]).length
            fraction = (
                LineString([previous, (crossing_point.x, crossing_point.y)]).length / travelled
                if travelled > 0
                else 0.0
            )
            crossing_time = pd.Timestamp(times[index - 1]) + pd.Timedelta(
                hours=gap_hours * float(np.clip(fraction, 0.0, 1.0))
            )

            records.append(
                {
                    "mmsi": mmsi,
                    "zone_key": zone.key,
                    "crossing_time": crossing_time,
                    "direction": zone.direction_label(1 if side_after > 0 else -1),
                    "gap_hours": gap_hours,
                    "crossing_lon": crossing_point.x,
                    "crossing_lat": crossing_point.y,
                    "ambiguous": gap_hours > max_gap_hours,
                }
            )

    transits = pd.DataFrame(records, columns=list(_empty_transits().columns))
    if not transits.empty:
        transits = transits.sort_values("crossing_time").reset_index(drop=True)

    log_stage(
        logger,
        f"transits.detect.{zone.key}",
        transits,
        vessels=int(frame["mmsi"].nunique()),
        ambiguous=int(transits["ambiguous"].sum()) if not transits.empty else 0,
    )
    return transits


def _empty_transits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mmsi": pd.Series(dtype="int64"),
            "zone_key": pd.Series(dtype="object"),
            "crossing_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "direction": pd.Series(dtype="object"),
            "gap_hours": pd.Series(dtype="float64"),
            "crossing_lon": pd.Series(dtype="float64"),
            "crossing_lat": pd.Series(dtype="float64"),
            "ambiguous": pd.Series(dtype="bool"),
        }
    )


def attach_vessel_identity(transits: pd.DataFrame, static: pd.DataFrame | None) -> pd.DataFrame:
    """Add a stable ``vessel_id``, preferring IMO over MMSI.

    MMSI is reassignable and changes with flag; IMO does not. Joining on IMO keeps one
    vessel as one vessel across a re-flagging.

    Args:
        transits: Output of :func:`detect_transits`.
        static: Optional static-data frame with ``mmsi`` and ``imo`` columns.

    Returns:
        ``transits`` with ``imo`` and ``vessel_id`` columns added.
    """
    out = transits.copy()
    if out.empty:
        out["imo"] = pd.Series(dtype="object")
        out["vessel_id"] = pd.Series(dtype="object")
        return out

    if static is not None and not static.empty and {"mmsi", "imo"} <= set(static.columns):
        latest = (
            static.dropna(subset=["imo"])
            .sort_values("timestamp" if "timestamp" in static.columns else "mmsi")
            .drop_duplicates("mmsi", keep="last")
            .loc[:, ["mmsi", "imo"]]
        )
        out = out.merge(latest, on="mmsi", how="left")
    else:
        out["imo"] = pd.NA

    out["vessel_id"] = np.where(
        out["imo"].notna(), "imo:" + out["imo"].astype(str), "mmsi:" + out["mmsi"].astype(str)
    )
    resolved = int(out["imo"].notna().sum())
    logger.info(
        "stage=transits.identity rows=%d imo_resolved=%d imo_missing=%d",
        len(out),
        resolved,
        len(out) - resolved,
    )
    return out


def deduplicate_transits(
    transits: pd.DataFrame,
    min_separation_hours: float = 6.0,
) -> pd.DataFrame:
    """Collapse repeat crossings by the same vessel in the same direction.

    A vessel anchored on the gate can oscillate across it many times in an hour. Only the
    first crossing of each direction within ``min_separation_hours`` is kept.

    Args:
        transits: Frame with ``vessel_id``, ``direction`` and ``crossing_time``.
        min_separation_hours: Minimum spacing between two counted crossings.

    Returns:
        The deduplicated frame.
    """
    if transits.empty:
        return transits

    if "vessel_id" not in transits.columns:
        raise KeyError("call attach_vessel_identity() before deduplicate_transits()")

    out = transits.sort_values(["vessel_id", "direction", "crossing_time"]).copy()
    keep = np.ones(len(out), dtype=bool)
    last_seen: dict[tuple[str, str], pd.Timestamp] = {}

    for position, (vessel, direction, time) in enumerate(
        zip(out["vessel_id"], out["direction"], out["crossing_time"], strict=True)
    ):
        key = (vessel, direction)
        previous = last_seen.get(key)
        if previous is not None and (time - previous) < pd.Timedelta(hours=min_separation_hours):
            keep[position] = False
            continue
        last_seen[key] = time

    deduplicated = out.loc[keep].sort_values("crossing_time").reset_index(drop=True)
    log_stage(
        logger,
        "transits.deduplicate",
        deduplicated,
        removed=len(out) - len(deduplicated),
        min_separation_hours=min_separation_hours,
    )
    return deduplicated


def daily_transit_counts(
    transits: pd.DataFrame,
    include_ambiguous: bool = False,
) -> pd.DataFrame:
    """Aggregate transits to one row per date, zone, and direction.

    Args:
        transits: Deduplicated transits.
        include_ambiguous: Whether to count crossings bracketed by an over-long gap.

    Returns:
        Columns ``date``, ``zone_key``, ``direction``, ``n_transits``, ``n_vessels``.
    """
    if transits.empty:
        return pd.DataFrame(columns=["date", "zone_key", "direction", "n_transits", "n_vessels"])

    frame = transits if include_ambiguous else transits[~transits["ambiguous"]]
    if frame.empty:
        return pd.DataFrame(columns=["date", "zone_key", "direction", "n_transits", "n_vessels"])

    frame = frame.assign(date=pd.to_datetime(frame["crossing_time"], utc=True).dt.date)
    identity = "vessel_id" if "vessel_id" in frame.columns else "mmsi"
    counts = (
        frame.groupby(["date", "zone_key", "direction"], as_index=False)
        .agg(n_transits=(identity, "size"), n_vessels=(identity, "nunique"))
        .sort_values(["date", "zone_key", "direction"])
        .reset_index(drop=True)
    )
    log_stage(logger, "transits.daily_counts", counts, include_ambiguous=include_ambiguous)
    return counts
