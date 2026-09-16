"""Turn collected raw AIS into daily per-zone metrics.

This is the orchestration layer between the collector and the feature table. The
collector writes raw position and static reports; :func:`compute_ais_metrics` runs
transit detection, the waiting fleet, dark gaps, laden classification and the
data-quality flags over them and returns daily tables.

These metrics are the part of the project PortWatch cannot supply. PortWatch gives
transit counts; it does not tell you how many loaded tankers are sitting at anchor
outside a closed strait, or how many went dark inside it. That is the reason for
collecting raw AIS at all.

Everything here is computed from our own collection and is therefore **available
the same day** — unlike PortWatch, which publishes weekly. The publication lag in
``features.py`` applies to PortWatch-derived columns, not these.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..config import Zone, live_zones
from ..logging_utils import get_logger, log_stage
from .transits import (
    attach_vessel_identity,
    daily_transit_counts,
    deduplicate_transits,
    detect_transits,
)
from .vessel_state import (
    classify_laden,
    daily_waiting_fleet,
    find_dark_gaps,
    find_waiting_episodes,
    flag_quality,
)

logger = get_logger(__name__)

# How stale a vessel's last static report may be before its laden state is treated
# as unknown. Draught is hand-entered and often not refreshed between voyages, so a
# month-old reading is not evidence about today's cargo.
MAX_LADEN_STALENESS_DAYS = 30


def ensure_zone_keys(positions: pd.DataFrame, zones: dict[str, Zone]) -> pd.DataFrame:
    """Fill in ``zone_key`` geometrically wherever the collector did not tag it.

    The collector stamps each message with the zone it fell in, but frames read from
    elsewhere (a manual export, another collector, a test) may not carry the column.
    Trusting a missing tag silently attributes every position to ``outside_zones``,
    which splits one zone-day into two rows in the combined table and makes the
    data-quality panel report zero collected positions for a zone whose transit count
    is plainly non-zero.

    Args:
        positions: Position reports with ``lon`` and ``lat``.
        zones: Zones to test membership against.

    Returns:
        A copy with ``zone_key`` present and filled where a polygon contains the point.
    """
    from .vessel_state import _points_in_polygon

    out = positions.copy()
    if "zone_key" not in out.columns:
        out["zone_key"] = pd.Series([None] * len(out), index=out.index, dtype="object")

    missing = out["zone_key"].isna()
    if not missing.any():
        return out

    for key, zone in zones.items():
        still_missing = out["zone_key"].isna()
        if not still_missing.any():
            break
        inside = _points_in_polygon(out["lon"], out["lat"], zone) & still_missing
        out.loc[inside, "zone_key"] = key

    filled = int(missing.sum() - out["zone_key"].isna().sum())
    if filled:
        logger.info(
            "stage=ais_metrics.ensure_zone_keys filled=%d still_untagged=%d",
            filled,
            int(out["zone_key"].isna().sum()),
        )
    return out


def _positions_near_zone(positions: pd.DataFrame, zone: Zone) -> pd.DataFrame:
    """Restrict positions to a zone's bounding box.

    Two reasons, one of them a correctness issue rather than speed: with all zones
    in one frame, consecutive fixes for a vessel seen first in one region and then
    another form a movement segment spanning half the world, which can cross an
    unrelated gate and register a transit that never happened. (The long time gap
    would flag it ``ambiguous``, but not counting it at all is better.)
    """
    min_lon, min_lat, max_lon, max_lat = zone.polygon.bounds
    return positions[
        positions["lon"].between(min_lon, max_lon) & positions["lat"].between(min_lat, max_lat)
    ]


def daily_laden_share(
    positions: pd.DataFrame,
    static: pd.DataFrame,
    zone: Zone,
    threshold: float = 0.75,
    max_staleness_days: int = MAX_LADEN_STALENESS_DAYS,
) -> pd.DataFrame:
    """Share of vessels in a zone each day that appear to be laden.

    Each vessel-day is matched to that vessel's most recent *prior* static report
    using a backward as-of join, so a report filed on the 10th never informs the
    9th. Reports older than ``max_staleness_days`` are not used at all.

    The denominator is vessels whose laden state is *known*, and the unknown count
    is returned alongside. A laden share of 0.8 computed from four vessels out of
    ninety is not the same claim as one computed from eighty, and collapsing them
    into a single number would hide that.

    Args:
        positions: Position reports with ``mmsi``, ``timestamp``, ``lat``, ``lon``.
        static: Static reports with ``mmsi``, ``timestamp``, ``draught`` and
            optionally ``length_m``.
        zone: Zone whose polygon defines presence.
        threshold: Laden ratio at or above which a vessel counts as laden.
        max_staleness_days: Oldest static report still considered informative.

    Returns:
        Columns ``date``, ``zone_key``, ``n_present``, ``n_laden_known``,
        ``n_laden``, ``laden_share``, ``median_draught_age_days``.
    """
    columns = [
        "date",
        "zone_key",
        "n_present",
        "n_laden_known",
        "n_laden",
        "laden_share",
        "median_draught_age_days",
    ]
    if positions.empty:
        return pd.DataFrame(columns=columns)

    from .vessel_state import _points_in_polygon

    inside = positions[_points_in_polygon(positions["lon"], positions["lat"], zone)].copy()
    if inside.empty:
        return pd.DataFrame(columns=columns)

    inside["timestamp"] = pd.to_datetime(inside["timestamp"], utc=True)
    inside["day"] = inside["timestamp"].dt.floor("D")
    presence = (
        inside.loc[:, ["day", "mmsi"]]
        .drop_duplicates()
        .sort_values(["day", "mmsi"])
        .reset_index(drop=True)
    )

    if static is None or static.empty:
        presence_counts = presence.groupby("day", as_index=False)["mmsi"].nunique()
        presence_counts = presence_counts.rename(columns={"mmsi": "n_present"})
        presence_counts["date"] = presence_counts["day"].dt.date
        presence_counts["zone_key"] = zone.key
        presence_counts["n_laden_known"] = 0
        presence_counts["n_laden"] = 0
        presence_counts["laden_share"] = np.nan
        presence_counts["median_draught_age_days"] = np.nan
        logger.warning(
            "no static reports available for %s; laden share is undefined for every day",
            zone.key,
        )
        return presence_counts.loc[:, columns]

    classified = classify_laden(static, threshold=threshold)
    classified = classified.dropna(subset=["laden_ratio"]).copy()
    classified["timestamp"] = pd.to_datetime(classified["timestamp"], utc=True)
    classified = classified.sort_values("timestamp")

    if classified.empty:
        matched = presence.assign(is_laden=None, report_time=pd.NaT)
    else:
        matched = pd.merge_asof(
            presence.sort_values("day"),
            classified.loc[:, ["timestamp", "mmsi", "is_laden"]].rename(
                columns={"timestamp": "report_time"}
            ),
            left_on="day",
            right_on="report_time",
            by="mmsi",
            direction="backward",
            tolerance=pd.Timedelta(days=max_staleness_days),
        )

    matched["draught_age_days"] = (matched["day"] - matched["report_time"]).dt.days
    known = matched["is_laden"].notna()

    daily = (
        matched.assign(
            _known=known.astype(int),
            _laden=(known & (matched["is_laden"] == True)).astype(int),  # noqa: E712
        )
        .groupby("day", as_index=False)
        .agg(
            n_present=("mmsi", "nunique"),
            n_laden_known=("_known", "sum"),
            n_laden=("_laden", "sum"),
            median_draught_age_days=("draught_age_days", "median"),
        )
    )
    daily["laden_share"] = np.where(
        daily["n_laden_known"] > 0, daily["n_laden"] / daily["n_laden_known"], np.nan
    )
    daily["date"] = daily["day"].dt.date
    daily["zone_key"] = zone.key

    coverage = (
        daily["n_laden_known"].sum() / daily["n_present"].sum() if daily["n_present"].sum() else 0.0
    )
    log_stage(logger, f"ais_metrics.laden.{zone.key}", daily, known_coverage=f"{coverage:.1%}")
    if coverage < 0.25:
        logger.warning(
            "laden state is known for only %.0f%% of vessel-days in %s; treat laden_share "
            "as indicative at best",
            100 * coverage,
            zone.key,
        )
    return daily.loc[:, columns]


def build_quality_daily(
    flagged: pd.DataFrame,
    dark_gaps: pd.DataFrame,
) -> pd.DataFrame:
    """Daily data-quality and collector-health summary.

    ``hours_with_data`` is the collector-uptime proxy: a day with 24 is healthy, a
    day with 3 means the collector was down or the feed was empty for most of it.
    That distinction matters because a gap in collection and a genuine drop in
    traffic look identical in a transit count.

    Args:
        flagged: Output of :func:`tanker_tape.process.vessel_state.flag_quality`.
        dark_gaps: Concatenated output of
            :func:`tanker_tape.process.vessel_state.find_dark_gaps`.

    Returns:
        One row per date and zone.
    """
    columns = [
        "date",
        "zone_key",
        "n_positions",
        "n_vessels",
        "hours_with_data",
        "n_impossible_speed",
        "n_shared_position",
        "n_on_land",
        "n_flagged_any",
        "n_dark_gaps",
    ]
    if flagged.empty:
        return pd.DataFrame(columns=columns)

    frame = flagged.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["date"] = frame["timestamp"].dt.date
    # Messages arriving from inside the subscribed box but outside any zone polygon
    # are real data and must stay visible rather than being dropped.
    frame["zone_key"] = frame.get("zone_key", pd.Series(index=frame.index, dtype="object"))
    frame["zone_key"] = frame["zone_key"].fillna("outside_zones")

    daily = (
        frame.groupby(["date", "zone_key"], as_index=False)
        .agg(
            n_positions=("mmsi", "size"),
            n_vessels=("mmsi", "nunique"),
            hours_with_data=("timestamp", lambda values: values.dt.floor("h").nunique()),
            n_impossible_speed=("flag_impossible_speed", "sum"),
            n_shared_position=("flag_shared_position", "sum"),
            n_on_land=("flag_on_land", "sum"),
            n_flagged_any=("flag_any", "sum"),
        )
        .sort_values(["date", "zone_key"])
    )

    if dark_gaps is not None and not dark_gaps.empty:
        gaps = dark_gaps.copy()
        gaps["date"] = pd.to_datetime(gaps["gap_start"], utc=True).dt.date
        counts = gaps.groupby(["date", "zone_key"], as_index=False).size()
        counts = counts.rename(columns={"size": "n_dark_gaps"})
        daily = daily.merge(counts, on=["date", "zone_key"], how="left")
    else:
        daily["n_dark_gaps"] = 0
    daily["n_dark_gaps"] = daily["n_dark_gaps"].fillna(0).astype(int)

    for column in ("n_impossible_speed", "n_shared_position", "n_on_land", "n_flagged_any"):
        daily[column] = daily[column].astype(int)

    daily = daily.loc[:, columns].reset_index(drop=True)

    low_uptime = daily[daily["hours_with_data"] < 20]
    if not low_uptime.empty:
        logger.warning(
            "%d zone-day(s) have fewer than 20 hours of collected data; transit counts for "
            "those days are undercounts, not low traffic. Worst: %s",
            len(low_uptime),
            low_uptime.nsmallest(1, "hours_with_data")[
                ["date", "zone_key", "hours_with_data"]
            ].to_dict("records"),
        )

    log_stage(logger, "ais_metrics.quality_daily", daily)
    return daily


def compute_ais_metrics(
    positions: pd.DataFrame,
    static: pd.DataFrame | None = None,
    zones: dict[str, Zone] | None = None,
    laden_threshold: float = 0.75,
    max_gap_hours: float = 6.0,
    waiting_max_speed: float = 1.0,
    waiting_min_hours: float = 6.0,
    dark_gap_hours: float = 12.0,
) -> dict[str, pd.DataFrame]:
    """Run the full raw-AIS pipeline and return the daily tables.

    Pure with respect to storage: it takes frames and returns frames, so it can be
    tested without touching the data lake.

    Args:
        positions: Raw position reports.
        static: Raw static reports; laden metrics are skipped without them.
        zones: Zones to process; defaults to those flagged ``collect_live``.
        laden_threshold: Laden ratio cutoff.
        max_gap_hours: Maximum gap bracketing a counted transit.
        waiting_max_speed: Speed at or below which a vessel counts as stopped.
        waiting_min_hours: Minimum loiter duration to count as waiting.
        dark_gap_hours: Minimum in-zone AIS silence to call a vessel dark.

    Returns:
        Mapping with ``transits_daily``, ``waiting_daily``, ``laden_daily``,
        ``quality_daily`` and ``ais_daily`` (the first three joined on date/zone).
    """
    zones = zones if zones is not None else live_zones()
    if positions.empty:
        logger.warning("no positions supplied; every AIS metric will be empty")
        empty = pd.DataFrame()
        return {
            "transits_daily": empty,
            "waiting_daily": empty,
            "laden_daily": empty,
            "quality_daily": empty,
            "ais_daily": empty,
        }

    positions = positions.copy()
    positions["timestamp"] = pd.to_datetime(positions["timestamp"], utc=True)
    positions = ensure_zone_keys(positions, zones)

    flagged = flag_quality(positions)
    # Flagged rows are excluded from the *counts* but never from the quality
    # report, so a day of heavy spoofing shows up as a low count next to a high
    # flag total rather than as a quiet day.
    clean = flagged[~flagged["flag_any"]]
    logger.info(
        "stage=ais_metrics.clean kept=%d flagged_out=%d", len(clean), len(flagged) - len(clean)
    )

    transit_frames, waiting_frames, laden_frames, gap_frames = [], [], [], []

    for key, zone in zones.items():
        near = _positions_near_zone(clean, zone)
        if near.empty:
            logger.warning("no positions inside the bounding box for zone %r", key)
            continue

        transits = detect_transits(near, zone, max_gap_hours=max_gap_hours)
        transits = attach_vessel_identity(transits, static)
        transits = deduplicate_transits(transits, min_separation_hours=max_gap_hours)
        counts = daily_transit_counts(transits)
        if not counts.empty:
            transit_frames.append(counts)

        episodes = find_waiting_episodes(
            near, zone, max_speed_knots=waiting_max_speed, min_hours=waiting_min_hours
        )
        waiting = daily_waiting_fleet(episodes)
        if not waiting.empty:
            waiting_frames.append(waiting)

        laden = daily_laden_share(near, static, zone, threshold=laden_threshold)
        if not laden.empty:
            laden_frames.append(laden)

        gaps = find_dark_gaps(near, zone, min_gap_hours=dark_gap_hours)
        if not gaps.empty:
            gap_frames.append(gaps)

    transits_daily = (
        pd.concat(transit_frames, ignore_index=True) if transit_frames else pd.DataFrame()
    )
    waiting_daily = (
        pd.concat(waiting_frames, ignore_index=True) if waiting_frames else pd.DataFrame()
    )
    laden_daily = pd.concat(laden_frames, ignore_index=True) if laden_frames else pd.DataFrame()
    dark_gaps = pd.concat(gap_frames, ignore_index=True) if gap_frames else pd.DataFrame()
    quality_daily = build_quality_daily(flagged, dark_gaps)

    ais_daily = _combine(transits_daily, waiting_daily, laden_daily, quality_daily)

    log_stage(
        logger,
        "ais_metrics.compute",
        ais_daily,
        zones=len(zones),
        transits=int(transits_daily["n_transits"].sum()) if not transits_daily.empty else 0,
    )
    return {
        "transits_daily": transits_daily,
        "waiting_daily": waiting_daily,
        "laden_daily": laden_daily,
        "quality_daily": quality_daily,
        "ais_daily": ais_daily,
    }


def _combine(
    transits_daily: pd.DataFrame,
    waiting_daily: pd.DataFrame,
    laden_daily: pd.DataFrame,
    quality_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Join the per-zone daily tables into one row per date and zone."""
    if not transits_daily.empty:
        # Directions become columns so a closure shows as inbound and outbound
        # separately; a strait can stop admitting traffic while still emptying.
        wide = transits_daily.pivot_table(
            index=["date", "zone_key"],
            columns="direction",
            values="n_transits",
            aggfunc="sum",
            fill_value=0,
        )
        wide.columns = [f"n_transits_{direction}" for direction in wide.columns]
        combined = wide.reset_index()
        combined["n_transits"] = combined.filter(like="n_transits_").sum(axis=1)
    else:
        combined = pd.DataFrame(columns=["date", "zone_key", "n_transits"])

    for frame in (waiting_daily, laden_daily, quality_daily):
        if frame is None or frame.empty:
            continue
        if combined.empty:
            combined = frame.copy()
            continue
        overlap = [
            column
            for column in frame.columns
            if column in combined.columns and column not in ("date", "zone_key")
        ]
        combined = combined.merge(frame.drop(columns=overlap), on=["date", "zone_key"], how="outer")

    if combined.empty:
        return combined
    return combined.sort_values(["date", "zone_key"]).reset_index(drop=True)


def build_ais_metrics(
    start: dt.date | None = None,
    end: dt.date | None = None,
    store: bool = True,
    **kwargs: object,
) -> dict[str, pd.DataFrame]:
    """Read collected AIS from the data lake, compute daily metrics, and store them.

    Args:
        start: Inclusive first collection date to read.
        end: Inclusive last collection date to read.
        store: Whether to write the processed tables.
        **kwargs: Passed through to :func:`compute_ais_metrics`.

    Returns:
        The same mapping :func:`compute_ais_metrics` returns.
    """
    from ..storage import read_partitions, write_processed

    positions = read_partitions("ais_positions", start=start, end=end)
    static = read_partitions("ais_static", start=start, end=end)

    if positions.empty:
        logger.error(
            "no collected AIS positions between %s and %s. Is the collector running? "
            "See deploy/README.md - there is no backfill for this source.",
            start,
            end,
        )

    tables = compute_ais_metrics(positions, static, **kwargs)  # type: ignore[arg-type]

    if store:
        for name, frame in tables.items():
            if frame is not None and not frame.empty:
                write_processed(frame, f"ais_{name}" if not name.startswith("ais_") else name)
    return tables
