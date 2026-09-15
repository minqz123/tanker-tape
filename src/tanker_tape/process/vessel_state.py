"""Vessel state derivation: laden/ballast, waiting fleet, dark gaps, and quality flags.

Everything here **flags** rather than drops. A spoofed position and a missing position are
different facts about the world, and during a chokepoint closure the spoofing rate is
itself a signal. Counts for every flag are logged daily.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ..config import Zone
from ..logging_utils import get_logger, log_quality, log_stage

logger = get_logger(__name__)

EARTH_RADIUS_NM = 3440.065

# Implied speeds above this are physically impossible for a merchant vessel and indicate
# a position jump: bad decode, spoofing, or two vessels sharing an MMSI.
MAX_PLAUSIBLE_SPEED_KNOTS = 40.0

# Navigational status codes (ITU-R M.1371) that mean "stopped on purpose".
ANCHORED_NAV_STATUSES = (1, 5)  # 1 = at anchor, 5 = moored

# Class-typical maximum draught in metres, keyed by overall length band. Used only when a
# vessel has no observed draught history of its own. Approximate by design.
CLASS_TYPICAL_MAX_DRAUGHT = (
    (120.0, 7.0),  # small products / coastal
    (180.0, 12.2),  # MR
    (250.0, 17.0),  # Aframax / Suezmax
    (330.0, 22.0),  # VLCC
    (float("inf"), 24.5),  # ULCC
)


def haversine_nm(
    lon1: np.ndarray | float,
    lat1: np.ndarray | float,
    lon2: np.ndarray | float,
    lat2: np.ndarray | float,
) -> np.ndarray:
    """Great-circle distance in nautical miles."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_implied_speed(positions: pd.DataFrame) -> pd.DataFrame:
    """Add distance/elapsed/implied-speed columns between consecutive fixes per vessel.

    Args:
        positions: Columns ``mmsi``, ``timestamp``, ``lat``, ``lon``.

    Returns:
        A sorted copy with ``step_nm``, ``step_hours`` and ``implied_speed_kn``.
    """
    frame = positions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    previous_lon = frame.groupby("mmsi")["lon"].shift(1)
    previous_lat = frame.groupby("mmsi")["lat"].shift(1)
    previous_time = frame.groupby("mmsi")["timestamp"].shift(1)

    frame["step_nm"] = haversine_nm(previous_lon, previous_lat, frame["lon"], frame["lat"])
    frame["step_hours"] = (frame["timestamp"] - previous_time).dt.total_seconds() / 3600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["implied_speed_kn"] = frame["step_nm"] / frame["step_hours"].replace(0.0, np.nan)
    return frame


def flag_quality(
    positions: pd.DataFrame,
    max_speed_knots: float = MAX_PLAUSIBLE_SPEED_KNOTS,
    spoof_precision: int = 4,
    spoof_min_vessels: int = 5,
    on_land: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    """Attach boolean data-quality flags to position reports.

    Flags added:

    * ``flag_impossible_speed`` — implied speed between fixes exceeds ``max_speed_knots``.
    * ``flag_shared_position`` — the rounded position is reported by at least
      ``spoof_min_vessels`` distinct MMSIs at the same minute, the classic GPS-jamming /
      spoofing-cluster signature around Hormuz.
    * ``flag_on_land`` — set by the optional ``on_land`` predicate; all-False without one.
    * ``flag_any`` — logical OR of the above.

    Args:
        positions: Columns ``mmsi``, ``timestamp``, ``lat``, ``lon``.
        max_speed_knots: Impossible-speed threshold.
        spoof_precision: Decimal places used when rounding positions into clusters.
        spoof_min_vessels: Distinct MMSIs at one rounded position/minute to call it a cluster.
        on_land: Optional callable taking the frame and returning a boolean Series. Kept
            injectable so the core package does not depend on a land-polygon dataset.

    Returns:
        A copy of ``positions`` with the flag columns and the speed columns added.
    """
    frame = add_implied_speed(positions)

    frame["flag_impossible_speed"] = frame["implied_speed_kn"].notna() & (
        frame["implied_speed_kn"] > max_speed_knots
    )

    minute = frame["timestamp"].dt.floor("min")
    cluster_key = pd.Series(
        list(
            zip(
                minute,
                frame["lat"].round(spoof_precision),
                frame["lon"].round(spoof_precision),
                strict=True,
            )
        ),
        index=frame.index,
    )
    distinct_vessels = frame.groupby(cluster_key)["mmsi"].transform("nunique")
    frame["flag_shared_position"] = distinct_vessels >= spoof_min_vessels

    frame["flag_on_land"] = (
        on_land(frame).astype(bool) if on_land is not None else pd.Series(False, index=frame.index)
    )

    flag_columns = [column for column in frame.columns if column.startswith("flag_")]
    frame["flag_any"] = frame[flag_columns].any(axis=1)

    log_quality(logger, "vessel_state.flag_quality", frame)
    return frame


def find_dark_gaps(
    positions: pd.DataFrame,
    zone: Zone,
    min_gap_hours: float = 12.0,
) -> pd.DataFrame:
    """Find AIS silences inside a zone — vessels that "went dark".

    A gap only counts when the fixes on *both* sides of it are inside the zone polygon; a
    vessel that simply left the monitored box has not gone dark.

    Args:
        positions: Columns ``mmsi``, ``timestamp``, ``lat``, ``lon``.
        zone: The zone whose polygon bounds the test.
        min_gap_hours: Minimum silence to report.

    Returns:
        Columns ``mmsi``, ``zone_key``, ``gap_start``, ``gap_end``, ``gap_hours``,
        ``start_lon``, ``start_lat``.
    """
    frame = add_implied_speed(positions)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "mmsi",
                "zone_key",
                "gap_start",
                "gap_end",
                "gap_hours",
                "start_lon",
                "start_lat",
            ]
        )

    inside = _points_in_polygon(frame["lon"], frame["lat"], zone)
    previous_inside = pd.Series(inside, index=frame.index).groupby(frame["mmsi"]).shift(1)
    previous_lon = frame.groupby("mmsi")["lon"].shift(1)
    previous_lat = frame.groupby("mmsi")["lat"].shift(1)
    previous_time = frame.groupby("mmsi")["timestamp"].shift(1)

    is_gap = (
        (frame["step_hours"] >= min_gap_hours) & inside & previous_inside.fillna(False).astype(bool)
    )

    gaps = pd.DataFrame(
        {
            "mmsi": frame.loc[is_gap, "mmsi"],
            "zone_key": zone.key,
            "gap_start": previous_time[is_gap],
            "gap_end": frame.loc[is_gap, "timestamp"],
            "gap_hours": frame.loc[is_gap, "step_hours"],
            "start_lon": previous_lon[is_gap],
            "start_lat": previous_lat[is_gap],
        }
    ).reset_index(drop=True)

    log_stage(logger, f"vessel_state.dark_gaps.{zone.key}", gaps, min_gap_hours=min_gap_hours)
    return gaps


def _points_in_polygon(lon: pd.Series, lat: pd.Series, zone: Zone) -> pd.Series:
    """Vectorised point-in-polygon test, falling back to shapely objects if needed."""
    try:
        from shapely import contains_xy

        return pd.Series(contains_xy(zone.polygon, lon.to_numpy(), lat.to_numpy()), index=lon.index)
    except ImportError:  # pragma: no cover - shapely < 2.0 only
        from shapely.geometry import Point

        return pd.Series(
            [zone.polygon.contains(Point(x, y)) for x, y in zip(lon, lat, strict=True)],
            index=lon.index,
        )


def find_waiting_episodes(
    positions: pd.DataFrame,
    zone: Zone,
    max_speed_knots: float = 1.0,
    min_hours: float = 6.0,
) -> pd.DataFrame:
    """Identify vessels loitering inside a zone — the "waiting fleet".

    During a closure this is likely the most informative live metric: ships that cannot
    transit pile up rather than disappear.

    Args:
        positions: Columns ``mmsi``, ``timestamp``, ``lat``, ``lon``, ``sog``, and
            optionally ``nav_status``.
        zone: Zone whose polygon bounds the search.
        max_speed_knots: Speed at or below which a vessel counts as stopped.
        min_hours: Minimum episode duration to report.

    Returns:
        Columns ``mmsi``, ``zone_key``, ``start``, ``end``, ``hours``, ``state``
        (``anchored`` or ``drifting``), ``n_fixes``.
    """
    columns = ["mmsi", "zone_key", "start", "end", "hours", "state", "n_fixes"]
    if positions.empty or "sog" not in positions.columns:
        return pd.DataFrame(columns=columns)

    frame = positions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    frame["_inside"] = _points_in_polygon(frame["lon"], frame["lat"], zone)
    frame["_stopped"] = frame["_inside"] & (frame["sog"].fillna(99.0) <= max_speed_knots)

    # Label runs of consecutive stopped fixes within each vessel.
    run_break = frame["_stopped"].ne(frame.groupby("mmsi")["_stopped"].shift(1))
    frame["_run"] = run_break.groupby(frame["mmsi"]).cumsum()

    stopped = frame[frame["_stopped"]]
    if stopped.empty:
        return pd.DataFrame(columns=columns)

    grouped = stopped.groupby(["mmsi", "_run"], as_index=False).agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        n_fixes=("timestamp", "size"),
        anchored_fixes=(
            "nav_status",
            lambda values: int(values.isin(ANCHORED_NAV_STATUSES).sum()),
        )
        if "nav_status" in stopped.columns
        else ("timestamp", "size"),
    )
    grouped["hours"] = (grouped["end"] - grouped["start"]).dt.total_seconds() / 3600.0
    episodes = grouped[grouped["hours"] >= min_hours].copy()
    if episodes.empty:
        return pd.DataFrame(columns=columns)

    if "nav_status" in stopped.columns:
        # Majority vote across the episode: a single stale status should not decide it.
        episodes["state"] = np.where(
            episodes["anchored_fixes"] >= episodes["n_fixes"] / 2.0, "anchored", "drifting"
        )
    else:
        episodes["state"] = "unknown"

    episodes["zone_key"] = zone.key
    episodes = episodes.loc[:, columns].sort_values("start").reset_index(drop=True)
    log_stage(
        logger,
        f"vessel_state.waiting.{zone.key}",
        episodes,
        anchored=int((episodes["state"] == "anchored").sum()),
        drifting=int((episodes["state"] == "drifting").sum()),
    )
    return episodes


def daily_waiting_fleet(episodes: pd.DataFrame) -> pd.DataFrame:
    """Expand waiting episodes into a daily count of vessels waiting.

    An episode spanning three days contributes to all three.

    Returns:
        Columns ``date``, ``zone_key``, ``n_waiting``, ``n_anchored``, ``n_drifting``.
    """
    columns = ["date", "zone_key", "n_waiting", "n_anchored", "n_drifting"]
    if episodes.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for episode in episodes.itertuples(index=False):
        days = pd.date_range(
            pd.Timestamp(episode.start).normalize(),
            pd.Timestamp(episode.end).normalize(),
            freq="D",
            tz="UTC",
        )
        rows.extend(
            {
                "date": day.date(),
                "zone_key": episode.zone_key,
                "mmsi": episode.mmsi,
                "state": episode.state,
            }
            for day in days
        )

    expanded = pd.DataFrame(rows).drop_duplicates(subset=["date", "zone_key", "mmsi"])
    daily = (
        expanded.groupby(["date", "zone_key"], as_index=False)
        .agg(
            n_waiting=("mmsi", "nunique"),
            n_anchored=("state", lambda values: int((values == "anchored").sum())),
            n_drifting=("state", lambda values: int((values == "drifting").sum())),
        )
        .sort_values(["date", "zone_key"])
        .reset_index(drop=True)
    )
    log_stage(logger, "vessel_state.daily_waiting", daily)
    return daily


def class_typical_max_draught(length_m: float | None) -> float | None:
    """Look up a class-typical maximum draught for a vessel length, in metres."""
    if length_m is None or not np.isfinite(length_m) or length_m <= 0:
        return None
    for max_length, draught in CLASS_TYPICAL_MAX_DRAUGHT:
        if length_m < max_length:
            return draught
    return None


def classify_laden(
    static: pd.DataFrame,
    threshold: float = 0.75,
    reference_draughts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify reports as laden or ballast from the reported draught.

    ``laden_ratio = reported_draught / max_draught_for_that_vessel``. The denominator is
    the vessel's own observed maximum where available, otherwise a class-typical value
    from its length. Reported draught is entered by hand and often not updated between
    voyages, so ``draught_age_hours`` is carried alongside: a confident-looking ratio
    computed from a three-week-old draught is not evidence.

    Args:
        static: Columns ``mmsi``, ``timestamp``, ``draught``, and optionally ``length_m``.
        threshold: Ratio at or above which a vessel counts as laden. Start at 0.75 and
            calibrate against known laden/ballast voyages.
        reference_draughts: Optional precomputed ``mmsi`` -> ``max_draught`` table. When
            omitted it is derived from ``static`` itself.

    Returns:
        A copy of ``static`` with ``max_draught``, ``max_draught_source``, ``laden_ratio``,
        ``is_laden`` and ``draught_age_hours``.
    """
    if static.empty:
        return static.assign(
            max_draught=pd.Series(dtype="float64"),
            max_draught_source=pd.Series(dtype="object"),
            laden_ratio=pd.Series(dtype="float64"),
            is_laden=pd.Series(dtype="object"),
            draught_age_hours=pd.Series(dtype="float64"),
        )

    frame = static.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["draught"] = pd.to_numeric(frame["draught"], errors="coerce")
    # A zero draught means "not reported", not "floating on the surface".
    frame.loc[frame["draught"] <= 0, "draught"] = np.nan

    if reference_draughts is None:
        reference_draughts = (
            frame.groupby("mmsi", as_index=False)["draught"]
            .max()
            .rename(columns={"draught": "observed_max_draught"})
        )
    else:
        reference_draughts = reference_draughts.rename(
            columns={"max_draught": "observed_max_draught"}
        )

    frame = frame.merge(reference_draughts, on="mmsi", how="left")
    lengths = (
        frame["length_m"] if "length_m" in frame.columns else pd.Series(np.nan, index=frame.index)
    )
    fallback = lengths.map(class_typical_max_draught)

    frame["max_draught"] = frame["observed_max_draught"].fillna(fallback)
    frame["max_draught_source"] = np.where(
        frame["observed_max_draught"].notna(),
        "observed",
        np.where(fallback.notna(), "class_typical", "unknown"),
    )
    frame = frame.drop(columns=["observed_max_draught"])

    with np.errstate(divide="ignore", invalid="ignore"):
        frame["laden_ratio"] = frame["draught"] / frame["max_draught"].replace(0.0, np.nan)

    # Object dtype, not bool: "unknown" is a real third outcome and must not collapse to False.
    frame["is_laden"] = np.where(
        frame["laden_ratio"].isna(), None, frame["laden_ratio"] >= threshold
    )
    frame["draught_age_hours"] = (
        frame["timestamp"].max() - frame["timestamp"]
    ).dt.total_seconds() / 3600.0

    known = frame["laden_ratio"].notna()
    log_stage(
        logger,
        "vessel_state.classify_laden",
        frame,
        threshold=threshold,
        ratio_known=int(known.sum()),
        ratio_unknown=int((~known).sum()),
        class_typical_fallback=int((frame["max_draught_source"] == "class_typical").sum()),
    )
    return frame
