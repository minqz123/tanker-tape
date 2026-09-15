"""The raw-AIS-to-daily-metrics pipeline."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from tanker_tape.config import load_zones
from tanker_tape.process.ais_metrics import (
    build_quality_daily,
    compute_ais_metrics,
    daily_laden_share,
)


@pytest.fixture
def zones():
    return {"hormuz": load_zones()["hormuz"]}


def _track(mmsi, start, points, sog=12.0, nav_status=0, step_hours=1):
    """Build a position track for one vessel through the given (lon, lat) points."""
    return [
        {
            "mmsi": mmsi,
            "timestamp": start + pd.Timedelta(hours=step_hours * index),
            "lon": lon,
            "lat": lat,
            "sog": sog,
            "nav_status": nav_status,
            "zone_key": "hormuz",
        }
        for index, (lon, lat) in enumerate(points)
    ]


def test_compute_metrics_on_an_empty_frame_is_safe(zones):
    tables = compute_ais_metrics(pd.DataFrame(), None, zones=zones)
    assert set(tables) == {
        "transits_daily",
        "waiting_daily",
        "laden_daily",
        "quality_daily",
        "ais_daily",
    }
    assert all(frame.empty for frame in tables.values())


def test_transits_and_waiting_appear_in_the_combined_table(zones, gate_points, base_time):
    positive, negative = gate_points
    rows = []
    # One vessel transits the gate.
    rows += _track(1, base_time, [negative, positive])
    # Another sits stationary inside the zone for ten hours: the waiting fleet.
    rows += _track(2, base_time, [(56.2, 26.6)] * 11, sog=0.2, nav_status=1)

    tables = compute_ais_metrics(pd.DataFrame(rows), None, zones=zones)

    assert not tables["transits_daily"].empty
    assert tables["waiting_daily"]["n_waiting"].max() == 1

    combined = tables["ais_daily"]
    assert "n_transits" in combined.columns
    assert "n_waiting" in combined.columns
    assert (combined["zone_key"] == "hormuz").all()


def test_transit_directions_are_kept_separate(zones, gate_points, base_time):
    """A strait can stop admitting traffic while still emptying, so direction matters."""
    positive, negative = gate_points
    rows = _track(1, base_time, [negative, positive]) + _track(2, base_time, [positive, negative])

    combined = compute_ais_metrics(pd.DataFrame(rows), None, zones=zones)["ais_daily"]

    hormuz = load_zones()["hormuz"]
    assert f"n_transits_{hormuz.direction_positive}" in combined.columns
    assert f"n_transits_{hormuz.direction_negative}" in combined.columns
    assert combined["n_transits"].sum() == 2


def test_positions_are_scoped_to_their_own_zone(base_time, gate_points):
    """A vessel seen in two far-apart zones must not fabricate a gate crossing.

    Without the bounding-box filter, consecutive fixes thousands of miles apart form
    one movement segment that can cross an unrelated gate.
    """
    all_zones = {key: load_zones()[key] for key in ("hormuz", "suez")}
    positive, negative = gate_points
    rows = [
        {
            "mmsi": 1,
            "timestamp": base_time,
            "lon": negative[0],
            "lat": negative[1],
            "sog": 12.0,
            "nav_status": 0,
            "zone_key": "hormuz",
        },
        # Same vessel, next fix, in the Suez box.
        {
            "mmsi": 1,
            "timestamp": base_time + pd.Timedelta(hours=2),
            "lon": 32.35,
            "lat": 30.6,
            "sog": 12.0,
            "nav_status": 0,
            "zone_key": "suez",
        },
    ]

    tables = compute_ais_metrics(pd.DataFrame(rows), None, zones=all_zones)

    # Neither zone should record a transit from that impossible jump.
    assert tables["transits_daily"].empty


def test_quality_daily_counts_flags_and_uptime(base_time):
    flagged = pd.DataFrame(
        {
            "mmsi": [1, 2, 3],
            "timestamp": [
                base_time,
                base_time + pd.Timedelta(hours=1),
                base_time + pd.Timedelta(hours=2),
            ],
            "zone_key": ["hormuz", "hormuz", "hormuz"],
            "flag_impossible_speed": [True, False, False],
            "flag_shared_position": [False, True, False],
            "flag_on_land": [False, False, False],
            "flag_any": [True, True, False],
        }
    )
    daily = build_quality_daily(flagged, pd.DataFrame())

    row = daily.iloc[0]
    assert row["n_positions"] == 3
    assert row["n_vessels"] == 3
    assert row["hours_with_data"] == 3
    assert row["n_impossible_speed"] == 1
    assert row["n_shared_position"] == 1
    assert row["n_flagged_any"] == 2
    assert row["n_dark_gaps"] == 0


def test_quality_daily_keeps_messages_outside_any_zone(base_time):
    flagged = pd.DataFrame(
        {
            "mmsi": [1],
            "timestamp": [base_time],
            "zone_key": [None],
            "flag_impossible_speed": [False],
            "flag_shared_position": [False],
            "flag_on_land": [False],
            "flag_any": [False],
        }
    )
    daily = build_quality_daily(flagged, pd.DataFrame())
    assert daily.iloc[0]["zone_key"] == "outside_zones"


def test_quality_daily_merges_dark_gap_counts(base_time):
    flagged = pd.DataFrame(
        {
            "mmsi": [1],
            "timestamp": [base_time],
            "zone_key": ["hormuz"],
            "flag_impossible_speed": [False],
            "flag_shared_position": [False],
            "flag_on_land": [False],
            "flag_any": [False],
        }
    )
    gaps = pd.DataFrame(
        {
            "mmsi": [1, 2],
            "zone_key": ["hormuz", "hormuz"],
            "gap_start": [base_time, base_time],
            "gap_end": [base_time + pd.Timedelta(hours=13)] * 2,
            "gap_hours": [13.0, 13.0],
            "start_lon": [56.2, 56.2],
            "start_lat": [26.6, 26.6],
        }
    )
    daily = build_quality_daily(flagged, gaps)
    assert daily.iloc[0]["n_dark_gaps"] == 2


def test_laden_share_uses_only_prior_static_reports(zones, base_time):
    zone = zones["hormuz"]
    positions = pd.DataFrame(
        _track(1, base_time, [(56.2, 26.6)]) + _track(2, base_time, [(56.21, 26.61)])
    )
    static = pd.DataFrame(
        [
            # Filed before the vessels were seen: usable.
            {
                "mmsi": 1,
                "timestamp": base_time - pd.Timedelta(days=1),
                "draught": 22.0,
                "length_m": 330.0,
            },
            # Filed a week later: must not inform today.
            {
                "mmsi": 2,
                "timestamp": base_time + pd.Timedelta(days=7),
                "draught": 20.0,
                "length_m": 330.0,
            },
        ]
    )

    daily = daily_laden_share(positions, static, zone)

    row = daily.iloc[0]
    assert row["n_present"] == 2
    # Only vessel 1 has a prior report, so the denominator is 1, not 2.
    assert row["n_laden_known"] == 1


def test_laden_share_denominator_excludes_unknown_vessels(zones, base_time):
    zone = zones["hormuz"]
    positions = pd.DataFrame(
        _track(1, base_time, [(56.2, 26.6)])
        + _track(2, base_time, [(56.21, 26.61)])
        + _track(3, base_time, [(56.22, 26.62)])
    )
    static = pd.DataFrame(
        [
            {
                "mmsi": 1,
                "timestamp": base_time - pd.Timedelta(hours=1),
                "draught": 22.0,
                "length_m": 330.0,
            }
        ]
    )

    row = daily_laden_share(positions, static, zone).iloc[0]

    assert row["n_present"] == 3
    assert row["n_laden_known"] == 1
    assert row["laden_share"] == 1.0  # of the vessels we know about


def test_laden_share_is_undefined_without_static_data(zones, base_time):
    zone = zones["hormuz"]
    positions = pd.DataFrame(_track(1, base_time, [(56.2, 26.6)]))

    row = daily_laden_share(positions, None, zone).iloc[0]

    assert row["n_present"] == 1
    assert row["n_laden_known"] == 0
    assert pd.isna(row["laden_share"]), "no static data must mean unknown, not zero"


def test_stale_static_reports_are_not_used(zones, base_time):
    zone = zones["hormuz"]
    positions = pd.DataFrame(_track(1, base_time, [(56.2, 26.6)]))
    static = pd.DataFrame(
        [
            {
                "mmsi": 1,
                "timestamp": base_time - pd.Timedelta(days=200),
                "draught": 22.0,
                "length_m": 330.0,
            }
        ]
    )

    row = daily_laden_share(positions, static, zone, max_staleness_days=30).iloc[0]

    assert row["n_laden_known"] == 0


def test_read_partitions_round_trips(isolated_data_dir, base_time):
    from tanker_tape.storage import append_partition, read_partitions

    frame = pd.DataFrame({"mmsi": [1], "timestamp": [base_time], "lon": [56.0], "lat": [26.0]})
    append_partition(frame, "ais_positions", "2026-09-14/07")
    append_partition(frame, "ais_positions", "2026-09-15/08")

    assert len(read_partitions("ais_positions")) == 2
    assert len(read_partitions("ais_positions", start=dt.date(2026, 9, 15))) == 1
    assert len(read_partitions("ais_positions", end=dt.date(2026, 9, 14))) == 1
    assert read_partitions("ais_positions", start=dt.date(2026, 10, 1)).empty


def test_read_partitions_on_missing_dataset_returns_empty(isolated_data_dir):
    from tanker_tape.storage import read_partitions

    assert read_partitions("never_collected").empty
