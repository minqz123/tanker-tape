"""Laden classification, waiting fleet, dark gaps, and data-quality flags."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tanker_tape.process.vessel_state import (
    class_typical_max_draught,
    classify_laden,
    daily_waiting_fleet,
    find_dark_gaps,
    find_waiting_episodes,
    flag_quality,
    haversine_nm,
)


def test_haversine_matches_a_known_distance():
    # One degree of latitude is 60 nautical miles by definition.
    assert np.isclose(haversine_nm(0.0, 0.0, 0.0, 1.0), 60.0, rtol=0.01)


def test_laden_uses_the_vessels_own_observed_maximum(static_reports):
    result = classify_laden(static_reports, threshold=0.75)
    vessel = result[result["mmsi"] == 1].sort_values("timestamp")

    assert (vessel["max_draught"] == 22.0).all()
    assert (vessel["max_draught_source"] == "observed").all()
    # 22.0/22.0 and 21.5/22.0 are laden; 9.0/22.0 is ballast.
    assert list(vessel["is_laden"]) == [True, True, False]


def test_laden_falls_back_to_class_typical_when_no_history(static_reports):
    result = classify_laden(static_reports, threshold=0.75)
    vessel = result[result["mmsi"] == 2].iloc[0]

    # A single 15.0 m report is its own observed max, so the ratio is 1.0 by construction.
    # This is exactly why max_draught_source must be carried through to the analysis.
    assert vessel["max_draught_source"] == "observed"
    assert vessel["laden_ratio"] == 1.0


def test_zero_draught_is_treated_as_unreported_not_as_ballast(static_reports):
    result = classify_laden(static_reports, threshold=0.75)
    vessel = result[result["mmsi"] == 3].iloc[0]

    assert pd.isna(vessel["draught"])
    assert pd.isna(vessel["laden_ratio"])
    # Must be None, not False: "we don't know" is not "in ballast".
    assert vessel["is_laden"] is None


def test_laden_carries_draught_age(static_reports):
    result = classify_laden(static_reports)
    assert "draught_age_hours" in result.columns
    assert result["draught_age_hours"].min() == 0.0
    assert result["draught_age_hours"].max() > 0.0


def test_class_typical_draught_increases_with_length():
    assert class_typical_max_draught(100.0) < class_typical_max_draught(200.0)
    assert class_typical_max_draught(200.0) < class_typical_max_draught(320.0)
    assert class_typical_max_draught(None) is None
    assert class_typical_max_draught(-5.0) is None


def test_impossible_speed_is_flagged(base_time):
    positions = pd.DataFrame(
        [
            {"mmsi": 1, "timestamp": base_time, "lon": 56.0, "lat": 26.0},
            # Five degrees of longitude in six minutes is far beyond 40 knots.
            {"mmsi": 1, "timestamp": base_time + pd.Timedelta(minutes=6), "lon": 61.0, "lat": 26.0},
        ]
    )
    flagged = flag_quality(positions)
    assert bool(flagged["flag_impossible_speed"].iloc[1]) is True
    assert bool(flagged["flag_any"].iloc[1]) is True


def test_plausible_movement_is_not_flagged(base_time):
    positions = pd.DataFrame(
        [
            {"mmsi": 1, "timestamp": base_time, "lon": 56.0, "lat": 26.0},
            {"mmsi": 1, "timestamp": base_time + pd.Timedelta(hours=1), "lon": 56.2, "lat": 26.0},
        ]
    )
    assert not flag_quality(positions)["flag_impossible_speed"].any()


def test_spoofing_cluster_is_flagged(base_time):
    # Six vessels reporting the identical position at the same minute: the GPS-jamming
    # signature PortWatch warns about around Hormuz.
    positions = pd.DataFrame(
        [{"mmsi": mmsi, "timestamp": base_time, "lon": 56.2, "lat": 26.6} for mmsi in range(1, 7)]
    )
    flagged = flag_quality(positions, spoof_min_vessels=5)
    assert flagged["flag_shared_position"].all()


def test_distinct_positions_are_not_flagged_as_spoofing(base_time):
    positions = pd.DataFrame(
        [
            {"mmsi": mmsi, "timestamp": base_time, "lon": 56.2 + mmsi * 0.01, "lat": 26.6}
            for mmsi in range(1, 7)
        ]
    )
    assert not flag_quality(positions, spoof_min_vessels=5)["flag_shared_position"].any()


def test_flag_quality_never_drops_rows(base_time):
    positions = pd.DataFrame(
        [
            {"mmsi": 1, "timestamp": base_time, "lon": 56.0, "lat": 26.0},
            {"mmsi": 1, "timestamp": base_time + pd.Timedelta(minutes=1), "lon": 61.0, "lat": 26.0},
        ]
    )
    assert len(flag_quality(positions)) == len(positions)


def test_dark_gap_detected_between_two_in_zone_fixes(hormuz, base_time):
    inside = (56.2, 26.6)
    positions = pd.DataFrame(
        [
            {"mmsi": 1, "timestamp": base_time, "lon": inside[0], "lat": inside[1]},
            {
                "mmsi": 1,
                "timestamp": base_time + pd.Timedelta(hours=15),
                "lon": inside[0] + 0.01,
                "lat": inside[1],
            },
        ]
    )
    gaps = find_dark_gaps(positions, hormuz, min_gap_hours=12.0)

    assert len(gaps) == 1
    assert gaps.iloc[0]["gap_hours"] == 15.0


def test_gap_is_not_dark_when_the_vessel_left_the_zone(hormuz, base_time):
    positions = pd.DataFrame(
        [
            {"mmsi": 1, "timestamp": base_time, "lon": 56.2, "lat": 26.6},
            # Well outside the Hormuz zone polygon.
            {"mmsi": 1, "timestamp": base_time + pd.Timedelta(hours=15), "lon": 70.0, "lat": 15.0},
        ]
    )
    assert find_dark_gaps(positions, hormuz, min_gap_hours=12.0).empty


def test_waiting_episode_requires_both_slow_speed_and_duration(hormuz, base_time):
    rows = []
    for hour in range(0, 10):
        rows.append(
            {
                "mmsi": 1,
                "timestamp": base_time + pd.Timedelta(hours=hour),
                "lon": 56.2,
                "lat": 26.6,
                "sog": 0.2,
                "nav_status": 1,
            }
        )
    episodes = find_waiting_episodes(pd.DataFrame(rows), hormuz, min_hours=6.0)

    assert len(episodes) == 1
    assert episodes.iloc[0]["state"] == "anchored"
    assert episodes.iloc[0]["hours"] == 9.0


def test_moving_vessel_is_not_waiting(hormuz, base_time):
    rows = [
        {
            "mmsi": 1,
            "timestamp": base_time + pd.Timedelta(hours=hour),
            "lon": 56.2,
            "lat": 26.6,
            "sog": 12.0,
            "nav_status": 0,
        }
        for hour in range(10)
    ]
    assert find_waiting_episodes(pd.DataFrame(rows), hormuz, min_hours=6.0).empty


def test_short_stop_is_not_a_waiting_episode(hormuz, base_time):
    rows = [
        {
            "mmsi": 1,
            "timestamp": base_time + pd.Timedelta(hours=hour),
            "lon": 56.2,
            "lat": 26.6,
            "sog": 0.1,
            "nav_status": 1,
        }
        for hour in range(3)
    ]
    assert find_waiting_episodes(pd.DataFrame(rows), hormuz, min_hours=6.0).empty


def test_drifting_is_distinguished_from_anchored(hormuz, base_time):
    rows = [
        {
            "mmsi": 1,
            "timestamp": base_time + pd.Timedelta(hours=hour),
            "lon": 56.2,
            "lat": 26.6,
            "sog": 0.3,
            "nav_status": 0,  # under way, but not moving: drifting
        }
        for hour in range(10)
    ]
    episodes = find_waiting_episodes(pd.DataFrame(rows), hormuz, min_hours=6.0)
    assert episodes.iloc[0]["state"] == "drifting"


def test_daily_waiting_fleet_spans_every_day_of_an_episode(hormuz, base_time):
    episodes = pd.DataFrame(
        [
            {
                "mmsi": 1,
                "zone_key": "hormuz",
                "start": base_time,
                "end": base_time + pd.Timedelta(days=2),
                "hours": 48.0,
                "state": "anchored",
                "n_fixes": 48,
            }
        ]
    )
    daily = daily_waiting_fleet(episodes)

    assert len(daily) == 3
    assert (daily["n_waiting"] == 1).all()
