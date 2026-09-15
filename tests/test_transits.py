"""Gate-crossing detection, direction, identity, and deduplication."""

from __future__ import annotations

import pandas as pd
import pytest

from tanker_tape.process.transits import (
    attach_vessel_identity,
    daily_transit_counts,
    deduplicate_transits,
    detect_transits,
    signed_side,
)


def _positions(rows):
    return pd.DataFrame(rows, columns=["mmsi", "timestamp", "lon", "lat"]).assign(
        timestamp=lambda frame: pd.to_datetime(frame["timestamp"], utc=True)
    )


def test_signed_side_is_opposite_across_the_line():
    assert signed_side((0.0, 0.0), (0.0, 1.0), (-1.0, 0.5)) > 0
    assert signed_side((0.0, 0.0), (0.0, 1.0), (1.0, 0.5)) < 0
    assert signed_side((0.0, 0.0), (0.0, 1.0), (0.0, 0.5)) == 0


def test_detects_a_crossing_in_each_direction(hormuz, gate_points, base_time):
    positive, negative = gate_points
    positions = _positions(
        [
            (1, base_time, *negative),
            (1, base_time + pd.Timedelta(hours=1), *positive),
            (2, base_time, *positive),
            (2, base_time + pd.Timedelta(hours=1), *negative),
        ]
    )

    transits = detect_transits(positions, hormuz)

    assert len(transits) == 2
    directions = dict(zip(transits["mmsi"], transits["direction"], strict=True))
    assert directions[1] == hormuz.direction_positive
    assert directions[2] == hormuz.direction_negative


def test_no_transit_when_the_vessel_stays_on_one_side(hormuz, gate_points, base_time):
    positive, _ = gate_points
    positions = _positions(
        [
            (1, base_time, *positive),
            (1, base_time + pd.Timedelta(hours=1), positive[0] + 0.01, positive[1] + 0.01),
        ]
    )
    assert detect_transits(positions, hormuz).empty


def test_long_gap_crossing_is_flagged_ambiguous_not_dropped(hormuz, gate_points, base_time):
    positive, negative = gate_points
    positions = _positions(
        [
            (1, base_time, *negative),
            (1, base_time + pd.Timedelta(hours=12), *positive),
        ]
    )

    transits = detect_transits(positions, hormuz, max_gap_hours=6.0)

    assert len(transits) == 1, "an over-long gap must be reported, not silently discarded"
    assert bool(transits.iloc[0]["ambiguous"]) is True


def test_crossing_time_falls_between_the_bracketing_fixes(hormuz, gate_points, base_time):
    positive, negative = gate_points
    end = base_time + pd.Timedelta(hours=2)
    positions = _positions([(1, base_time, *negative), (1, end, *positive)])

    crossing = detect_transits(positions, hormuz).iloc[0]["crossing_time"]

    assert base_time <= crossing <= end


def test_missing_columns_raise_a_clear_error(hormuz):
    with pytest.raises(KeyError, match="missing required columns"):
        detect_transits(
            pd.DataFrame({"mmsi": [1], "timestamp": [pd.Timestamp.now(tz="UTC")]}), hormuz
        )


def test_identity_prefers_imo_over_mmsi(hormuz, gate_points, base_time):
    positive, negative = gate_points
    positions = _positions(
        [(1, base_time, *negative), (1, base_time + pd.Timedelta(hours=1), *positive)]
    )
    static = pd.DataFrame([{"mmsi": 1, "imo": 9999999, "timestamp": base_time}])

    transits = attach_vessel_identity(detect_transits(positions, hormuz), static)

    assert transits.iloc[0]["vessel_id"] == "imo:9999999"


def test_identity_falls_back_to_mmsi_when_imo_unknown(hormuz, gate_points, base_time):
    positive, negative = gate_points
    positions = _positions(
        [(1, base_time, *negative), (1, base_time + pd.Timedelta(hours=1), *positive)]
    )

    transits = attach_vessel_identity(detect_transits(positions, hormuz), None)

    assert transits.iloc[0]["vessel_id"] == "mmsi:1"


def test_deduplicate_collapses_a_vessel_oscillating_on_the_gate(hormuz, gate_points, base_time):
    positive, negative = gate_points
    rows = []
    # Drift back and forth across the gate every 20 minutes for two hours.
    for step in range(7):
        point = negative if step % 2 == 0 else positive
        rows.append((1, base_time + pd.Timedelta(minutes=20 * step), *point))
    positions = _positions(rows)

    transits = attach_vessel_identity(detect_transits(positions, hormuz), None)
    assert len(transits) > 2, "sanity: the raw detector should see every oscillation"

    deduplicated = deduplicate_transits(transits, min_separation_hours=6.0)

    # One inbound and one outbound survive; the rest are the same vessel loitering.
    assert len(deduplicated) == 2
    assert set(deduplicated["direction"]) == {
        hormuz.direction_positive,
        hormuz.direction_negative,
    }


def test_deduplicate_keeps_genuinely_separate_transits(hormuz, gate_points, base_time):
    positive, negative = gate_points
    rows = [
        (1, base_time, *negative),
        (1, base_time + pd.Timedelta(hours=1), *positive),
        # Same direction again, three days later: a real second voyage.
        (1, base_time + pd.Timedelta(days=3), *negative),
        (1, base_time + pd.Timedelta(days=3, hours=1), *positive),
    ]
    transits = attach_vessel_identity(detect_transits(_positions(rows), hormuz), None)

    deduplicated = deduplicate_transits(transits, min_separation_hours=6.0)

    inbound = deduplicated[deduplicated["direction"] == hormuz.direction_positive]
    assert len(inbound) == 2


def test_deduplicate_requires_vessel_identity(hormuz, gate_points, base_time):
    positive, negative = gate_points
    positions = _positions(
        [(1, base_time, *negative), (1, base_time + pd.Timedelta(hours=1), *positive)]
    )
    with pytest.raises(KeyError, match="attach_vessel_identity"):
        deduplicate_transits(detect_transits(positions, hormuz))


def test_daily_counts_exclude_ambiguous_by_default(hormuz, gate_points, base_time):
    positive, negative = gate_points
    positions = _positions(
        [(1, base_time, *negative), (1, base_time + pd.Timedelta(hours=12), *positive)]
    )
    transits = attach_vessel_identity(detect_transits(positions, hormuz, max_gap_hours=6.0), None)

    assert daily_transit_counts(transits).empty
    assert len(daily_transit_counts(transits, include_ambiguous=True)) == 1


def test_empty_input_returns_empty_frame_with_schema(hormuz):
    result = detect_transits(pd.DataFrame(columns=["mmsi", "timestamp", "lat", "lon"]), hormuz)
    assert result.empty
    assert "direction" in result.columns
