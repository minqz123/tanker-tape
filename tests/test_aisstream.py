"""aisstream envelope parsing and subscription construction.

The subscription test matters disproportionately: a malformed bounding box does not
error, it just yields an empty stream, and there is no way to backfill the days lost
before anyone notices.
"""

from __future__ import annotations

import pandas as pd

from tanker_tape.config import load_zones
from tanker_tape.ingest.aisstream import (
    AisCollector,
    parse_position,
    parse_static,
    ship_type_category,
)


def _position_envelope(**overrides):
    envelope = {
        "MessageType": "PositionReport",
        "MetaData": {
            "MMSI": 636019825,
            "ShipName": "TEST TANKER",
            "latitude": 26.6,
            "longitude": 56.2,
            "time_utc": "2026-03-02 07:21:33.123456789 +0000 UTC",
        },
        "Message": {
            "PositionReport": {
                "UserID": 636019825,
                "Latitude": 26.6,
                "Longitude": 56.2,
                "Sog": 11.4,
                "Cog": 122.0,
                "TrueHeading": 120,
                "NavigationalStatus": 0,
            }
        },
    }
    envelope["Message"]["PositionReport"].update(overrides)
    return envelope


def test_ship_type_categories_follow_itu_ranges():
    assert ship_type_category(80) == "tanker"
    assert ship_type_category(89) == "tanker"
    assert ship_type_category(70) == "cargo"
    assert ship_type_category(79) == "cargo"
    assert ship_type_category(60) == "other"
    assert ship_type_category(None) == "unknown"


def test_parse_position_extracts_the_expected_fields():
    row = parse_position(_position_envelope(), load_zones())

    assert row["mmsi"] == 636019825
    assert row["lat"] == 26.6
    assert row["lon"] == 56.2
    assert row["sog"] == 11.4
    assert row["zone_key"] == "hormuz"


def test_parse_position_handles_the_go_style_timestamp():
    row = parse_position(_position_envelope(), load_zones())
    assert row["timestamp"] == pd.Timestamp("2026-03-02 07:21:33.123456789", tz="UTC")


def test_parse_position_outside_any_zone_is_kept_with_null_zone():
    envelope = _position_envelope(Latitude=0.0, Longitude=0.0)
    envelope["MetaData"]["latitude"] = 0.0
    envelope["MetaData"]["longitude"] = 0.0

    row = parse_position(envelope, load_zones())

    assert row is not None, "a message outside the polygon must be kept, not dropped"
    assert row["zone_key"] is None


def test_parse_position_returns_none_without_coordinates():
    envelope = {"MessageType": "PositionReport", "MetaData": {}, "Message": {"PositionReport": {}}}
    assert parse_position(envelope, load_zones()) is None


def test_parse_static_extracts_draught_and_length():
    envelope = {
        "MessageType": "ShipStaticData",
        "MetaData": {"MMSI": 636019825, "time_utc": "2026-03-02 07:00:00 +0000 UTC"},
        "Message": {
            "ShipStaticData": {
                "UserID": 636019825,
                "ImoNumber": 9876543,
                "Name": "TEST TANKER  ",
                "Type": 80,
                "MaximumStaticDraught": 21.5,
                "Dimension": {"A": 280, "B": 50, "C": 30, "D": 30},
                "Destination": "FUJAIRAH",
            }
        },
    }

    row = parse_static(envelope)

    assert row["imo"] == 9876543
    assert row["name"] == "TEST TANKER"  # trailing AIS padding stripped
    assert row["ship_category"] == "tanker"
    assert row["draught"] == 21.5
    assert row["length_m"] == 330  # A + B


def test_subscription_message_uses_lat_lon_bounding_boxes():
    collector = AisCollector(api_key="test-key")
    message = collector.subscription_message()

    assert message["APIKey"] == "test-key"
    assert "PositionReport" in message["FilterMessageTypes"]
    assert "ShipStaticData" in message["FilterMessageTypes"]

    for box in message["BoundingBoxes"]:
        (south, west), (north, east) = box
        assert -90 <= south <= north <= 90, "first element of each corner must be latitude"
        assert -180 <= west <= east <= 180


def test_subscription_covers_every_live_zone():
    collector = AisCollector(api_key="test-key")
    message = collector.subscription_message()

    assert len(message["BoundingBoxes"]) == len(collector.zones)
    assert all(zone.collect_live for zone in collector.zones.values())


def test_handle_envelope_routes_and_counts():
    collector = AisCollector(api_key="test-key")
    collector.handle_envelope(_position_envelope())
    collector.handle_envelope({"MessageType": "SomethingElse", "MetaData": {}, "Message": {}})

    assert collector.stats.positions == 1
    assert collector.stats.unparsed == 1
    assert collector.stats.received == 2


def test_buffer_overflow_is_counted_not_silent():
    collector = AisCollector(api_key="test-key", max_queue=3)
    for _ in range(6):
        collector.handle_envelope(_position_envelope())

    assert len(collector._positions) == 3
    assert collector.stats.dropped_queue_full == 3


def test_flush_writes_and_clears_buffers(isolated_data_dir):
    collector = AisCollector(api_key="test-key")
    collector.handle_envelope(_position_envelope())

    written = collector.flush()

    assert written["positions"] == 1
    assert collector._positions == []
    assert list((isolated_data_dir / "raw" / "ais_positions").rglob("*.parquet"))
