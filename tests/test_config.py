"""Config and geometry tests.

The coordinate-order test earns its place: a latitude/longitude swap in the aisstream
subscription is silent — the collector connects, subscribes to open ocean, and returns
nothing. That failure costs a week of unrecoverable collection before anyone notices.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tanker_tape.config import Zone, load_ports, load_zones


def test_all_configured_zones_load():
    zones = load_zones()
    assert "hormuz" in zones
    assert zones["hormuz"].name == "Strait of Hormuz"


def test_every_gate_intersects_its_zone():
    # Enforced by the model validator; this asserts the shipped config satisfies it.
    for key, zone in load_zones().items():
        assert zone.polygon.intersects(zone.line), f"{key} gate lies outside its zone"


def test_aisstream_bounding_box_is_lat_lon_not_lon_lat():
    zone = load_zones()["hormuz"]
    (south, west), (north, east) = zone.aisstream_bounding_box()
    min_lon, min_lat, max_lon, max_lat = zone.polygon.bounds

    assert (south, north) == (min_lat, max_lat)
    assert (west, east) == (min_lon, max_lon)
    # Hormuz sits near 26N 56E: latitude must be the smaller of the two.
    assert south < west, "latitude and longitude appear to be swapped"


def test_bounding_box_corners_are_ordered_south_west_then_north_east():
    for zone in load_zones().values():
        (south, west), (north, east) = zone.aisstream_bounding_box()
        assert south <= north
        assert west <= east


def test_zone_rejects_swapped_coordinates():
    with pytest.raises(ValidationError, match="latitude"):
        Zone(
            key="bad",
            name="Swapped",
            # (26.5, 56.5) reads as longitude 26.5, latitude 56.5 - plausible, but
            # (26.5, 156.0) puts latitude out of range and must be caught.
            gate=[(26.5, 156.0), (26.8, 156.2)],
            zone=[(26.0, 155.0), (27.0, 155.0), (27.0, 157.0)],
            direction_positive="in",
            direction_negative="out",
        )


def test_zone_rejects_gate_outside_its_polygon():
    with pytest.raises(ValidationError, match="does not intersect"):
        Zone(
            key="bad",
            name="Disjoint",
            gate=[(10.0, 10.0), (10.5, 10.5)],
            zone=[(50.0, 20.0), (51.0, 20.0), (51.0, 21.0), (50.0, 21.0)],
            direction_positive="in",
            direction_negative="out",
        )


def test_direction_labels_map_to_configured_sides():
    zone = load_zones()["hormuz"]
    assert zone.direction_label(1) == zone.direction_positive
    assert zone.direction_label(-1) == zone.direction_negative


def test_ports_config_loads_with_expected_keys():
    ports = load_ports()
    assert "ras_tanura" in ports
    assert ports["ras_tanura"].country == "SAU"
    # Yanbu is the East-West pipeline outlet; it must not be grouped with Gulf loadings.
    assert ports["yanbu"].group != ports["ras_tanura"].group
