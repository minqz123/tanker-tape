"""Shared fixtures.

The Hormuz gate points below are derived from ``config/zones.yaml`` rather than
hardcoded: ``gate_offsets`` reflects points across the real configured gate, so if the
gate is recalibrated the tests still exercise a genuine crossing.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tanker_tape.config import Zone, load_zones


@pytest.fixture(scope="session")
def hormuz() -> Zone:
    return load_zones()["hormuz"]


@pytest.fixture(scope="session")
def gate_points(hormuz: Zone) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two points straddling the Hormuz gate at its midpoint.

    Returns:
        ``(positive_side, negative_side)`` as ``(lon, lat)`` pairs. Crossing from the
        negative point to the positive point is ``direction_positive`` (inbound).
    """
    (start_lon, start_lat), (end_lon, end_lat) = hormuz.gate[0], hormuz.gate[-1]
    mid_lon, mid_lat = (start_lon + end_lon) / 2, (start_lat + end_lat) / 2

    # Normal to the gate, normalised, then stepped out a tenth of a degree either side.
    normal = np.array([end_lat - start_lat, -(end_lon - start_lon)], dtype=float)
    normal /= np.linalg.norm(normal)
    offset = 0.1 * normal

    candidate_a = (mid_lon + offset[0], mid_lat + offset[1])
    candidate_b = (mid_lon - offset[0], mid_lat - offset[1])

    from tanker_tape.process.transits import signed_side

    side_a = signed_side(hormuz.gate[0], hormuz.gate[-1], candidate_a)
    return (candidate_a, candidate_b) if side_a > 0 else (candidate_b, candidate_a)


@pytest.fixture
def base_time() -> pd.Timestamp:
    return pd.Timestamp("2026-03-02 00:00:00", tz="UTC")


@pytest.fixture
def static_reports() -> pd.DataFrame:
    """Static AIS data for three vessels with differing draught histories."""
    return pd.DataFrame(
        [
            # A VLCC that has been seen at its full 22.0 m draught before.
            {
                "mmsi": 1,
                "timestamp": pd.Timestamp("2026-03-01", tz="UTC"),
                "draught": 22.0,
                "length_m": 330.0,
            },
            {
                "mmsi": 1,
                "timestamp": pd.Timestamp("2026-03-02", tz="UTC"),
                "draught": 21.5,
                "length_m": 330.0,
            },
            # The same vessel in ballast.
            {
                "mmsi": 1,
                "timestamp": pd.Timestamp("2026-03-03", tz="UTC"),
                "draught": 9.0,
                "length_m": 330.0,
            },
            # A vessel with no history, so the class-typical fallback applies.
            {
                "mmsi": 2,
                "timestamp": pd.Timestamp("2026-03-02", tz="UTC"),
                "draught": 15.0,
                "length_m": 200.0,
            },
            # A vessel reporting zero draught, i.e. not reported at all.
            {
                "mmsi": 3,
                "timestamp": pd.Timestamp("2026-03-02", tz="UTC"),
                "draught": 0.0,
                "length_m": 250.0,
            },
        ]
    )


@pytest.fixture
def daily_series() -> pd.DataFrame:
    """A deterministic daily series for baseline and lookahead tests."""
    rng = np.random.default_rng(20260302)
    dates = pd.date_range("2024-01-01", periods=400, freq="D").date
    return pd.DataFrame(
        {
            "date": dates,
            "zone_key": "hormuz",
            "n_transits": rng.poisson(35, size=len(dates)).astype(float),
        }
    )


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Point the settings singleton at a temporary data directory."""
    from tanker_tape import config

    monkeypatch.setenv("TANKER_TAPE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TANKER_TAPE_DUCKDB_PATH", str(tmp_path / "test.duckdb"))
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


@pytest.fixture
def utc_day() -> dt.date:
    return dt.date(2026, 3, 2)
