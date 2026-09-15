"""Port-group, cross-route, and self-collected-AIS wiring in the feature table.

The distinction these tests protect: PortWatch features are lagged and carried
forward, our own AIS features are same-day and deliberately not carried forward.
Getting that backwards would either leak data or invent traffic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tanker_tape.process.features import (
    build_feature_table,
    build_port_group_daily,
    resolve_configured_ports,
)


@pytest.fixture
def prices() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=400, freq="B").date
    rng = np.random.default_rng(5)
    level = 70 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    return pd.DataFrame(
        {
            "date": dates,
            "brent_spot": level,
            "brent_ret_1d": pd.Series(np.log(level)).diff(),
        }
    )


@pytest.fixture
def chokepoint_daily() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=400, freq="D").date
    rng = np.random.default_rng(6)
    frames = []
    for zone in ("hormuz", "suez", "cape_of_good_hope"):
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "zone_key": zone,
                    "n_transits": rng.poisson(30, len(dates)).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def port_daily() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=400, freq="D").date
    rng = np.random.default_rng(7)
    rows = []
    for port_id, name in (("port1", "Ras Tanura"), ("port2", "Yanbu"), ("port3", "Fujairah")):
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "port_id": port_id,
                    "port_name": name,
                    "n_tanker": rng.poisson(12, len(dates)).astype(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_ports_resolve_by_name_when_id_is_unset(port_daily):
    mapping, unresolved = resolve_configured_ports(port_daily)

    assert set(mapping["port_key"]) == {"ras_tanura", "yanbu", "fujairah"}
    # The other configured ports are absent from this vintage and must be named.
    assert "novorossiysk" in unresolved


def test_unresolved_ports_are_excluded_not_zeroed(port_daily):
    mapping, unresolved = resolve_configured_ports(port_daily)

    assert "basrah" not in set(mapping["port_key"])
    assert "basrah" in unresolved


def test_empty_vintage_leaves_every_port_unresolved():
    mapping, unresolved = resolve_configured_ports(pd.DataFrame())
    assert mapping.empty
    assert len(unresolved) > 0


def test_port_groups_aggregate_separately(port_daily):
    grouped = build_port_group_daily(port_daily, "n_tanker")

    groups = set(grouped["port_group"])
    # Yanbu is the East-West pipeline outlet and must not be pooled with Gulf loadings.
    assert "gulf_export" in groups
    assert "red_sea_export" in groups
    assert "outside_hormuz" in groups


def test_port_groups_report_missing_measure(port_daily):
    assert build_port_group_daily(port_daily, "does_not_exist").empty


def test_feature_table_includes_port_and_cross_route_columns(prices, chokepoint_daily, port_daily):
    events = pd.DataFrame(columns=["date", "event", "category", "source_url"])
    table = build_feature_table(
        prices,
        chokepoint_daily,
        events=events,
        port_daily=port_daily,
        port_value_column="n_tanker",
    )

    assert any(column.startswith("port_gulf_export_") for column in table.columns)
    assert any("cape_suez_share" in column for column in table.columns)


def test_own_ais_columns_are_prefixed_and_do_not_collide(prices, chokepoint_daily):
    """PortWatch and our own collection both measure Hormuz transits."""
    dates = pd.date_range("2025-01-01", periods=400, freq="D").date
    ais_daily = pd.DataFrame(
        {
            "date": dates,
            "zone_key": "hormuz",
            "n_transits": np.arange(len(dates), dtype="float64"),
            "n_waiting": np.full(len(dates), 5.0),
        }
    )
    events = pd.DataFrame(columns=["date", "event", "category", "source_url"])

    table = build_feature_table(prices, chokepoint_daily, events=events, ais_daily=ais_daily)

    assert "hormuz_n_transits" in table.columns  # PortWatch
    assert "ais_hormuz_n_transits" in table.columns  # our own collection
    assert "ais_hormuz_n_waiting" in table.columns


def test_portwatch_features_are_carried_forward_but_ais_is_not(prices, chokepoint_daily):
    dates = pd.date_range("2025-01-01", periods=400, freq="D").date
    # Our own metrics exist only on the first of each month: gaps mean the collector
    # was down, and inventing traffic for the missing days would be a false claim.
    sparse = pd.DataFrame(
        {
            "date": dates,
            "zone_key": "hormuz",
            "n_transits": [40.0 if day.day == 1 else np.nan for day in dates],
        }
    ).dropna()
    events = pd.DataFrame(columns=["date", "event", "category", "source_url"])

    table = build_feature_table(prices, chokepoint_daily, events=events, ais_daily=sparse)

    # PortWatch weekly data is carried forward, and its staleness is recorded.
    assert "hormuz_n_transits_age_days" in table.columns
    assert table["hormuz_n_transits"].notna().mean() > 0.9

    # Our own sparse series is left sparse, with no age column implying otherwise.
    assert table["ais_hormuz_n_transits"].notna().mean() < 0.3
    assert "ais_hormuz_n_transits_age_days" not in table.columns


def test_feature_table_still_works_without_ports_or_ais(prices, chokepoint_daily):
    events = pd.DataFrame(columns=["date", "event", "category", "source_url"])
    table = build_feature_table(prices, chokepoint_daily, events=events)

    assert "hormuz_n_transits" in table.columns
    assert not any(column.startswith("ais_") for column in table.columns)
    assert not any(column.startswith("port_") for column in table.columns)
