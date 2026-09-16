"""PortWatch paging guards, port resolution, and the server-side filter.

These cover the defect the first live scheduled pull exposed: an unfiltered ports
query is millions of rows and cannot finish inside any sane job timeout.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tanker_tape.ingest import portwatch


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    """Stands in for httpx.Client, serving canned pages."""

    def __init__(self, pages, record=None):
        self._pages = pages
        self._record = record if record is not None else []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, params=None):
        self._record.append((url, params or {}))
        offset = int((params or {}).get("resultOffset", 0))
        page = self._pages(offset, params or {})
        return _FakeResponse(page)


def _rows(count, start=0):
    return {
        "features": [
            {"attributes": {"ObjectId": start + i, "portid": f"port{i}"}} for i in range(count)
        ],
        "exceededTransferLimit": True,
    }


def test_pager_stops_on_a_short_page(monkeypatch):
    def pages(offset, params):
        if offset == 0:
            return _rows(1000)
        return {"features": [{"attributes": {"ObjectId": 1}}], "exceededTransferLimit": False}

    monkeypatch.setattr(portwatch.httpx, "Client", lambda **kw: _FakeClient(pages))
    result = list(portwatch.iter_features("http://x/0", page_size=1000))
    assert len(result) == 1001


def test_pager_aborts_on_a_runaway_query(monkeypatch):
    """The failure mode that killed the first scheduled run: paging forever."""

    monkeypatch.setattr(
        portwatch.httpx, "Client", lambda **kw: _FakeClient(lambda offset, params: _rows(1000))
    )
    with pytest.raises(RuntimeError, match="filter it server-side"):
        list(portwatch.iter_features("http://x/0", page_size=1000, max_records=5000))


def test_pager_guard_can_be_disabled(monkeypatch):
    calls = {"n": 0}

    def pages(offset, params):
        calls["n"] += 1
        if calls["n"] > 3:
            return {"features": [], "exceededTransferLimit": False}
        return _rows(1000)

    monkeypatch.setattr(portwatch.httpx, "Client", lambda **kw: _FakeClient(pages))
    assert len(list(portwatch.iter_features("http://x/0", max_records=None))) == 3000


def test_pager_surfaces_service_errors(monkeypatch):
    monkeypatch.setattr(
        portwatch.httpx,
        "Client",
        lambda **kw: _FakeClient(lambda offset, params: {"error": {"code": 400}}),
    )
    with pytest.raises(RuntimeError, match="ArcGIS error"):
        list(portwatch.iter_features("http://x/0"))


def test_resolve_port_ids_matches_by_name(monkeypatch):
    directory = pd.DataFrame(
        {
            "portid": ["port1", "port2", "port3"],
            "portname": ["Ras Tanura", "Yanbu", "Rotterdam"],
        }
    )
    monkeypatch.setattr(portwatch, "fetch_distinct", lambda *a, **k: directory)

    resolved, unresolved, id_column = portwatch.resolve_port_ids()

    assert resolved["ras_tanura"] == ["port1"]
    assert resolved["yanbu"] == ["port2"]
    assert "basrah" in unresolved
    assert id_column == "portid"


def test_resolve_port_ids_returns_the_live_id_column_name(monkeypatch):
    """Schema tolerance has to reach the WHERE clause, not stop at parsing."""
    directory = pd.DataFrame({"PORTID": ["p1"], "PORTNAME": ["Ras Tanura"]})
    monkeypatch.setattr(portwatch, "fetch_distinct", lambda *a, **k: directory)

    _, _, id_column = portwatch.resolve_port_ids()

    assert id_column == "PORTID"


def test_ingest_ports_filters_server_side(monkeypatch):
    directory = pd.DataFrame({"portid": ["port1", "port2"], "portname": ["Ras Tanura", "Yanbu"]})
    monkeypatch.setattr(portwatch, "fetch_distinct", lambda *a, **k: directory)

    seen = {}

    def fake_fetch_layer(layer_url, where="1=1", **kwargs):
        seen["where"] = where
        return pd.DataFrame(
            {
                "portid": ["port1"],
                "portname": ["Ras Tanura"],
                "date": [pd.Timestamp("2026-09-01", tz="UTC")],
                "n_tanker": [12.0],
            }
        )

    monkeypatch.setattr(portwatch, "fetch_layer", fake_fetch_layer)

    frame = portwatch.ingest_ports(store=False)

    assert "port1" in seen["where"]
    assert seen["where"].startswith("portid IN ("), seen["where"]
    assert "1=1" not in seen["where"], "an unfiltered query is what broke the scheduled run"
    assert frame["port_id"].iloc[0] == "port1"


def test_ingest_ports_uses_the_discovered_id_column(monkeypatch):
    directory = pd.DataFrame({"PORTID": ["p1"], "PORTNAME": ["Ras Tanura"]})
    monkeypatch.setattr(portwatch, "fetch_distinct", lambda *a, **k: directory)

    seen = {}

    def fake_fetch_layer(layer_url, where="1=1", **kwargs):
        seen["where"] = where
        return pd.DataFrame()

    monkeypatch.setattr(portwatch, "fetch_layer", fake_fetch_layer)
    portwatch.ingest_ports(store=False)

    assert seen["where"].startswith("PORTID IN ("), seen["where"]


def test_ingest_ports_raises_when_nothing_resolves(monkeypatch):
    directory = pd.DataFrame({"portid": ["p1"], "portname": ["Nowhere At All"]})
    monkeypatch.setattr(portwatch, "fetch_distinct", lambda *a, **k: directory)

    with pytest.raises(RuntimeError, match="none of the ports"):
        portwatch.ingest_ports(store=False)


def test_explicit_where_bypasses_resolution(monkeypatch):
    seen = {}

    def fake_fetch_layer(layer_url, where="1=1", **kwargs):
        seen["where"] = where
        return pd.DataFrame()

    monkeypatch.setattr(portwatch, "fetch_layer", fake_fetch_layer)
    monkeypatch.setattr(
        portwatch, "fetch_distinct", lambda *a, **k: pytest.fail("should not resolve")
    )

    portwatch.ingest_ports(where="portid = 'port1'", store=False)
    assert seen["where"] == "portid = 'port1'"


def test_fetch_distinct_reports_a_rejected_query(monkeypatch):
    monkeypatch.setattr(
        portwatch.httpx,
        "Client",
        lambda **kw: _FakeClient(lambda offset, params: {"error": {"code": 400}}),
    )
    with pytest.raises(RuntimeError, match="Pin explicit"):
        portwatch.fetch_distinct("http://x/0", "portid,portname")


def test_normalise_names_the_real_columns_when_discovery_fails():
    frame = pd.DataFrame({"weird": [1], "other": [2]})
    with pytest.raises(KeyError, match="the layer returned"):
        portwatch.normalise_daily_table(frame, "port")
