"""PortWatch paging guards, port resolution, and the server-side filter.

These cover the defect the first live scheduled pull exposed: an unfiltered ports
query is millions of rows and cannot finish inside any sane job timeout.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
import pytest

from tanker_tape.ingest import portwatch


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"status {self.status_code}", request=None, response=None)
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


def _fake_directory(monkeypatch, id_field="portid", name_field="portname", names=None):
    """Stand in for the live layer: metadata plus one distinct query.

    Also neutralises the committed ID cache. Once ``data/reference/portwatch_ports.csv``
    exists in the repo, resolution reads it and never reaches these fakes — which is the
    intended production behaviour, and exactly why a test of the live path has to opt out
    of it explicitly.
    """
    monkeypatch.setattr(
        portwatch, "cached_id_map_path", lambda entity: Path("/nonexistent/cache.csv")
    )
    names = names or ["Ras Tanura", "Yanbu", "Rotterdam"]
    monkeypatch.setattr(
        portwatch,
        "fetch_layer_metadata",
        lambda *a, **k: {"fields": [{"name": id_field}, {"name": name_field}]},
    )
    directory = pd.DataFrame(
        {id_field: [f"p{index}" for index in range(len(names))], name_field: names}
    )
    calls = []

    def fake_distinct(layer_url, fields, **kwargs):
        calls.append(fields)
        return directory

    monkeypatch.setattr(portwatch, "fetch_distinct", fake_distinct)
    return calls


def test_id_map_uses_one_distinct_query_not_a_full_pull(monkeypatch):
    """Regression: building the map by paging every daily row took ~15 minutes and
    then tripped the record guard, to learn about two thousand names."""
    _fake_directory(monkeypatch)
    monkeypatch.setattr(
        portwatch, "iter_features", lambda *a, **k: pytest.fail("must not page the layer")
    )

    frame = portwatch.fetch_id_map(portwatch.PORTS_LAYER, "port")

    assert list(frame.columns) == ["port_id", "port_name"]
    assert len(frame) == 3


def test_id_map_discovers_field_names_from_metadata(monkeypatch):
    calls = _fake_directory(monkeypatch, id_field="PORTID", name_field="PORTNAME")

    frame = portwatch.fetch_id_map(portwatch.PORTS_LAYER, "port")

    assert calls == ["PORTID,PORTNAME"], "field names must come from the live metadata"
    assert list(frame.columns) == ["port_id", "port_name"]


def test_id_map_reports_an_unrecognisable_layer(monkeypatch):
    monkeypatch.setattr(
        portwatch, "fetch_layer_metadata", lambda *a, **k: {"fields": [{"name": "mystery"}]}
    )
    with pytest.raises(KeyError, match="publishes"):
        portwatch.fetch_id_map(portwatch.PORTS_LAYER, "port")


def test_resolve_port_ids_matches_by_name(monkeypatch):
    _fake_directory(monkeypatch)

    resolved, unresolved, id_column = portwatch.resolve_port_ids()

    assert resolved["ras_tanura"] == ["p0"]
    assert resolved["yanbu"] == ["p1"]
    assert "basrah" in unresolved
    assert id_column == "portid"


def test_resolve_port_ids_returns_the_live_id_column_name(monkeypatch):
    """Schema tolerance has to reach the WHERE clause, not stop at parsing."""
    _fake_directory(monkeypatch, id_field="PORTID", name_field="PORTNAME")

    _, _, id_column = portwatch.resolve_port_ids()

    assert id_column == "PORTID"


def test_ingest_ports_filters_server_side(monkeypatch):
    _fake_directory(monkeypatch, names=["Ras Tanura", "Yanbu"])

    seen = {}

    def fake_fetch_layer(layer_url, where="1=1", **kwargs):
        seen["where"] = where
        return pd.DataFrame(
            {
                "portid": ["p0"],
                "portname": ["Ras Tanura"],
                "date": [pd.Timestamp("2026-09-01", tz="UTC")],
                "n_tanker": [12.0],
            }
        )

    monkeypatch.setattr(portwatch, "fetch_layer", fake_fetch_layer)

    frame = portwatch.ingest_ports(store=False)

    assert "p0" in seen["where"]
    assert seen["where"].startswith("portid IN ("), seen["where"]
    assert "1=1" not in seen["where"], "an unfiltered query is what broke the scheduled run"
    assert frame["port_id"].iloc[0] == "p0"


def test_ingest_ports_uses_the_discovered_id_column(monkeypatch):
    _fake_directory(monkeypatch, id_field="PORTID", name_field="PORTNAME", names=["Ras Tanura"])

    seen = {}

    def fake_fetch_layer(layer_url, where="1=1", **kwargs):
        seen["where"] = where
        return pd.DataFrame()

    monkeypatch.setattr(portwatch, "fetch_layer", fake_fetch_layer)
    portwatch.ingest_ports(store=False)

    assert seen["where"].startswith("PORTID IN ("), seen["where"]


def test_ingest_ports_raises_when_nothing_resolves(monkeypatch):
    _fake_directory(monkeypatch, names=["Nowhere At All"])

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


def test_distinct_query_retries_a_gateway_timeout(monkeypatch):
    """Observed live: the service 504s on the distinct query over millions of rows,
    succeeding and failing on identical requests minutes apart."""
    attempts = {"n": 0}

    class Flaky:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, params=None):
            attempts["n"] += 1
            if attempts["n"] < 3:
                return _FakeResponse({}, status_code=504)
            return _FakeResponse(
                {"features": [{"attributes": {"portid": "p1", "portname": "Ras Tanura"}}]}
            )

    monkeypatch.setattr(portwatch.httpx, "Client", lambda **kw: Flaky())
    monkeypatch.setattr(portwatch.time, "sleep", lambda seconds: None)

    frame = portwatch.fetch_distinct("http://x/0", "portid,portname")

    assert attempts["n"] == 3
    assert len(frame) == 1


def test_a_client_error_is_not_retried(monkeypatch):
    attempts = {"n": 0}

    class Broken:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, params=None):
            attempts["n"] += 1
            return _FakeResponse({}, status_code=400)

    monkeypatch.setattr(portwatch.httpx, "Client", lambda **kw: Broken())
    monkeypatch.setattr(portwatch.time, "sleep", lambda seconds: None)

    with pytest.raises(httpx.HTTPStatusError):
        portwatch.fetch_distinct("http://x/0", "portid,portname")

    assert attempts["n"] == 1, "a 4xx is our fault; retrying it just wastes time"


def test_committed_directory_is_used_instead_of_querying(monkeypatch, tmp_path):
    """The whole point of committing the map: no live query on the weekly path."""
    cached = tmp_path / "portwatch_ports.csv"
    pd.DataFrame({"port_id": ["p1", "p2"], "port_name": ["Ras Tanura", "Yanbu"]}).to_csv(
        cached, index=False
    )

    monkeypatch.setattr(portwatch, "cached_id_map_path", lambda entity: cached)
    monkeypatch.setattr(
        portwatch, "fetch_id_map", lambda *a, **k: pytest.fail("must not query the service")
    )
    monkeypatch.setattr(
        portwatch,
        "fetch_layer_metadata",
        lambda *a, **k: {"fields": [{"name": "portid"}, {"name": "portname"}]},
    )

    resolved, _, id_column = portwatch.resolve_port_ids()

    assert resolved["ras_tanura"] == ["p1"]
    assert id_column == "portid"


def test_a_malformed_cache_falls_back_to_the_service(monkeypatch, tmp_path):
    cached = tmp_path / "portwatch_ports.csv"
    pd.DataFrame({"something_else": ["x"]}).to_csv(cached, index=False)
    monkeypatch.setattr(portwatch, "cached_id_map_path", lambda entity: cached)
    _fake_directory(monkeypatch, names=["Ras Tanura"])

    resolved, _, _ = portwatch.resolve_port_ids()

    assert "ras_tanura" in resolved


def test_no_cache_falls_back_to_the_service(monkeypatch, tmp_path):
    monkeypatch.setattr(portwatch, "cached_id_map_path", lambda entity: tmp_path / "absent.csv")
    _fake_directory(monkeypatch, names=["Ras Tanura"])

    resolved, _, _ = portwatch.resolve_port_ids()

    assert "ras_tanura" in resolved
