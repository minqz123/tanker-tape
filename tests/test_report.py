"""Research report generation.

The behaviour worth pinning here is failure isolation: a report that drops the
forecasting section because the event study raised is worse than useless, because
the omission is invisible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tanker_tape.analysis.report import _md_table, discover_columns, generate_report


@pytest.fixture
def feature_table() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    n = 900
    dates = pd.date_range("2023-01-02", periods=n, freq="B").date
    transits = 40 + rng.normal(0, 4, n)
    log_price = np.cumsum(rng.normal(0, 0.012, n)) + np.log(70)
    price = np.exp(log_price)

    table = pd.DataFrame(
        {
            "date": dates,
            "brent_spot": price,
            "wti_spot": price - 4,
            "brent_ret_1d": pd.Series(log_price).diff(),
            "brent_ret_5d": pd.Series(log_price).diff(5),
            "brent_rv_20d": pd.Series(log_price).diff().rolling(20).std(),
            "brent_wti_spread": 4.0,
            "hormuz_n_transits": transits,
            "hormuz_n_transits_z28d": rng.normal(0, 1, n),
            "hormuz_n_transits_z90d": rng.normal(0, 1, n),
        }
    )
    for horizon in (1, 5, 20):
        table[f"brent_fwd_ret_{horizon}d"] = pd.Series(log_price).shift(-horizon) - log_price
    return table


@pytest.fixture
def events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-06-03").date(), pd.Timestamp("2025-02-10").date()],
            "event": ["Test closure", "Test reopening"],
            "category": ["chokepoint_closure", "chokepoint_reopening"],
            "source_url": ["https://example.invalid/a", "https://example.invalid/b"],
        }
    )


def test_md_table_renders_headers_and_rows():
    rendered = _md_table(pd.DataFrame({"a": [1.5], "b": ["x"]}))
    assert "| a | b |" in rendered
    assert "1.5000" in rendered


def test_md_table_handles_empty_and_non_finite():
    assert "no rows" in _md_table(pd.DataFrame())
    assert "—" in _md_table(pd.DataFrame({"a": [np.nan]}))


def test_discover_columns_separates_features_from_targets(feature_table):
    found = discover_columns(feature_table)

    assert "brent_ret_1d" in found["price_features"]
    assert "hormuz_n_transits_z28d" in found["ais_features"]
    assert "brent_fwd_ret_5d" in found["targets"]
    assert found["zones"] == ["hormuz"]
    # A forward return must never be classified as a usable feature.
    assert not any("fwd_" in column for column in found["ais_features"])
    assert not any("fwd_" in column for column in found["price_features"])


def test_discover_columns_excludes_staleness_columns():
    table = pd.DataFrame({"hormuz_n_transits_z28d": [1.0], "hormuz_n_transits_z28d_age_days": [3]})
    assert discover_columns(table)["ais_features"] == ["hormuz_n_transits_z28d"]


def test_generate_report_writes_every_section(tmp_path, feature_table, events):
    target = tmp_path / "report.md"

    written = generate_report(
        feature_table, events=events, output_path=target, horizons=(1, 5), min_train=400
    )

    text = written.read_text()
    for heading in (
        "# Tanker Tape — research report",
        "## Data coverage",
        "## Events",
        "## Stationarity",
        "## Event study",
        "## Dynamic relationships",
        "## Forecasting",
        "## Limitations",
    ):
        assert heading in text, f"missing section: {heading}"


def test_report_always_carries_the_disclaimer(tmp_path, feature_table, events):
    written = generate_report(
        feature_table, events=events, output_path=tmp_path / "r.md", horizons=(1,), min_train=400
    )
    assert "investment advice" in written.read_text()


def test_report_states_both_granger_directions(tmp_path, feature_table, events):
    written = generate_report(
        feature_table, events=events, output_path=tmp_path / "r.md", horizons=(1,), min_train=400
    )
    text = written.read_text()
    assert "Granger causality, both directions" in text


def test_a_failing_section_does_not_lose_the_rest(tmp_path, feature_table, monkeypatch):
    """The whole point of the per-section guard."""
    import tanker_tape.analysis.event_study as event_study

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(event_study, "run_event_study", explode)

    written = generate_report(
        feature_table,
        events=pd.DataFrame({"date": [], "event": [], "category": [], "source_url": []}),
        output_path=tmp_path / "r.md",
        horizons=(1,),
        min_train=400,
    )

    text = written.read_text()
    assert "could not be produced" in text
    assert "synthetic failure" in text
    # The sections after the failure must still be present.
    assert "## Limitations" in text
    assert "## Forecasting" in text


def test_report_reports_a_null_result_rather_than_hiding_it(tmp_path, feature_table, events):
    # The synthetic AIS features are pure noise, so AIS cannot genuinely help.
    written = generate_report(
        feature_table, events=events, output_path=tmp_path / "r.md", horizons=(5,), min_train=400
    )
    text = written.read_text()
    assert "no distinguishable difference" in text or "makes it worse" in text
    assert "A null result here is the expected outcome" in text
