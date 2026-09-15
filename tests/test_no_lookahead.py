"""The no-lookahead guarantee.

The test strategy is the strongest one available: compute features, then *change the
future* and recompute. Any feature value at or before the perturbation date that moves is
reading data it could not have had.

This catches the whole class of leakage bugs — a centred rolling window, a full-sample
scaler, a baseline that forgot to shift — without needing to know which one was written.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tanker_tape.process.features import (
    add_baseline_features,
    apply_publication_lag,
    prior_year_baseline,
    publication_date,
    rolling_zscore,
)


def test_rolling_zscore_matches_a_hand_computation():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rolling_zscore(series, window=3, min_periods=1)

    # At index 3 the baseline is the prior three observations [1, 2, 3]:
    # mean 2.0, population sd sqrt(2/3) = 0.8165, so z = (4 - 2) / 0.8165.
    assert np.isclose(result.iloc[3], 2.0 / np.sqrt(2.0 / 3.0))


def test_rolling_zscore_excludes_the_current_observation_from_its_own_baseline():
    # A constant series with one spike: if the spike were in its own baseline the
    # z-score would be pulled toward zero.
    series = pd.Series([10.0] * 10 + [50.0])
    result = rolling_zscore(series, window=10, min_periods=5)

    # The baseline has zero variance, so the honest answer is NaN, never a finite number.
    assert pd.isna(result.iloc[10])


def test_rolling_zscore_is_unchanged_by_future_values():
    rng = np.random.default_rng(7)
    values = rng.normal(size=200)
    original = pd.Series(values)

    perturbed_values = values.copy()
    perturbed_values[150:] += 100.0  # a violent regime change, entirely in the future
    perturbed = pd.Series(perturbed_values)

    baseline = rolling_zscore(original, window=28)
    after = rolling_zscore(perturbed, window=28)

    pd.testing.assert_series_equal(baseline.iloc[:150], after.iloc[:150])


def test_baseline_features_are_unchanged_by_future_values(daily_series):
    perturb_at = 300
    features = add_baseline_features(daily_series.copy(), ["n_transits"])

    perturbed_input = daily_series.copy()
    perturbed_input.loc[perturb_at:, "n_transits"] += 500.0
    perturbed = add_baseline_features(perturbed_input, ["n_transits"])

    generated = [
        column
        for column in features.columns
        if column.startswith("n_transits_") and column != "n_transits"
    ]
    assert generated, "sanity: the fixture should generate baseline columns"

    for column in generated:
        pd.testing.assert_series_equal(
            features[column].iloc[:perturb_at],
            perturbed[column].iloc[:perturb_at],
            check_names=False,
            obj=f"{column} changed when only future values were altered",
        )


def test_prior_year_baseline_only_looks_backward(daily_series):
    perturb_at = 300
    original = prior_year_baseline(daily_series, "n_transits")

    perturbed_input = daily_series.copy()
    perturbed_input.loc[perturb_at:, "n_transits"] += 500.0
    perturbed = prior_year_baseline(perturbed_input, "n_transits")

    pd.testing.assert_series_equal(
        original.iloc[:perturb_at], perturbed.iloc[:perturb_at], check_names=False
    )


@pytest.mark.parametrize("offset", range(14))
def test_publication_date_always_lands_on_the_publication_weekday(offset):
    data_date = dt.date(2026, 3, 2) + dt.timedelta(days=offset)
    published = publication_date(data_date)

    assert published.weekday() == 1, "PortWatch publishes on Tuesdays"
    assert published >= data_date + dt.timedelta(days=2)
    assert published - data_date <= dt.timedelta(days=9)


def test_publication_date_never_precedes_the_observation():
    for offset in range(60):
        data_date = dt.date(2026, 1, 1) + dt.timedelta(days=offset)
        assert publication_date(data_date) > data_date


def test_apply_publication_lag_adds_a_strictly_later_availability_date(daily_series):
    lagged = apply_publication_lag(daily_series)

    assert "available_from" in lagged.columns
    assert (pd.to_datetime(lagged["available_from"]) > pd.to_datetime(lagged["date"])).all(), (
        "a feature must never be available on or before the day it describes"
    )


def test_forward_fill_only_propagates_values_forward():
    from tanker_tape.process.features import _forward_fill_published

    table = pd.DataFrame(
        {
            "date": pd.date_range("2026-03-02", periods=6, freq="D").date,
            "hormuz_n_transits": [np.nan, 10.0, np.nan, np.nan, 20.0, np.nan],
        }
    )

    filled = _forward_fill_published(table, ["hormuz_n_transits"])

    # The leading NaN must stay NaN: nothing had been published yet.
    assert pd.isna(filled["hormuz_n_transits"].iloc[0])
    # Tuesday's 10.0 carries forward until the next publication replaces it.
    assert list(filled["hormuz_n_transits"].iloc[1:5]) == [10.0, 10.0, 10.0, 20.0]
    assert filled["hormuz_n_transits"].iloc[5] == 20.0


def test_forward_fill_records_how_stale_each_value_is():
    from tanker_tape.process.features import _forward_fill_published

    table = pd.DataFrame(
        {
            "date": pd.date_range("2026-03-02", periods=4, freq="D").date,
            "hormuz_n_transits": [10.0, np.nan, np.nan, 20.0],
        }
    )

    filled = _forward_fill_published(table, ["hormuz_n_transits"])

    assert list(filled["hormuz_n_transits_age_days"]) == [0, 1, 2, 0]


def test_forward_fill_is_unchanged_by_future_publications():
    from tanker_tape.process.features import _forward_fill_published

    dates = pd.date_range("2026-01-06", periods=60, freq="D").date
    values = [float(index) if index % 7 == 0 else np.nan for index in range(60)]

    original = _forward_fill_published(pd.DataFrame({"date": dates, "x": values}), ["x"])

    perturbed_values = list(values)
    for index in range(40, 60):
        if not np.isnan(perturbed_values[index]):
            perturbed_values[index] += 1000.0
    perturbed = _forward_fill_published(pd.DataFrame({"date": dates, "x": perturbed_values}), ["x"])

    pd.testing.assert_series_equal(original["x"].iloc[:40], perturbed["x"].iloc[:40])


def test_a_centred_window_would_fail_this_suite():
    """Guard the guard: confirm the perturbation test can actually detect leakage.

    A test that passes against a knowingly-broken implementation is worth nothing, so
    this asserts that a centred (lookahead) rolling mean *does* change when the future
    changes.
    """
    values = np.arange(100, dtype="float64")
    original = pd.Series(values).rolling(11, center=True).mean()

    perturbed_values = values.copy()
    perturbed_values[60:] += 100.0
    perturbed = pd.Series(perturbed_values).rolling(11, center=True).mean()

    assert not original.iloc[:60].equals(perturbed.iloc[:60])
