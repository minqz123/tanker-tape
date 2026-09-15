"""Vintage storage and as-of resolution.

PortWatch revises history, so reading "latest" inside a backtest silently substitutes
numbers that did not exist at the time. These tests pin the as-of behaviour.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from tanker_tape.storage import (
    decode_vintage,
    encode_vintage,
    list_vintages,
    read_vintage,
    resolve_vintage,
    write_vintage,
)


def _frame(value: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"date": [dt.date(2026, 3, 2)], "chokepoint_id": ["chokepoint6"], "n_transits": [value]}
    )


def test_vintage_stamp_round_trips():
    stamp = dt.datetime(2026, 3, 2, 9, 30, 0, tzinfo=dt.UTC)
    assert decode_vintage(encode_vintage(stamp)) == stamp


def test_write_and_list_vintages(isolated_data_dir):
    first = dt.datetime(2026, 3, 3, 9, 0, tzinfo=dt.UTC)
    second = dt.datetime(2026, 3, 10, 9, 0, tzinfo=dt.UTC)
    write_vintage(_frame(30), "portwatch_chokepoints", retrieved_at=first)
    write_vintage(_frame(41), "portwatch_chokepoints", retrieved_at=second)

    assert list_vintages("portwatch_chokepoints") == [first, second]


def test_as_of_returns_the_vintage_that_existed_then(isolated_data_dir):
    first = dt.datetime(2026, 3, 3, 9, 0, tzinfo=dt.UTC)
    second = dt.datetime(2026, 3, 10, 9, 0, tzinfo=dt.UTC)
    write_vintage(_frame(30), "portwatch_chokepoints", retrieved_at=first)
    write_vintage(_frame(41), "portwatch_chokepoints", retrieved_at=second)

    # On 5 March only the first pull had happened, even though the value was later revised.
    as_of_early = read_vintage("portwatch_chokepoints", as_of=dt.date(2026, 3, 5))
    assert as_of_early["n_transits"].iloc[0] == 30

    as_of_late = read_vintage("portwatch_chokepoints", as_of=dt.date(2026, 3, 12))
    assert as_of_late["n_transits"].iloc[0] == 41


def test_latest_is_returned_when_no_cutoff_given(isolated_data_dir):
    write_vintage(
        _frame(30), "portwatch_chokepoints", retrieved_at=dt.datetime(2026, 3, 3, tzinfo=dt.UTC)
    )
    write_vintage(
        _frame(41), "portwatch_chokepoints", retrieved_at=dt.datetime(2026, 3, 10, tzinfo=dt.UTC)
    )

    assert read_vintage("portwatch_chokepoints")["n_transits"].iloc[0] == 41


def test_as_of_before_any_vintage_raises_rather_than_backfilling(isolated_data_dir):
    write_vintage(
        _frame(30), "portwatch_chokepoints", retrieved_at=dt.datetime(2026, 3, 3, tzinfo=dt.UTC)
    )

    with pytest.raises(FileNotFoundError, match="leak revised data"):
        resolve_vintage("portwatch_chokepoints", as_of=dt.date(2026, 1, 1))


def test_missing_dataset_raises(isolated_data_dir):
    with pytest.raises(FileNotFoundError, match="no vintages stored"):
        resolve_vintage("does_not_exist")


def test_retrieved_at_column_is_stamped(isolated_data_dir):
    write_vintage(
        _frame(30), "portwatch_chokepoints", retrieved_at=dt.datetime(2026, 3, 3, tzinfo=dt.UTC)
    )
    stored = read_vintage("portwatch_chokepoints")
    assert "retrieved_at" in stored.columns
