"""Event study around dated geopolitical and chokepoint events.

Abnormal returns use a constant-mean model estimated on a window that ends before the
event window opens, so the benchmark is never contaminated by the event it is measuring.

A caveat that belongs in the write-up, not just here: with a handful of events in a single
crisis, these statistics are descriptive. The events are not independent draws — they are
successive stages of one conflict — so the conventional t-statistics overstate confidence.
Report them with that stated.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_utils import get_logger, log_stage

logger = get_logger(__name__)


def _position_of(dates: pd.Series, target: dt.date) -> int | None:
    """Index of the first trading date on or after ``target``."""
    array = pd.to_datetime(dates).to_numpy()
    position = int(np.searchsorted(array, np.datetime64(pd.Timestamp(target)), side="left"))
    return position if position < len(array) else None


def abnormal_returns(
    returns: pd.Series,
    event_position: int,
    window: tuple[int, int] = (-5, 20),
    estimation_length: int = 120,
    estimation_gap: int = 10,
) -> pd.DataFrame | None:
    """Abnormal returns around one event under a constant-mean benchmark.

    Args:
        returns: Daily returns indexed positionally (ascending date order).
        event_position: Index of the event day (relative day 0).
        window: Inclusive event window in trading days.
        estimation_length: Length of the benchmark estimation window.
        estimation_gap: Trading days left between the estimation window and the event
            window, so a pre-event drift does not leak into the benchmark.

    Returns:
        Columns ``relative_day``, ``ret``, ``abnormal_ret``, ``car``, or ``None`` when
        there is not enough history on either side.
    """
    start, end = window
    estimation_end = event_position + start - estimation_gap
    estimation_start = estimation_end - estimation_length
    if estimation_start < 0 or event_position + end >= len(returns):
        return None

    benchmark = returns.iloc[estimation_start:estimation_end].dropna()
    if len(benchmark) < estimation_length // 2:
        return None

    mean = float(benchmark.mean())
    sigma = float(benchmark.std(ddof=1))
    if not np.isfinite(sigma) or sigma == 0.0:
        return None

    event_slice = returns.iloc[event_position + start : event_position + end + 1]
    frame = pd.DataFrame(
        {
            "relative_day": np.arange(start, end + 1),
            "ret": event_slice.to_numpy(dtype="float64"),
        }
    )
    frame["abnormal_ret"] = frame["ret"] - mean
    frame["car"] = frame["abnormal_ret"].cumsum()
    frame.attrs["benchmark_mean"] = mean
    frame.attrs["benchmark_sigma"] = sigma
    return frame


def run_event_study(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    return_column: str = "brent_ret_1d",
    date_column: str = "date",
    window: tuple[int, int] = (-5, 20),
    estimation_length: int = 120,
    estimation_gap: int = 10,
    categories: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the event study across every dated event.

    Args:
        prices: Daily frame with a date column and a return column.
        events: Event table with ``date`` and ``category``.
        return_column: Return series to study.
        date_column: Date column in ``prices``.
        window: Inclusive event window in trading days.
        estimation_length: Benchmark estimation window length.
        estimation_gap: Gap between estimation and event windows.
        categories: Restrict to these event categories.

    Returns:
        ``(per_event, average)``. ``per_event`` has one row per event/relative-day;
        ``average`` holds the cross-event mean CAR with a t-statistic per relative day.
    """
    frame = prices.sort_values(date_column).reset_index(drop=True)
    returns = frame[return_column]

    selected = events if categories is None else events[events["category"].isin(categories)]
    rows = []
    skipped = 0

    for event in selected.itertuples(index=False):
        position = _position_of(frame[date_column], event.date)
        if position is None:
            skipped += 1
            continue
        result = abnormal_returns(returns, position, window, estimation_length, estimation_gap)
        if result is None:
            skipped += 1
            logger.warning(
                "skipping event %s on %s: insufficient estimation or event window",
                event.event[:60],
                event.date,
            )
            continue
        result["event_date"] = event.date
        result["event"] = event.event
        result["category"] = event.category
        rows.append(result)

    if not rows:
        logger.error("event study produced no usable events (skipped=%d)", skipped)
        return pd.DataFrame(), pd.DataFrame()

    per_event = pd.concat(rows, ignore_index=True)

    grouped = per_event.groupby("relative_day")["car"]
    average = grouped.agg(mean_car="mean", std_car="std", n_events="count").reset_index()
    standard_error = average["std_car"] / np.sqrt(average["n_events"])
    average["t_stat"] = average["mean_car"] / standard_error.replace(0.0, np.nan)
    average["p_value"] = 2 * (
        1 - stats.t.cdf(average["t_stat"].abs(), df=(average["n_events"] - 1).clip(lower=1))
    )

    log_stage(
        logger,
        "event_study.run",
        per_event,
        events_used=int(per_event["event_date"].nunique()),
        events_skipped=skipped,
    )
    return per_event, average


def split_by_regime(
    events: pd.DataFrame,
    regime_start: dt.date = dt.date(2026, 2, 28),
) -> dict[str, pd.DataFrame]:
    """Split events into pre-war and war regimes.

    The default boundary is the 28 Feb 2026 start of the war (see
    ``data/reference/events.csv``). Pooling across the boundary averages two very
    different price-formation regimes and is the single easiest way to produce a
    result that does not replicate.
    """
    pre = events[events["date"] < regime_start]
    war = events[events["date"] >= regime_start]
    logger.info(
        "stage=event_study.regimes pre_war=%d war=%d boundary=%s", len(pre), len(war), regime_start
    )
    return {"pre_war": pre, "war": war}
