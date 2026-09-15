"""Daily feature table assembly.

This is the module where leakage would enter the project, so the no-lookahead rules are
implemented here explicitly rather than assumed:

* :func:`rolling_zscore` builds its baseline from ``series.shift(1)``, so the statistic at
  date *t* uses observations strictly before *t*. The observation at *t* appears only in
  the numerator, where it is legitimately known.
* :func:`prior_year_baseline` compares against a window centred 52 weeks back — entirely
  in the past.
* :func:`publication_date` and :func:`apply_publication_lag` move each PortWatch
  observation forward to the date it could first have been read. PortWatch publishes
  weekly on Tuesdays, so Friday's transit count is not a Friday feature.

``tests/test_no_lookahead.py`` enforces all of this by perturbing future observations and
asserting that nothing at or before the perturbation changes.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..logging_utils import get_logger, log_stage

logger = get_logger(__name__)

# PortWatch publishes weekly on Tuesdays at 09:00 ET.
PUBLISH_WEEKDAY = 1  # Monday = 0
# Conservative floor between an observation's date and the publication that carries it.
# Set deliberately high: assuming data is available sooner than it is inflates results.
MIN_PUBLICATION_LAG_DAYS = 2

DEFAULT_BASELINE_WINDOWS = (28, 90)


def rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Z-score of each observation against a strictly prior rolling baseline.

    The baseline for date *t* is computed from ``series[t-window : t-1]`` inclusive — the
    ``shift(1)`` is what keeps date *t* itself out of its own mean and standard deviation,
    and keeps every later observation out entirely.

    Args:
        series: Values indexed in ascending date order.
        window: Baseline length in observations.
        min_periods: Minimum observations before emitting a value; defaults to
            ``max(5, window // 4)`` so early rows do not produce confident nonsense.

    Returns:
        The z-score series, NaN where the baseline is too short or has zero variance.
    """
    if min_periods is None:
        min_periods = max(5, window // 4)

    baseline = series.shift(1)
    mean = baseline.rolling(window, min_periods=min_periods).mean()
    std = baseline.rolling(window, min_periods=min_periods).std(ddof=0)
    # A flat baseline gives an undefined z-score; NaN is the honest answer, not infinity.
    return (series - mean) / std.replace(0.0, np.nan)


def prior_year_baseline(
    frame: pd.DataFrame,
    value_column: str,
    date_column: str = "date",
    lag_days: int = 364,
    half_window_days: int = 14,
) -> pd.Series:
    """Mean of the same calendar weeks one year earlier.

    Uses a 364-day lag so the comparison lands on the same weekday, which matters for
    port calls and transits with strong day-of-week structure.

    Args:
        frame: Frame with a date column and the value column.
        value_column: Column to average.
        date_column: Date column name.
        lag_days: Days back to centre the window.
        half_window_days: Half-width of the comparison window.

    Returns:
        A Series aligned to ``frame.index`` holding the prior-year mean.
    """
    dates = pd.to_datetime(frame[date_column])
    values = frame[value_column].to_numpy(dtype="float64")
    order = np.argsort(dates.to_numpy())
    sorted_dates = dates.to_numpy()[order]
    sorted_values = values[order]

    centres = dates - pd.Timedelta(days=lag_days)
    starts = (centres - pd.Timedelta(days=half_window_days)).to_numpy()
    ends = (centres + pd.Timedelta(days=half_window_days)).to_numpy()

    left = np.searchsorted(sorted_dates, starts, side="left")
    right = np.searchsorted(sorted_dates, ends, side="right")

    cumulative = np.concatenate([[0.0], np.nancumsum(sorted_values)])
    counts = np.concatenate([[0], np.cumsum(~np.isnan(sorted_values))])

    total = cumulative[right] - cumulative[left]
    observations = counts[right] - counts[left]
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(observations > 0, total / observations, np.nan)
    return pd.Series(means, index=frame.index)


def publication_date(
    data_date: dt.date,
    publish_weekday: int = PUBLISH_WEEKDAY,
    min_lag_days: int = MIN_PUBLICATION_LAG_DAYS,
) -> dt.date:
    """Date on which an observation for ``data_date`` first became readable.

    PortWatch publishes weekly, so an observation waits for the next publication day at
    least ``min_lag_days`` after it occurred.

    Args:
        data_date: The date the observation describes.
        publish_weekday: Weekday of publication (Monday = 0; PortWatch is Tuesday = 1).
        min_lag_days: Minimum days between observation and publication.

    Returns:
        The first publication date on or after ``data_date + min_lag_days``.
    """
    earliest = data_date + dt.timedelta(days=min_lag_days)
    days_ahead = (publish_weekday - earliest.weekday()) % 7
    return earliest + dt.timedelta(days=days_ahead)


def apply_publication_lag(
    frame: pd.DataFrame,
    date_column: str = "date",
    publish_weekday: int = PUBLISH_WEEKDAY,
    min_lag_days: int = MIN_PUBLICATION_LAG_DAYS,
) -> pd.DataFrame:
    """Add ``available_from``: the date each row may first be used as a feature.

    Join features to prices on ``available_from``, never on ``date``. The gap between the
    two is exactly the information a real-time user would not have had.

    Returns:
        A copy of ``frame`` with an ``available_from`` column.
    """
    out = frame.copy()
    dates = pd.to_datetime(out[date_column]).dt.date
    out["available_from"] = [
        publication_date(value, publish_weekday, min_lag_days) for value in dates
    ]
    lag_days = (pd.to_datetime(out["available_from"]) - pd.to_datetime(dates)).dt.days
    log_stage(
        logger,
        "features.publication_lag",
        out,
        median_lag_days=float(lag_days.median()) if len(lag_days) else float("nan"),
        max_lag_days=int(lag_days.max()) if len(lag_days) else 0,
    )
    return out


def add_baseline_features(
    frame: pd.DataFrame,
    value_columns: list[str],
    group_column: str | None = "zone_key",
    date_column: str = "date",
    windows: tuple[int, ...] = DEFAULT_BASELINE_WINDOWS,
    include_prior_year: bool = True,
) -> pd.DataFrame:
    """Add rolling z-scores and prior-year deviations for each value column.

    Args:
        frame: Long frame, one row per date (per group).
        value_columns: Columns to build baselines for.
        group_column: Column identifying independent series; ``None`` for a single series.
        date_column: Date column name.
        windows: Rolling baseline lengths in days.
        include_prior_year: Whether to add the prior-year comparison.

    Returns:
        A copy of ``frame`` with ``{column}_z{window}d`` and ``{column}_yoy_dev`` columns.
    """
    out = frame.sort_values(([group_column] if group_column else []) + [date_column]).copy()
    groups = out.groupby(group_column, sort=False) if group_column else None

    for column in value_columns:
        if column not in out.columns:
            logger.warning("skipping baselines for missing column %r", column)
            continue
        for window in windows:
            name = f"{column}_z{window}d"
            if groups is not None:
                out[name] = groups[column].transform(lambda s, w=window: rolling_zscore(s, w))
            else:
                out[name] = rolling_zscore(out[column], window)

        if include_prior_year:
            name = f"{column}_yoy_dev"
            if groups is not None:
                pieces = []
                for _, chunk in out.groupby(group_column, sort=False):
                    baseline = prior_year_baseline(chunk, column, date_column)
                    pieces.append(chunk[column] - baseline)
                out[name] = pd.concat(pieces).reindex(out.index)
            else:
                out[name] = out[column] - prior_year_baseline(out, column, date_column)

    log_stage(logger, "features.baselines", out, windows=",".join(map(str, windows)))
    return out


def add_cross_route_features(chokepoint_daily: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Compute rerouting features from per-chokepoint daily values.

    Adds ``cape_suez_share``: the Cape of Good Hope share of combined Cape+Suez traffic.
    A sustained rise is tankers taking the long way round, which is the physical signature
    of a Red Sea or Suez disruption.

    Args:
        chokepoint_daily: Long frame with ``date``, ``zone_key`` and ``value_column``.
        value_column: The traffic measure, e.g. ``"n_transits"``.

    Returns:
        Wide frame with one row per date and the cross-route columns.
    """
    wide = chokepoint_daily.pivot_table(
        index="date", columns="zone_key", values=value_column, aggfunc="sum"
    )
    out = pd.DataFrame(index=wide.index)

    cape = wide["cape_of_good_hope"] if "cape_of_good_hope" in wide.columns else None
    suez = wide["suez"] if "suez" in wide.columns else None
    if cape is not None and suez is not None:
        denominator = (cape + suez).replace(0.0, np.nan)
        out["cape_suez_share"] = cape / denominator
    else:
        logger.warning(
            "cape_suez_share not computed: need both cape_of_good_hope and suez, have %s",
            sorted(wide.columns),
        )

    return out.reset_index()


def load_events(path: str | None = None) -> pd.DataFrame:
    """Load the dated event table.

    Args:
        path: Override the default ``data/reference/events.csv``.

    Returns:
        Columns ``date``, ``event``, ``category``, ``source_url``, sorted by date.
    """
    from ..config import REPO_ROOT

    target = path or (REPO_ROOT / "data" / "reference" / "events.csv")
    events = pd.read_csv(target)
    events["date"] = pd.to_datetime(events["date"]).dt.date
    events = events.sort_values("date").reset_index(drop=True)
    log_stage(logger, "features.load_events", events, categories=events["category"].nunique())
    return events


def add_event_dummies(
    frame: pd.DataFrame, events: pd.DataFrame, date_column: str = "date"
) -> pd.DataFrame:
    """Add one 0/1 column per event category, plus ``days_since_last_event``.

    Dummies mark the event date itself. ``days_since_last_event`` counts forward from the
    most recent past event, so it is backward-looking by construction.
    """
    out = frame.copy()
    dates = pd.to_datetime(out[date_column])

    for category in sorted(events["category"].dropna().unique()):
        category_dates = set(events.loc[events["category"] == category, "date"])
        out[f"event_{category}"] = dates.dt.date.isin(category_dates).astype(int)

    event_dates = np.sort(pd.to_datetime(events["date"]).to_numpy())
    if len(event_dates):
        positions = np.searchsorted(event_dates, dates.to_numpy(), side="right") - 1
        last_event = np.where(
            positions >= 0, event_dates[np.clip(positions, 0, None)], np.datetime64("NaT")
        )
        out["days_since_last_event"] = (dates.to_numpy() - last_event) / np.timedelta64(1, "D")
    else:
        out["days_since_last_event"] = np.nan

    log_stage(logger, "features.event_dummies", out, n_events=len(events))
    return out


def _forward_fill_published(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Carry weekly published values forward across the days between publications.

    PortWatch publishes on Tuesdays. Without this, every non-Tuesday row is NaN, and a
    ``.diff()`` over the resulting sparse column returns NaN everywhere — the feature
    silently disappears from the model instead of failing loudly.

    Forward-filling is **not** lookahead: it propagates a value only to dates *after* it
    was published, which is exactly what a real-time user would have on screen. The
    accompanying ``{column}_age_days`` records how stale the carried value is, so a model
    (and a reader) can tell a fresh Tuesday reading from a six-day-old one.

    Args:
        table: The merged daily table, sorted ascending by date.
        columns: The published columns to carry forward.

    Returns:
        A copy with the columns filled and an ``age_days`` column per source column.
    """
    out = table.sort_values("date").reset_index(drop=True).copy()
    dates = pd.to_datetime(out["date"])

    for column in columns:
        observed = out[column].notna()
        # Date of the most recent publication at or before each row.
        last_published = dates.where(observed).ffill()
        out[column] = out[column].ffill()
        out[f"{column}_age_days"] = (dates - last_published).dt.days

    filled = out.loc[:, columns].notna().mean().mean() if columns else float("nan")
    log_stage(
        logger,
        "features.forward_fill",
        out,
        columns_filled=len(columns),
        coverage=f"{filled:.2%}",
    )
    return out


def build_feature_table(
    prices: pd.DataFrame,
    chokepoint_daily: pd.DataFrame,
    waiting_daily: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    value_columns: tuple[str, ...] = ("n_transits",),
    respect_publication_lag: bool = True,
    forward_fill: bool = True,
) -> pd.DataFrame:
    """Assemble the modelling table: one row per date, prices joined to AIS features.

    AIS features are joined on ``available_from`` rather than ``date`` when
    ``respect_publication_lag`` is set, so each row contains only what a user could have
    read that morning.

    Args:
        prices: Output of :func:`tanker_tape.ingest.prices.build_price_panel`.
        chokepoint_daily: Long frame with ``date``, ``zone_key`` and the value columns.
        waiting_daily: Optional output of :func:`daily_waiting_fleet`.
        events: Optional event table; loaded from disk when omitted.
        value_columns: AIS measures to build baselines for.
        respect_publication_lag: Join on publication date rather than observation date.
            Only set False for exploratory plots, never for forecasting.
        forward_fill: Carry each published value forward until the next publication, and
            add ``*_age_days`` columns recording how stale it is. See
            :func:`_forward_fill_published` for why this is not lookahead.

    Returns:
        The wide daily feature table.
    """
    chokepoint = add_baseline_features(
        chokepoint_daily, list(value_columns), group_column="zone_key"
    )

    if respect_publication_lag:
        chokepoint = apply_publication_lag(chokepoint)
        join_column = "available_from"
    else:
        logger.warning(
            "respect_publication_lag=False - features will contain data published after "
            "the row date. Exploratory use only; never forecast on this table."
        )
        chokepoint = chokepoint.assign(available_from=chokepoint["date"])
        join_column = "available_from"

    feature_columns = [
        column
        for column in chokepoint.columns
        if column not in {"date", "zone_key", "available_from"}
    ]
    # Several observation dates can share one publication date; keep the most recent.
    collapsed = (
        chokepoint.sort_values("date").groupby([join_column, "zone_key"], as_index=False).last()
    )
    # unstack rather than pivot_table: pivot_table silently drops a value column that is
    # entirely NaN, which would make a feature vanish from the table without a word.
    wide = collapsed.set_index([join_column, "zone_key"])[feature_columns].unstack("zone_key")
    wide.columns = [f"{zone}_{measure}" for measure, zone in wide.columns]
    ais_columns = list(wide.columns)
    wide = wide.reset_index().rename(columns={join_column: "date"})

    table = prices.merge(wide, on="date", how="left")

    if forward_fill and ais_columns:
        table = _forward_fill_published(table, ais_columns)

    if waiting_daily is not None and not waiting_daily.empty:
        waiting = waiting_daily.pivot_table(
            index="date",
            columns="zone_key",
            values=["n_waiting", "n_anchored", "n_drifting"],
            aggfunc="last",
        )
        waiting.columns = [f"{zone}_{measure}" for measure, zone in waiting.columns]
        table = table.merge(waiting.reset_index(), on="date", how="left")

    events_table = events if events is not None else load_events()
    table = add_event_dummies(table, events_table)

    log_stage(
        logger,
        "features.build_feature_table",
        table,
        columns=table.shape[1],
        publication_lag=respect_publication_lag,
    )
    return table
