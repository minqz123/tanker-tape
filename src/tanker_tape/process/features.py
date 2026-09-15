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


def resolve_configured_ports(
    port_daily: pd.DataFrame,
    ports: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Match configured ports to the IDs present in a PortWatch vintage.

    Resolution prefers an explicit ``portwatch_id`` from ``config/ports.yaml`` and
    falls back to a case-insensitive name match on ``match_name``.

    A port that cannot be resolved is **reported and excluded**, never treated as
    zero activity. A terminal that has gone quiet and a terminal we failed to look
    up produce the same number otherwise, and during a supply disruption those mean
    opposite things.

    Args:
        port_daily: Normalised daily port table with ``port_id`` and ``port_name``.
        ports: Port config; loaded from ``config/ports.yaml`` when omitted.

    Returns:
        ``(mapping, unresolved)`` where ``mapping`` has columns ``port_id``,
        ``port_key`` and ``group``, and ``unresolved`` lists the config keys that
        found no match.
    """
    from ..config import load_ports

    configured = ports if ports is not None else load_ports()
    if port_daily is None or port_daily.empty:
        return pd.DataFrame(columns=["port_id", "port_key", "group"]), list(configured)

    names = port_daily.loc[:, ["port_id", "port_name"]].drop_duplicates()
    lowered = names["port_name"].str.lower()

    rows, unresolved = [], []
    for key, port in configured.items():
        identifier = getattr(port, "portwatch_id", None)
        if identifier and (names["port_id"] == identifier).any():
            matched = names[names["port_id"] == identifier]
        else:
            matched = names[lowered.str.contains(port.match_name.lower(), regex=False, na=False)]

        if matched.empty:
            unresolved.append(key)
            continue
        if len(matched) > 1:
            logger.warning(
                "port %r matched %d PortWatch entries (%s); using all of them. Pin a "
                "portwatch_id in config/ports.yaml if that is wrong.",
                key,
                len(matched),
                ", ".join(matched["port_name"].head(5)),
            )
        for port_id in matched["port_id"]:
            rows.append({"port_id": port_id, "port_key": key, "group": port.group})

    if unresolved:
        logger.warning(
            "%d configured port(s) could not be resolved and are EXCLUDED, not zeroed: %s. "
            "Run `tanker-tape resolve-ids port` to find their real IDs.",
            len(unresolved),
            ", ".join(unresolved),
        )
    mapping = pd.DataFrame(rows, columns=["port_id", "port_key", "group"])
    logger.info(
        "stage=features.resolve_ports resolved=%d unresolved=%d",
        mapping["port_key"].nunique(),
        len(unresolved),
    )
    return mapping, unresolved


def build_port_group_daily(
    port_daily: pd.DataFrame,
    value_column: str,
    ports: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Aggregate configured ports into daily totals per group.

    Grouping is what makes this readable: ``gulf_export`` versus ``red_sea_export``
    versus ``outside_hormuz`` is the comparison that shows crude being pushed around
    a blocked chokepoint rather than simply disappearing.

    Args:
        port_daily: Normalised daily port table.
        value_column: Measure to sum, e.g. a tanker port-call count.
        ports: Port config; loaded from disk when omitted.

    Returns:
        Columns ``date``, ``port_group`` and ``value_column``.
    """
    mapping, _ = resolve_configured_ports(port_daily, ports)
    if mapping.empty or value_column not in port_daily.columns:
        if value_column not in (port_daily.columns if port_daily is not None else []):
            logger.warning(
                "port measure %r not found in the vintage; available: %s",
                value_column,
                sorted(port_daily.columns) if port_daily is not None else [],
            )
        return pd.DataFrame(columns=["date", "port_group", value_column])

    joined = port_daily.merge(mapping, on="port_id", how="inner")
    grouped = (
        joined.groupby(["date", "group"], as_index=False)[value_column]
        .sum()
        .rename(columns={"group": "port_group"})
        .sort_values(["date", "port_group"])
        .reset_index(drop=True)
    )
    log_stage(logger, "features.port_groups", grouped, groups=grouped["port_group"].nunique())
    return grouped


def _widen_daily(
    long_frame: pd.DataFrame,
    group_column: str,
    respect_publication_lag: bool,
    prefix: str = "",
) -> tuple[pd.DataFrame, list[str]]:
    """Collapse a long per-group daily frame into one wide row per date.

    Args:
        long_frame: Long frame with ``date``, ``group_column`` and measures.
        group_column: Column identifying the series (zone, port group, ...).
        respect_publication_lag: Move each row to its publication date first.
        prefix: Prepended to every generated column name, so PortWatch-derived and
            self-collected measures for the same chokepoint do not collide.

    Returns:
        ``(wide_frame, generated_column_names)``.
    """
    if long_frame is None or long_frame.empty:
        return pd.DataFrame(), []

    frame = (
        apply_publication_lag(long_frame)
        if respect_publication_lag
        else long_frame.assign(available_from=long_frame["date"])
    )
    measures = [
        column for column in frame.columns if column not in {"date", group_column, "available_from"}
    ]
    if not measures:
        return pd.DataFrame(), []

    # Several observation dates can share one publication date; keep the most recent.
    collapsed = (
        frame.sort_values("date").groupby(["available_from", group_column], as_index=False).last()
    )
    # unstack rather than pivot_table: pivot_table silently drops a value column that is
    # entirely NaN, which would make a feature vanish from the table without a word.
    wide = collapsed.set_index(["available_from", group_column])[measures].unstack(group_column)
    wide.columns = [f"{prefix}{group}_{measure}" for measure, group in wide.columns]
    columns = list(wide.columns)
    wide = wide.reset_index().rename(columns={"available_from": "date"})
    return wide, columns


def build_feature_table(
    prices: pd.DataFrame,
    chokepoint_daily: pd.DataFrame,
    waiting_daily: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    value_columns: tuple[str, ...] = ("n_transits",),
    respect_publication_lag: bool = True,
    forward_fill: bool = True,
    port_daily: pd.DataFrame | None = None,
    port_value_column: str | None = None,
    ais_daily: pd.DataFrame | None = None,
    ais_value_columns: tuple[str, ...] = ("n_transits", "n_waiting", "laden_share"),
) -> pd.DataFrame:
    """Assemble the modelling table: one row per date, prices joined to every feature.

    Three families of feature are joined here, and they are treated differently on
    purpose:

    * **PortWatch chokepoints and ports** are joined on publication date and carried
      forward, because a weekly release remains the latest known value until the next
      one lands.
    * **Cross-route shares** (Cape of Good Hope versus Suez) derive from the same
      PortWatch data and inherit the same lag.
    * **Our own AIS metrics** are available the same day, so they take no publication
      lag — and they are deliberately **not** carried forward. A missing day there
      means the collector was down, and filling it would invent traffic that was
      never observed, which is a different kind of claim from "last week's published
      figure still stands".

    Args:
        prices: Output of :func:`tanker_tape.ingest.prices.build_price_panel`.
        chokepoint_daily: PortWatch chokepoint table with ``date`` and ``zone_key``.
        waiting_daily: Optional standalone waiting-fleet table.
        events: Optional event table; loaded from disk when omitted.
        value_columns: PortWatch measures to build baselines for.
        respect_publication_lag: Join PortWatch features on publication date. Only
            set False for exploratory plots, never for forecasting.
        forward_fill: Carry published values forward, with ``*_age_days`` columns.
        port_daily: Optional PortWatch port table with ``port_id``/``port_name``.
        port_value_column: Port measure to aggregate by group.
        ais_daily: Optional output of
            :func:`tanker_tape.process.ais_metrics.compute_ais_metrics`.
        ais_value_columns: Self-collected measures to build baselines for.

    Returns:
        The wide daily feature table.
    """
    if not respect_publication_lag:
        logger.warning(
            "respect_publication_lag=False - features will contain data published after "
            "the row date. Exploratory use only; never forecast on this table."
        )

    table = prices.copy()
    published_columns: list[str] = []

    chokepoint = add_baseline_features(
        chokepoint_daily, list(value_columns), group_column="zone_key"
    )
    wide, columns = _widen_daily(chokepoint, "zone_key", respect_publication_lag)
    if not wide.empty:
        table = table.merge(wide, on="date", how="left")
        published_columns.extend(columns)

    cross_route = add_cross_route_features(chokepoint_daily, value_columns[0])
    if not cross_route.empty and cross_route.shape[1] > 1:
        # Give it a group column so the same lag/widen path applies.
        as_long = cross_route.assign(_route="cross")
        wide, columns = _widen_daily(as_long, "_route", respect_publication_lag)
        if not wide.empty:
            table = table.merge(wide, on="date", how="left")
            published_columns.extend(columns)

    if port_daily is not None and port_value_column:
        groups = build_port_group_daily(port_daily, port_value_column)
        if not groups.empty:
            groups = add_baseline_features(groups, [port_value_column], group_column="port_group")
            wide, columns = _widen_daily(
                groups, "port_group", respect_publication_lag, prefix="port_"
            )
            if not wide.empty:
                table = table.merge(wide, on="date", how="left")
                published_columns.extend(columns)

    if forward_fill and published_columns:
        table = _forward_fill_published(table, published_columns)

    if ais_daily is not None and not ais_daily.empty:
        measures = [column for column in ais_value_columns if column in ais_daily.columns]
        if measures:
            own = add_baseline_features(ais_daily, measures, group_column="zone_key")
            # respect_publication_lag=False: this is our own collection, readable today.
            wide, columns = _widen_daily(own, "zone_key", False, prefix="ais_")
            if not wide.empty:
                table = table.merge(wide, on="date", how="left")
                logger.info(
                    "stage=features.ais_merge columns=%d note=not_forward_filled", len(columns)
                )
        else:
            logger.warning(
                "none of %s present in ais_daily; available: %s",
                list(ais_value_columns),
                sorted(ais_daily.columns),
            )

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
        published_features=len(published_columns),
    )
    return table
