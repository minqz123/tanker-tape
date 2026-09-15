"""Generate the research report from a built feature table.

The report is deliberately opinionated about honesty. It always prints the
benchmark ladder, always prints Granger causality in both directions, and always
ends with the limitations — because the easiest way to mislead with this project
is to show the one panel that looks compelling and omit the context that makes it
ordinary.

Every section is independently guarded: if the event study cannot run, the report
says so in place and carries on to the forecasting. A partial report that names
what failed is far more useful than a traceback and no report at all.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import DISCLAIMER, __version__
from ..logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_HORIZONS = (1, 5, 20)


def _md_table(frame: pd.DataFrame, max_rows: int = 40, float_format: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Hand-rolled rather than ``DataFrame.to_markdown`` so the report does not pull
    in ``tabulate`` just to print a handful of tables.
    """
    if frame is None or frame.empty:
        return "_(no rows)_"

    display = frame.head(max_rows).copy()
    truncated = len(frame) > max_rows

    def render(value: Any) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "—"
        if isinstance(value, float | np.floating):
            # Counts arrive as floats via pandas; rendering them as "443.0000"
            # makes the tables harder to scan than they need to be.
            if float(value).is_integer() and abs(value) >= 1:
                return str(int(value))
            return float_format.format(value)
        return str(value)

    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    if truncated:
        lines.append(f"| _… {len(frame) - max_rows} more rows_ |" + " |" * (len(headers) - 1))
    return "\n".join(lines)


def discover_columns(table: pd.DataFrame) -> dict[str, list[str]]:
    """Work out which columns are prices, which are AIS features, and which are targets.

    Returns:
        Mapping with ``price_features``, ``ais_features``, ``targets`` and ``zones``.
    """
    price_candidates = [
        "brent_ret_1d",
        "brent_ret_5d",
        "brent_ret_20d",
        "brent_rv_20d",
        "brent_wti_spread",
    ]
    price_features = [column for column in price_candidates if column in table.columns]

    # AIS features are the baseline-relative ones: raw levels are non-stationary
    # and the z-scores are what the analysis plan actually calls for.
    ais_features = [
        column
        for column in table.columns
        if ("_z28d" in column or "_z90d" in column or "_yoy_dev" in column)
        and not column.endswith("_age_days")
    ]
    targets = [column for column in table.columns if "_fwd_ret_" in column]
    # Strip the source prefixes so a chokepoint covered by both PortWatch and our own
    # collection is listed once, not as "hormuz" and "ais_hormuz".
    zones = sorted(
        {
            column.split("_n_transits")[0].removeprefix("ais_").removeprefix("port_")
            for column in table.columns
            if "_n_transits" in column
        }
    )
    return {
        "price_features": price_features,
        "ais_features": ais_features,
        "targets": targets,
        "zones": zones,
    }


def _section(title: str, body: Callable[[], str]) -> str:
    """Run a report section, converting a failure into readable prose."""
    try:
        return f"## {title}\n\n{body()}\n"
    except Exception as exc:  # noqa: BLE001 - a failed section must not kill the report
        logger.warning("report section %r failed: %s", title, exc)
        return (
            f"## {title}\n\n"
            f"**This section could not be produced.** `{type(exc).__name__}: {exc}`\n\n"
            "The rest of the report is unaffected. Fix the cause and re-run.\n"
        )


def _coverage(table: pd.DataFrame, columns: dict[str, list[str]]) -> str:
    rows = []
    for name in ["brent_spot", "wti_spot", *columns["ais_features"][:6]]:
        if name not in table.columns:
            continue
        series = table[name]
        rows.append(
            {
                "column": name,
                "non_null": int(series.notna().sum()),
                "coverage": f"{series.notna().mean():.1%}",
                "first": table.loc[series.first_valid_index(), "date"]
                if series.first_valid_index() is not None
                else "—",
                "last": table.loc[series.last_valid_index(), "date"]
                if series.last_valid_index() is not None
                else "—",
            }
        )

    body = [
        f"Rows: **{len(table):,}** spanning **{table['date'].min()}** to **{table['date'].max()}**.",
        "",
        _md_table(pd.DataFrame(rows)),
        "",
        f"Detected {len(columns['zones'])} chokepoint(s): {', '.join(columns['zones']) or 'none'}.",
    ]

    ais = columns["ais_features"]
    if ais:
        worst = min(table[column].notna().mean() for column in ais if column in table.columns)
        if worst < 0.5:
            body.append(
                f"\n> **Coverage warning.** The sparsest AIS feature is populated on only "
                f"{worst:.0%} of rows. Results below are effectively driven by the subset "
                "of dates where it exists, which may not be representative."
            )
    return "\n".join(body)


def _stationarity(table: pd.DataFrame, columns: dict[str, list[str]]) -> str:
    from .causality import stationarity_report

    candidates = [
        column
        for column in ["brent_spot", *columns["price_features"], *columns["ais_features"][:4]]
        if column in table.columns
    ]
    report = stationarity_report(table, candidates)
    note = (
        "ADF's null is a unit root; KPSS's null is stationarity. They are reported together "
        "because agreement is meaningful and disagreement usually signals a structural break "
        "— and this sample contains an obvious one. Anything not clearly stationary must be "
        "differenced before the VAR and Granger sections below."
    )
    return f"{note}\n\n{_md_table(report)}"


def _event_study(table: pd.DataFrame, events: pd.DataFrame) -> str:
    from .event_study import run_event_study, split_by_regime

    per_event, average = run_event_study(table, events, return_column="brent_ret_1d")
    if average.empty:
        return (
            "No event produced a usable window. This normally means the price history does "
            "not extend far enough before the earliest event to estimate the benchmark."
        )

    regimes = split_by_regime(events)
    highlights = average[average["relative_day"].isin([0, 1, 5, 10, 20])]

    return "\n".join(
        [
            f"Events used: **{per_event['event_date'].nunique()}** "
            f"(pre-war {len(regimes['pre_war'])}, war {len(regimes['war'])}).",
            "",
            "Cumulative abnormal returns against a constant-mean benchmark estimated before "
            "the event window opens:",
            "",
            _md_table(highlights),
            "",
            "> **Read these t-statistics with suspicion.** The events are successive stages of "
            "one conflict, not independent draws, and their windows overlap. The conventional "
            "standard errors are therefore too small. Treat this section as descriptive.",
        ]
    )


def _causality(table: pd.DataFrame, columns: dict[str, list[str]]) -> str:
    from .causality import granger_both_directions, local_projection, rolling_correlation

    ais = columns["ais_features"]
    if not ais or "brent_ret_1d" not in table.columns:
        return "_Not enough columns: need `brent_ret_1d` and at least one AIS feature._"

    shock = ais[0]
    frame = table.loc[:, ["date", "brent_ret_1d", shock]].dropna().reset_index(drop=True)
    if len(frame) < 60:
        return f"_Only {len(frame)} aligned observations for `{shock}` — too few to model._"

    granger = granger_both_directions(frame, "brent_ret_1d", shock, max_lag=5)
    projections = local_projection(frame, "brent_ret_1d", shock, horizons=10)
    correlations = rolling_correlation(frame, "brent_ret_1d", shock)

    both_ways = granger[granger["p_value"] < 0.05].groupby("lag")["cause"].nunique()
    feedback = bool((both_ways > 1).any())

    parts = [
        f"Shock variable: `{shock}`. Aligned observations: **{len(frame):,}**.",
        "",
        "### Granger causality, both directions",
        "",
        _md_table(granger),
        "",
    ]
    if feedback:
        parts.append(
            "> **Both directions are significant.** That is feedback, not evidence that AIS "
            "leads prices. Oil prices influence tanker routing decisions as much as the "
            "reverse, and a one-directional reading of this table would be wrong."
        )
    else:
        parts.append(
            "Reporting both directions is the point: a significant result in one direction "
            "only means something if the reverse is not also significant."
        )

    parts.extend(
        [
            "",
            "### Local projections (Jordà)",
            "",
            "Impulse response of Brent returns to the shock, with Newey-West standard errors "
            "to account for the overlapping horizons:",
            "",
            _md_table(projections),
            "",
            "### Rolling correlation",
            "",
            f"Latest 60-day correlation: **{correlations['corr_60d'].dropna().iloc[-1]:+.3f}**"
            if correlations["corr_60d"].notna().any()
            else "_Insufficient history._",
            "",
            "A relationship that is stable across regimes and one that flips sign every few "
            "months have very different implications; the rolling window is what distinguishes "
            "them.",
        ]
    )
    return "\n".join(parts)


def _forecasting(
    table: pd.DataFrame,
    columns: dict[str, list[str]],
    horizons: tuple[int, ...],
    min_train: int,
) -> str:
    from .forecast import compare_models

    price_features = columns["price_features"]
    ais_features = columns["ais_features"]
    if not price_features:
        return "_No price features found; cannot build the benchmark ladder._"
    if not ais_features:
        return "_No AIS features found; there is nothing to test the incremental value of._"

    parts = [
        "The question is narrow and the benchmark ladder is what answers it: does adding AIS "
        "features beat price history alone, out of sample? Not does it beat nothing.",
        "",
        "Each model is walk-forward, expanding-window, refit out of sample, with an embargo "
        "equal to the forecast horizon.",
        "",
    ]

    verdicts = []
    for horizon in horizons:
        target = f"brent_fwd_ret_{horizon}d"
        if target not in table.columns:
            parts.append(f"### Horizon {horizon}d\n\n_Target `{target}` not in the table._\n")
            continue
        try:
            scoreboard, _ = compare_models(
                table,
                target,
                price_features,
                ais_features,
                horizon=horizon,
                min_train=min_train,
            )
        except ValueError as exc:
            parts.append(f"### Horizon {horizon}d\n\n_Could not run: {exc}_\n")
            continue

        parts.extend([f"### Horizon {horizon}d", "", _md_table(scoreboard.reset_index()), ""])

        if "dm_p_vs_price_only" in scoreboard.columns:
            stat = scoreboard.loc["price_plus_ais", "dm_stat_vs_price_only"]
            p_value = scoreboard.loc["price_plus_ais", "dm_p_vs_price_only"]
            if np.isfinite(p_value) and p_value < 0.05 and stat < 0:
                verdict = f"**{horizon}d: AIS improves on price-only** (DM p={p_value:.3f})."
            elif np.isfinite(p_value) and p_value < 0.05 and stat > 0:
                verdict = f"**{horizon}d: AIS makes it worse** (DM p={p_value:.3f})."
            else:
                verdict = (
                    f"{horizon}d: no distinguishable difference from price-only "
                    f"(DM p={p_value:.3f})."
                )
            verdicts.append(verdict)
            parts.extend([verdict, ""])

    if verdicts:
        parts.extend(["### Summary", "", *[f"- {verdict}" for verdict in verdicts], ""])
    parts.append(
        "> A null result here is the expected outcome and a perfectly good finding. Markets "
        "price public information quickly, and PortWatch publishes weekly — the news moved "
        "days before the data did."
    )
    return "\n".join(parts)


def _limitations() -> str:
    return (
        "These are not boilerplate; each one can change the sign of a conclusion above.\n\n"
        "- **AIS spoofing and dark vessels.** Measurement error is *correlated with the "
        "signal*: jamming and transponder-silencing rise exactly when a chokepoint is "
        "contested. Transit counts are a lower bound, and the gap widens under stress.\n"
        "- **Terrestrial coverage gaps.** Self-collected AIS depends on land-based receivers. "
        "Apparent changes can reflect receiver availability rather than ship movements.\n"
        "- **Draught is hand-entered.** Laden classification rests on a number crews often do "
        "not update between voyages. Check `max_draught_source` and `draught_age_hours` "
        "before believing a laden share.\n"
        "- **PortWatch revises.** Any result computed from 'latest' rather than an as-of "
        "vintage is using numbers that did not exist at the time.\n"
        "- **Short war-regime sample, overlapping windows.** Event-study inference overstates "
        "confidence and cannot be fixed by better code.\n"
        "- **Structural break.** Pooling across February 2026 averages two different "
        "price-formation regimes.\n"
        "- **Approximate gate geometry.** Gate lines are placed from published geography, not "
        "calibrated against observed traffic.\n"
        "- **Correlation around a geopolitical shock is not a tradable edge.** When a strait "
        "closes, transits fall and prices rise because of a third thing that was on every "
        "front page that morning. Recovering that relationship shows the pipeline works, not "
        "that it confers an information advantage."
    )


def generate_report(
    table: pd.DataFrame,
    events: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    min_train: int = 500,
) -> Path:
    """Build the research report and write it to disk as Markdown.

    Args:
        table: The daily feature table from
            :func:`tanker_tape.process.features.build_feature_table`.
        events: Event table; loaded from ``data/reference/events.csv`` when omitted.
        output_path: Destination; defaults to ``reports/research-report-<date>.md``.
        horizons: Forecast horizons in trading days.
        min_train: Minimum training rows before walk-forward starts predicting.

    Returns:
        The path written.
    """
    from ..config import REPO_ROOT
    from ..process.features import load_events

    if events is None:
        events = load_events()

    columns = discover_columns(table)
    generated_at = dt.datetime.now(dt.UTC)

    chunks = [
        "# Tanker Tape — research report",
        "",
        f"_Generated {generated_at:%Y-%m-%d %H:%M} UTC by tanker-tape {__version__}._",
        "",
        f"> {DISCLAIMER}",
        "",
        _section("Data coverage", lambda: _coverage(table, columns)),
        _section("Events", lambda: _md_table(events, max_rows=30)),
        _section("Stationarity", lambda: _stationarity(table, columns)),
        _section("Event study", lambda: _event_study(table, events)),
        _section("Dynamic relationships", lambda: _causality(table, columns)),
        _section(
            "Forecasting — the honest test",
            lambda: _forecasting(table, columns, horizons, min_train),
        ),
        _section("Limitations", _limitations),
    ]

    target = (
        Path(output_path)
        if output_path
        else (REPO_ROOT / "reports" / f"research-report-{generated_at:%Y-%m-%d}.md")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(chunks), encoding="utf-8")

    logger.info(
        "stage=report.generate path=%s sections=%d rows=%d ais_features=%d",
        target,
        len(chunks) - 5,
        len(table),
        len(columns["ais_features"]),
    )
    return target
