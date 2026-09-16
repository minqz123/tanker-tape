"""Figures for the research report, the README, and the dashboard.

Design rules this module follows, and why they are not stylistic preferences:

* **No dual axes, ever.** Brent price and tanker transits have unrelated scales, and
  overlaying them on two y-axes invents a visual correlation by choosing where the
  scales line up. They are drawn as stacked panels sharing one x-axis, so the reader
  compares timing without being handed a fabricated amplitude match.
* **One series per panel**, so no legend box is needed and the panel title names the
  series outright. Identity is never carried by colour alone.
* **Recessive chrome**: hairline solid gridlines on the value axis only, no dashes,
  no top or right spines. The data should be the darkest thing in the frame.
* **Light and dark are separately chosen palettes**, each validated against its own
  surface, not one flipped into the other.

Colours come from a palette validated for colour-vision deficiency separation and
contrast (adjacent CVD ΔE 9.2 light / 9.4 dark, comfortably over the 8 threshold).
Do not substitute hues casually; re-validate if you do.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..logging_utils import get_logger, log_stage

logger = get_logger(__name__)


@dataclass(frozen=True)
class Theme:
    """Colour roles for one rendering mode."""

    name: str
    surface: str
    text_primary: str
    text_secondary: str
    grid: str
    series_1: str  # price
    series_2: str  # flows
    series_3: str  # third series, used only where directly labelled
    event_rule: str
    band: str


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    grid="#e6e6e3",
    series_1="#2a78d6",
    series_2="#eb6834",
    series_3="#1baf7a",
    event_rule="#9a9a94",
    band="#2a78d6",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    grid="#2f2f2d",
    series_1="#3987e5",
    series_2="#d95926",
    series_3="#199e70",
    event_rule="#6d6d67",
    band="#3987e5",
)

THEMES = {"light": LIGHT, "dark": DARK}

# Only these event categories get a text label. Every dated event still gets a rule;
# labelling all of them turns the panel into a wall of overlapping text.
LABELLED_CATEGORIES = {
    "war_escalation": "war begins",
    "chokepoint_closure": "closure",
    "chokepoint_reopening": "reopening",
    "infrastructure_attack": "pipeline hit",
}


def _require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "matplotlib is not installed. Install the optional extra: uv sync --extra research"
        ) from exc
    return plt


def _style_axes(axes: Any, theme: Theme, ylabel: str) -> None:
    """Apply the recessive chrome every panel shares."""
    axes.set_facecolor(theme.surface)
    axes.set_ylabel(ylabel, color=theme.text_secondary, fontsize=9)
    axes.tick_params(colors=theme.text_secondary, labelsize=8, length=0)
    # Value-axis gridlines only, hairline and solid. Dashes read as noise.
    axes.grid(axis="y", color=theme.grid, linewidth=0.6, linestyle="-")
    axes.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        axes.spines[side].set_visible(False)


def _draw_events(
    axes: Any,
    events: pd.DataFrame,
    theme: Theme,
    label: bool,
    min_label_gap_days: int = 30,
) -> None:
    """Draw a thin rule per event, labelling only well-separated major ones.

    Every dated event gets a rule; labels are rationed. During the crisis events land
    days apart (the war opens 28 Feb and the strait closes 2 March), and labelling both
    simply overprints one on the other. A label is emitted only when it clears the
    previous one by ``min_label_gap_days``, so the panel stays readable and the full
    list lives in the report's event table where nothing has to be dropped.
    """
    if events is None or events.empty:
        return

    from matplotlib.dates import date2num

    limits = axes.get_xlim()
    last_labelled: float | None = None
    labelled = 0

    for event in events.sort_values("date").itertuples(index=False):
        position = pd.Timestamp(event.date)
        # Compare in the axis's own units. Timestamp.toordinal() counts days from year 1
        # (~739,000) while a matplotlib date axis counts from 1970 (~20,000), so mixing
        # them silently filters out every event and the chart loses its markers.
        numeric = date2num(position)
        if not (limits[0] <= numeric <= limits[1]):
            continue
        axes.axvline(position, color=theme.event_rule, linewidth=0.7, alpha=0.45, zorder=1)

        tag = LABELLED_CATEGORIES.get(getattr(event, "category", ""))
        if not (label and tag):
            continue
        if last_labelled is not None and (numeric - last_labelled) < min_label_gap_days:
            continue

        # One height for all labels. Staggering them was a workaround for collisions
        # that the minimum-gap rule now prevents, and the lower tier sat across the
        # price line - which is worse than the crowding it was meant to solve.
        axes.annotate(
            tag,
            xy=(position, 0.98),
            xycoords=("data", "axes fraction"),
            fontsize=7,
            color=theme.text_secondary,
            rotation=90,
            va="top",
            ha="right",
            alpha=0.95,
        )
        last_labelled = numeric
        labelled += 1


def resolve_flow_column(features: pd.DataFrame) -> tuple[str | None, str]:
    """Find the chokepoint traffic column, and a label that honestly names it.

    PortWatch publishes no column called "transits"; which measure carries traffic is
    discovered from the live table, so the earlier guess of ``*_n_transits`` matched
    nothing and the panel rendered empty while the data sat in the frame under another
    name. Resolution walks the same candidate list the feature builder uses, preferring
    the chokepoint configured as Hormuz.

    The label matters as much as the column. Titling a panel "Strait of Hormuz" while
    plotting whichever column happened to match first would put a false claim on the
    figure, so the label is derived from what was actually found.

    Args:
        features: The daily feature table.

    Returns:
        ``(column, label)``; ``column`` is ``None`` when nothing matches.
    """
    from ..config import load_zones
    from ..process.features import TRAFFIC_MEASURE_CANDIDATES

    def is_derived(name: str) -> bool:
        return name.endswith("_age_days") or any(
            marker in name for marker in ("_z28d", "_z90d", "_yoy_dev")
        )

    zones = load_zones()
    # Preferred prefixes: the configured Hormuz ID first, then its key.
    preferred: list[tuple[str, str]] = []
    for key, zone in zones.items():
        if key != "hormuz":
            continue
        if zone.portwatch_id:
            preferred.append((zone.portwatch_id, zone.name))
        preferred.append((key, zone.name))

    for prefix, name in preferred:
        for measure in TRAFFIC_MEASURE_CANDIDATES:
            candidate = f"{prefix}_{measure}"
            if candidate in features.columns and features[candidate].notna().any():
                return candidate, f"{name} tanker traffic"

    # Nothing configured matched; fall back to any non-derived traffic measure and
    # name the panel after the column so the figure never overstates what it shows.
    for measure in TRAFFIC_MEASURE_CANDIDATES:
        for column in features.columns:
            if (
                column.endswith(f"_{measure}")
                and not is_derived(column)
                and not column.startswith("ais_")
                and features[column].notna().any()
            ):
                return column, f"Chokepoint traffic ({column})"

    return None, "Chokepoint traffic"


def plot_price_and_flows(
    features: pd.DataFrame,
    events: pd.DataFrame | None = None,
    price_column: str = "brent_spot",
    flow_column: str | None = None,
    mode: str = "light",
    smooth_days: int = 7,
) -> Any:
    """The headline figure: Brent above, chokepoint traffic below, one shared timeline.

    Deliberately **not** a dual-axis chart. Two y-scales on one plot would let the
    reader infer an amplitude relationship that is an artefact of where the scales
    were pinned. Stacked panels preserve the only honest comparison here, which is
    timing.

    Args:
        features: Daily feature table with a ``date`` column.
        events: Optional dated events; every row gets a rule, major ones get a label.
        price_column: Price series to draw in the top panel.
        flow_column: Traffic series for the bottom panel; auto-detected when omitted.
        mode: ``"light"`` or ``"dark"``.
        smooth_days: Rolling mean applied to the noisy traffic series. The raw series
            stays visible underneath so the smoothing never hides the data.

    Returns:
        The matplotlib Figure.

    Raises:
        KeyError: If the price column is absent.
    """
    plt = _require_matplotlib()
    theme = THEMES[mode]

    if price_column not in features.columns:
        raise KeyError(f"{price_column!r} not in the feature table: {sorted(features.columns)}")

    flow_label = "Tanker transits"
    if flow_column is None:
        flow_column, flow_label = resolve_flow_column(features)

    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2], "hspace": 0.18}
    )
    figure.patch.set_facecolor(theme.surface)

    price = frame.dropna(subset=[price_column])
    top.plot(
        price["date"],
        price[price_column],
        color=theme.series_1,
        linewidth=1.6,
        solid_capstyle="round",
    )
    _style_axes(top, theme, "Brent spot ($/bbl)")
    # Reserve a band at the top for event labels so they never cross the price line.
    low, high = top.get_ylim()
    top.set_ylim(low, high + (high - low) * 0.30)
    top.set_title(
        f"Brent crude and {flow_label}",
        color=theme.text_primary,
        fontsize=13,
        loc="left",
        pad=14,
    )

    if flow_column and frame[flow_column].notna().any():
        flows = frame.dropna(subset=[flow_column])
        # Raw daily underneath, smoothed on top: the trend is legible without the
        # smoothing concealing how noisy the underlying counts are.
        bottom.plot(
            flows["date"], flows[flow_column], color=theme.series_2, linewidth=0.7, alpha=0.30
        )
        smoothed = (
            flows[flow_column].rolling(smooth_days, min_periods=max(2, smooth_days // 2)).mean()
        )
        bottom.plot(
            flows["date"], smoothed, color=theme.series_2, linewidth=1.8, solid_capstyle="round"
        )
        # Short label: the title already names the chokepoint, and repeating it here
        # produced an axis label long enough to overflow into the sources footer.
        label = f"Tanker transit calls ({smooth_days}-day mean)"
    else:
        bottom.text(
            0.5,
            0.5,
            "No chokepoint traffic in the feature table yet",
            transform=bottom.transAxes,
            ha="center",
            va="center",
            color=theme.text_secondary,
            fontsize=10,
        )
        label = "Tanker transit calls"
    _style_axes(bottom, theme, label)

    _draw_events(top, events, theme, label=True)
    _draw_events(bottom, events, theme, label=False)

    figure.text(
        0.008,
        0.025,
        "Sources: EIA/FRED (Brent), IMF PortWatch (transits). Research project — not investment advice.",
        color=theme.text_secondary,
        fontsize=7.5,
    )
    # Explicit spacing rather than tight_layout: the event labels mix data and
    # axes-fraction transforms, which tight_layout cannot measure - it warns and may
    # lay the figure out wrongly.
    figure.subplots_adjust(left=0.095, right=0.985, top=0.90, bottom=0.13)
    return figure


def plot_event_study(average: pd.DataFrame, mode: str = "light") -> Any:
    """Mean cumulative abnormal return across events, with a dispersion band.

    Args:
        average: Output of :func:`tanker_tape.analysis.event_study.run_event_study`.
        mode: ``"light"`` or ``"dark"``.

    Returns:
        The matplotlib Figure.
    """
    plt = _require_matplotlib()
    theme = THEMES[mode]

    figure, axes = plt.subplots(figsize=(8, 4.2))
    figure.patch.set_facecolor(theme.surface)

    if average is None or average.empty:
        axes.text(
            0.5,
            0.5,
            "No usable events",
            transform=axes.transAxes,
            ha="center",
            va="center",
            color=theme.text_secondary,
        )
        _style_axes(axes, theme, "Cumulative abnormal return")
        return figure

    days = average["relative_day"]
    mean = average["mean_car"]
    error = average["std_car"] / np.sqrt(average["n_events"].clip(lower=1))

    axes.fill_between(days, mean - error, mean + error, color=theme.band, alpha=0.16, linewidth=0)
    axes.plot(days, mean, color=theme.series_1, linewidth=1.8, solid_capstyle="round")
    axes.axvline(0, color=theme.event_rule, linewidth=0.8, alpha=0.8)
    axes.axhline(0, color=theme.grid, linewidth=0.8)

    _style_axes(axes, theme, "Mean cumulative abnormal return")
    axes.set_xlabel("Trading days relative to event", color=theme.text_secondary, fontsize=9)
    events_used = int(average["n_events"].max()) if len(average) else 0
    axes.set_title(
        f"Brent abnormal returns around dated events (n={events_used})",
        color=theme.text_primary,
        fontsize=12,
        loc="left",
        pad=12,
    )
    figure.text(
        0.008,
        0.02,
        "Band is ±1 standard error. Events overlap and are not independent draws, so this "
        "understates uncertainty.",
        color=theme.text_secondary,
        fontsize=7.5,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


def plot_forecast_skill(
    scoreboard: pd.DataFrame,
    benchmark: str = "price_only",
    mode: str = "light",
) -> Any:
    """Forecast skill *relative to the benchmark*, which is the question being asked.

    Plotting raw RMSE as bars from zero fails here: the models score 0.0291 against
    0.0294, so every bar is the same length and the chart says nothing. What matters
    is the difference from the price-only benchmark, so that is what is drawn, with
    zero meaning "no better than price history alone".

    The two directions are a genuine polarity, so they get a diverging warm/cool pair:
    cool for lower error, warm for higher. Bars are direct-labelled because the
    differences are small enough that the axis alone would not settle them.

    Args:
        scoreboard: Output of :func:`tanker_tape.analysis.forecast.compare_models`.
        benchmark: Index label to measure against; drawn as the zero line.
        mode: ``"light"`` or ``"dark"``.

    Returns:
        The matplotlib Figure.
    """
    plt = _require_matplotlib()
    theme = THEMES[mode]

    figure, axes = plt.subplots(figsize=(8, 3.4))
    figure.patch.set_facecolor(theme.surface)

    usable = (
        scoreboard is not None
        and not scoreboard.empty
        and "rmse" in scoreboard.columns
        and benchmark in scoreboard.index
    )
    if not usable:
        axes.text(
            0.5,
            0.5,
            "No forecast results"
            if scoreboard is None or scoreboard.empty
            else f"No {benchmark!r} row to compare against",
            transform=axes.transAxes,
            ha="center",
            va="center",
            color=theme.text_secondary,
        )
        _style_axes(axes, theme, "")
        return figure

    reference = float(scoreboard.loc[benchmark, "rmse"])
    others = scoreboard.drop(index=benchmark)
    relative = ((others["rmse"] - reference) / reference * 100.0).sort_values()

    labels = [str(name).replace("_", " ") for name in relative.index]
    positions = np.arange(len(relative))
    # Cool = lower error than the benchmark, warm = higher. Opposite meanings, so
    # opposite temperatures; a single hue would hide the sign.
    colours = [theme.series_1 if value < 0 else theme.series_2 for value in relative]

    axes.barh(positions, relative.to_numpy(), color=colours, height=0.55)
    axes.axvline(0, color=theme.text_secondary, linewidth=1.0, alpha=0.7)
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, color=theme.text_secondary, fontsize=9)

    _style_axes(axes, theme, "")
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=theme.grid, linewidth=0.6, linestyle="-")

    span = float(np.abs(relative).max()) or 1.0
    for position, value in zip(positions, relative, strict=True):
        offset = span * 0.04
        axes.annotate(
            f"{value:+.1f}%",
            xy=(value + (offset if value >= 0 else -offset), position),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
            color=theme.text_secondary,
        )
    axes.set_xlim(-span * 1.45, span * 1.45)
    axes.set_xlabel(
        f"Out-of-sample RMSE vs the {benchmark.replace('_', ' ')} benchmark (%) — left is better",
        color=theme.text_secondary,
        fontsize=9,
    )
    axes.set_title(
        "Does adding AIS beat price history alone?",
        color=theme.text_primary,
        fontsize=12,
        loc="left",
        pad=12,
    )

    note = "Zero is the price-only benchmark. Bars right of it forecast worse than price history."
    if "dm_p_vs_price_only" in scoreboard.columns:
        p_value = scoreboard.get("dm_p_vs_price_only", pd.Series(dtype=float)).dropna()
        if not p_value.empty:
            note += f"  Diebold-Mariano p = {float(p_value.iloc[0]):.3f}."
    figure.text(0.008, 0.02, note, color=theme.text_secondary, fontsize=7.5)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


def save_figure(figure: Any, path: str | Path, mode: str = "light", dpi: int = 160) -> Path:
    """Write a figure to PNG on its theme's surface."""
    theme = THEMES[mode]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, facecolor=theme.surface, bbox_inches="tight")
    logger.info("stage=charts.save path=%s mode=%s", target, mode)
    return target


def render_all(
    features: pd.DataFrame,
    events: pd.DataFrame | None = None,
    average_car: pd.DataFrame | None = None,
    scoreboard: pd.DataFrame | None = None,
    output_dir: str | Path = "reports/figures",
) -> dict[str, list[Path]]:
    """Render every figure in both light and dark, for embedding in Markdown.

    GitHub picks the variant matching the reader's theme via ``<picture>``, so a
    dark-mode reader does not get a glaring white rectangle.

    Returns:
        Mapping of figure name to the paths written.
    """
    plt = _require_matplotlib()
    directory = Path(output_dir)
    written: dict[str, list[Path]] = {}

    builders = {
        "brent-vs-transits": lambda mode: plot_price_and_flows(features, events, mode=mode),
    }
    if average_car is not None and not average_car.empty:
        builders["event-study"] = lambda mode: plot_event_study(average_car, mode=mode)
    if scoreboard is not None and not scoreboard.empty:
        builders["forecast-skill"] = lambda mode: plot_forecast_skill(scoreboard, mode=mode)

    for name, builder in builders.items():
        paths = []
        for mode in ("light", "dark"):
            figure = builder(mode)
            suffix = "" if mode == "light" else "-dark"
            paths.append(save_figure(figure, directory / f"{name}{suffix}.png", mode=mode))
            plt.close(figure)
        written[name] = paths

    log_stage(logger, "charts.render_all", None, figures=len(written), directory=str(directory))
    return written
