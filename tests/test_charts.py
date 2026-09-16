"""Figure construction.

Charts are easy to ship broken because they fail silently — a marker that never
renders looks identical to one the data did not call for. These tests assert the
structural invariants that a screenshot review would otherwise have to catch.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tanker_tape.analysis.charts import (  # noqa: E402
    DARK,
    LIGHT,
    plot_event_study,
    plot_forecast_skill,
    plot_price_and_flows,
    render_all,
    save_figure,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def features() -> pd.DataFrame:
    dates = pd.date_range("2025-06-01", "2026-09-15", freq="D")
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "date": dates.date,
            "brent_spot": 70 + np.cumsum(rng.normal(0, 0.4, len(dates))),
            "hormuz_n_transits": 90 + rng.normal(0, 5, len(dates)),
        }
    )


@pytest.fixture
def events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2026-02-28").date(),
                pd.Timestamp("2026-03-02").date(),  # 2 days later: must not double-label
                pd.Timestamp("2026-07-01").date(),
            ],
            "event": ["War begins", "Strait closed", "Closed again"],
            "category": ["war_escalation", "chokepoint_closure", "chokepoint_closure"],
            "source_url": ["u"] * 3,
        }
    )


def _event_rules(axes, theme):
    return [line for line in axes.lines if line.get_color() == theme.event_rule]


def test_price_and_flows_uses_two_panels_not_two_y_axes(features):
    figure = plot_price_and_flows(features)

    assert len(figure.axes) == 2, "a dual-axis chart would have a twinned axis here"
    # A twinned axis shares the subplot position with its parent; stacked panels do not.
    positions = [tuple(axes.get_position().bounds) for axes in figure.axes]
    assert positions[0] != positions[1], "overlapping axes means a dual-axis chart"


def test_event_rules_are_actually_drawn(features, events):
    """Regression: comparing Timestamp.toordinal() to a matplotlib date axis silently
    filtered out every event, so the chart rendered with no markers at all."""
    figure = plot_price_and_flows(features, events)

    top = figure.axes[0]
    assert len(_event_rules(top, LIGHT)) == len(events)


def test_close_together_events_are_not_double_labelled(features, events):
    figure = plot_price_and_flows(features, events)
    top = figure.axes[0]

    labels = [text.get_text() for text in top.texts]

    # 28 Feb and 2 March are two days apart; only one may carry a label.
    assert "war begins" in labels
    assert labels.count("closure") <= 1
    assert len(labels) < len(events)


def test_event_labels_sit_above_the_data(features, events):
    figure = plot_price_and_flows(features, events)
    top = figure.axes[0]

    headroom = top.get_ylim()[1]
    assert headroom > features["brent_spot"].max(), "labels would overprint the price line"


def test_missing_price_column_raises(features):
    with pytest.raises(KeyError, match="brent_spot"):
        plot_price_and_flows(features.drop(columns=["brent_spot"]))


def test_missing_flow_column_still_renders(features):
    figure = plot_price_and_flows(features.drop(columns=["hormuz_n_transits"]))
    assert len(figure.axes) == 2


def test_forecast_skill_is_measured_against_the_benchmark():
    board = pd.DataFrame(
        {"rmse": [0.0291, 0.0292, 0.0294]},
        index=pd.Index(["train_mean", "price_only", "price_plus_ais"], name="model"),
    )

    figure = plot_forecast_skill(board)
    axes = figure.axes[0]
    widths = sorted(patch.get_width() for patch in axes.patches)

    # The benchmark itself is the zero line, not a bar.
    assert len(axes.patches) == 2
    # price_plus_ais is worse than price_only, so at least one bar is positive.
    assert max(widths) > 0
    assert min(widths) < 0


def test_forecast_skill_colours_encode_direction():
    board = pd.DataFrame(
        {"rmse": [0.010, 0.020, 0.030]},
        index=pd.Index(["better", "price_only", "worse"], name="model"),
    )
    axes = plot_forecast_skill(board).axes[0]

    by_sign = {patch.get_width() > 0: patch.get_facecolor() for patch in axes.patches}
    assert len(by_sign) == 2, "positive and negative bars must differ in colour"


def test_forecast_skill_without_the_benchmark_row_says_so():
    board = pd.DataFrame({"rmse": [0.03]}, index=pd.Index(["something"], name="model"))
    axes = plot_forecast_skill(board).axes[0]
    assert any("price_only" in text.get_text() for text in axes.texts)


def test_event_study_draws_a_band_and_a_line():
    days = np.arange(-5, 21)
    average = pd.DataFrame(
        {
            "relative_day": days,
            "mean_car": np.linspace(0, 0.05, len(days)),
            "std_car": np.full(len(days), 0.02),
            "n_events": np.full(len(days), 9),
        }
    )
    axes = plot_event_study(average).axes[0]

    assert len(axes.collections) >= 1, "the uncertainty band is missing"
    assert len(axes.lines) >= 1


def test_event_study_handles_no_events():
    axes = plot_event_study(pd.DataFrame()).axes[0]
    assert any("No usable events" in text.get_text() for text in axes.texts)


def test_light_and_dark_are_separate_palettes():
    assert LIGHT.surface != DARK.surface
    assert LIGHT.series_1 != DARK.series_1, "dark must be re-stepped, not an inverted light"


def test_render_all_writes_both_modes(features, events, tmp_path):
    written = render_all(features, events, output_dir=tmp_path)

    paths = written["brent-vs-transits"]
    assert len(paths) == 2
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert any(path.name.endswith("-dark.png") for path in paths)


def test_save_figure_creates_parent_directories(features, tmp_path):
    target = tmp_path / "nested" / "deeper" / "figure.png"
    save_figure(plot_price_and_flows(features), target)
    assert target.exists()
