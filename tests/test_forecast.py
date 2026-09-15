"""Walk-forward mechanics, leakage guards, and the Diebold-Mariano test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tanker_tape.analysis.forecast import (
    MeanForecast,
    ZeroForecast,
    diebold_mariano,
    evaluate,
    ridge_model,
    walk_forward,
)


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 800
    dates = pd.date_range("2023-01-01", periods=n, freq="B").date
    signal = rng.normal(size=n)
    return pd.DataFrame(
        {
            "date": dates,
            "feature_a": signal,
            "feature_b": rng.normal(size=n),
            # Target is a noisy function of the feature, so a model should beat zero.
            "brent_fwd_ret_1d": 0.5 * signal + rng.normal(scale=0.5, size=n),
        }
    )


def test_walk_forward_rejects_forward_looking_features(toy_frame):
    with pytest.raises(ValueError, match="forward-looking"):
        walk_forward(
            toy_frame,
            target="brent_fwd_ret_1d",
            feature_columns=["feature_a", "brent_fwd_ret_1d"],
            model_factory=ridge_model,
            min_train=200,
        )


def test_walk_forward_predicts_only_out_of_sample(toy_frame):
    result = walk_forward(
        toy_frame,
        target="brent_fwd_ret_1d",
        feature_columns=["feature_a"],
        model_factory=ridge_model,
        horizon=1,
        min_train=200,
    )

    assert not result.empty
    # Every prediction must be trained on strictly fewer rows than its own position.
    assert (result["n_train"] < np.arange(200, 200 + len(result))).all()


def test_walk_forward_embargo_grows_with_horizon(toy_frame):
    short = walk_forward(
        toy_frame, "brent_fwd_ret_1d", ["feature_a"], ridge_model, horizon=1, min_train=200
    )
    long = walk_forward(
        toy_frame, "brent_fwd_ret_1d", ["feature_a"], ridge_model, horizon=20, min_train=200
    )

    # A 20-day horizon must hold back 19 more training rows than a 1-day horizon.
    assert long["n_train"].iloc[0] == short["n_train"].iloc[0] - 19


def test_walk_forward_raises_when_there_is_not_enough_data(toy_frame):
    with pytest.raises(ValueError, match="not enough complete rows"):
        walk_forward(
            toy_frame.head(50), "brent_fwd_ret_1d", ["feature_a"], ridge_model, min_train=500
        )


def test_a_real_signal_beats_the_zero_benchmark(toy_frame):
    model = walk_forward(toy_frame, "brent_fwd_ret_1d", ["feature_a"], ridge_model, min_train=200)
    benchmark = walk_forward(
        toy_frame, "brent_fwd_ret_1d", ["feature_a"], ZeroForecast, min_train=200
    )

    assert evaluate(model)["rmse"] < evaluate(benchmark)["rmse"]


def test_zero_and_mean_benchmarks_behave_as_documented(toy_frame):
    zero = walk_forward(toy_frame, "brent_fwd_ret_1d", ["feature_a"], ZeroForecast, min_train=200)
    assert (zero["y_pred"] == 0.0).all()

    mean = walk_forward(toy_frame, "brent_fwd_ret_1d", ["feature_a"], MeanForecast, min_train=200)
    assert mean["y_pred"].nunique() > 1  # refits over time
    assert mean["y_pred"].abs().max() < 1.0  # but stays near the sample mean


def test_evaluate_reports_a_beatable_directional_baseline():
    predictions = pd.DataFrame(
        {
            "y_true": [1.0, 1.0, 1.0, -1.0],
            "y_pred": [0.5, 0.5, 0.5, 0.5],
            "error": [0.5, 0.5, 0.5, -1.5],
        }
    )
    metrics = evaluate(predictions)

    assert metrics["directional_accuracy"] == 0.75
    # Always predicting "up" would also score 0.75, which is the point of the baseline.
    assert metrics["hit_rate_baseline"] == 0.75


def test_diebold_mariano_is_antisymmetric():
    rng = np.random.default_rng(3)
    errors_a = rng.normal(size=300)
    errors_b = rng.normal(size=300) * 1.5

    forward = diebold_mariano(errors_a, errors_b)
    backward = diebold_mariano(errors_b, errors_a)

    assert np.isclose(forward["dm_stat"], -backward["dm_stat"])


def test_diebold_mariano_detects_a_clearly_better_model():
    rng = np.random.default_rng(5)
    good = rng.normal(scale=0.5, size=400)
    bad = rng.normal(scale=2.0, size=400)

    result = diebold_mariano(good, bad)

    assert result["dm_stat"] < 0, "negative statistic means the first model has lower loss"
    assert result["p_value"] < 0.01


def test_diebold_mariano_finds_no_difference_between_equivalent_models():
    rng = np.random.default_rng(9)
    errors_a = rng.normal(size=500)
    errors_b = rng.normal(size=500)

    assert diebold_mariano(errors_a, errors_b)["p_value"] > 0.05


def test_diebold_mariano_rejects_misaligned_series():
    with pytest.raises(ValueError, match="must align"):
        diebold_mariano(np.zeros(10), np.zeros(11))


def test_diebold_mariano_returns_nan_on_tiny_samples():
    assert np.isnan(diebold_mariano(np.zeros(5), np.ones(5))["dm_stat"])
