"""Walk-forward forecasting with honest benchmarks.

This is the module that decides whether the project's headline claim survives. Three
design choices carry that weight:

* **Expanding window, refit out-of-sample.** The model predicting date *t* has never seen
  date *t*.
* **An embargo equal to the forecast horizon.** A 20-day-ahead target at date *t* is only
  observable at *t+20*, so when predicting date *i* the training set stops at ``i - h``.
  Without this the model trains on targets that overlap the thing it is predicting, which
  is the most common way a walk-forward backtest silently cheats.
* **Scaling inside the fold.** The scaler is fit on training rows only, via a Pipeline.

Benchmarks are not optional. "AIS features beat nothing" is not a finding; "AIS features
beat an AR(p) on price history alone, by a margin a Diebold-Mariano test can distinguish
from zero" is.

The expected result is that AIS adds little for daily direction and more for volatility
and spreads, concentrated in regime shifts. Report that outcome as readily as any other.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..logging_utils import get_logger, log_stage

logger = get_logger(__name__)


class Estimator(Protocol):
    """Minimal sklearn-compatible interface."""

    def fit(self, X: Any, y: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...


def ridge_model(alpha: float = 1.0) -> Pipeline:
    """Standardised ridge. Scaling lives in the pipeline so it is fit per fold."""
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha, random_state=None))])


def lightgbm_model(**kwargs: Any) -> Any:
    """LightGBM regressor with conservative defaults for a short, noisy sample.

    Raises:
        RuntimeError: If the optional ``research`` extra is not installed.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "lightgbm is not installed. Install the optional extra: uv sync --extra research"
        ) from exc

    defaults = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }
    return LGBMRegressor(**{**defaults, **kwargs})


@dataclass
class ZeroForecast:
    """Random-walk benchmark: the best guess for tomorrow's return is zero."""

    def fit(self, X: Any, y: Any) -> ZeroForecast:
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X))


@dataclass
class MeanForecast:
    """Historical-mean benchmark, fit on the training fold only."""

    mean_: float = 0.0

    def fit(self, X: Any, y: Any) -> MeanForecast:
        self.mean_ = float(np.nanmean(y)) if len(y) else 0.0
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), self.mean_)


def walk_forward(
    frame: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    model_factory: Callable[[], Estimator],
    horizon: int = 1,
    min_train: int = 500,
    refit_every: int = 21,
    date_column: str = "date",
) -> pd.DataFrame:
    """Expanding-window walk-forward forecast with a horizon-length embargo.

    Args:
        frame: Feature table sorted ascending by date.
        target: Forward-return column, e.g. ``"brent_fwd_ret_5d"``.
        feature_columns: Feature columns. Must not contain any ``fwd_`` column.
        model_factory: Callable returning a fresh unfitted estimator.
        horizon: Forecast horizon in trading days; also the embargo length.
        min_train: Minimum training rows before the first prediction.
        refit_every: Refit cadence in rows. The model still predicts every row; only
            refitting is throttled, which is a compute trade-off, not a leakage one.
        date_column: Date column name.

    Returns:
        Columns ``date``, ``y_true``, ``y_pred``, ``error``, ``n_train``.

    Raises:
        ValueError: If a forward-looking column is passed as a feature.
    """
    leaking = [column for column in feature_columns if "fwd_" in column]
    if leaking:
        raise ValueError(
            f"these look like forward-looking targets, not features: {leaking}. "
            "Using them would make the backtest meaningless."
        )

    data = frame.sort_values(date_column).reset_index(drop=True)
    usable = data.loc[:, [date_column, target, *feature_columns]].copy()
    valid = usable.dropna().reset_index(drop=True)

    dropped = len(usable) - len(valid)
    if len(valid) <= min_train + horizon:
        raise ValueError(
            f"not enough complete rows to walk forward: {len(valid)} usable, need more than "
            f"min_train ({min_train}) + horizon ({horizon}). Check for features that are "
            "mostly NaN."
        )

    features = valid.loc[:, feature_columns].to_numpy(dtype="float64")
    targets = valid[target].to_numpy(dtype="float64")
    dates = valid[date_column].to_numpy()

    predictions: list[dict[str, Any]] = []
    model: Estimator | None = None
    last_fit_at = -1

    for position in range(min_train, len(valid)):
        # Embargo: the target for row `position - horizon` is only observable at
        # `position`, so training must stop before it.
        train_end = position - horizon
        if train_end < min_train // 2:
            continue

        if model is None or (position - last_fit_at) >= refit_every:
            model = model_factory()
            model.fit(features[:train_end], targets[:train_end])
            last_fit_at = position

        prediction = float(np.asarray(model.predict(features[position : position + 1]))[0])
        predictions.append(
            {
                "date": dates[position],
                "y_true": targets[position],
                "y_pred": prediction,
                "error": targets[position] - prediction,
                "n_train": train_end,
            }
        )

    result = pd.DataFrame(predictions)
    log_stage(
        logger,
        "forecast.walk_forward",
        result,
        target=target,
        horizon=horizon,
        n_features=len(feature_columns),
        rows_dropped_for_nan=dropped,
    )
    return result


def evaluate(predictions: pd.DataFrame) -> dict[str, float]:
    """Score a walk-forward result.

    Directional accuracy ignores rows where the realised return is exactly zero, which
    would otherwise be scored arbitrarily.

    Returns:
        ``rmse``, ``mae``, ``directional_accuracy``, ``n``, and ``hit_rate_baseline``
        (the accuracy of always predicting the majority direction).
    """
    if predictions.empty:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "directional_accuracy": float("nan"),
            "n": 0.0,
            "hit_rate_baseline": float("nan"),
        }

    errors = predictions["error"].to_numpy(dtype="float64")
    truth = predictions["y_true"].to_numpy(dtype="float64")
    predicted = predictions["y_pred"].to_numpy(dtype="float64")

    directional = truth != 0
    correct = np.sign(predicted[directional]) == np.sign(truth[directional])
    up_share = float((truth[directional] > 0).mean()) if directional.any() else float("nan")

    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "directional_accuracy": float(correct.mean()) if directional.any() else float("nan"),
        "n": float(len(predictions)),
        # Always-predict-the-majority-direction accuracy. Beat this, not 50%.
        "hit_rate_baseline": max(up_share, 1 - up_share) if np.isfinite(up_share) else float("nan"),
    }


def diebold_mariano(
    errors_model: np.ndarray | pd.Series,
    errors_benchmark: np.ndarray | pd.Series,
    horizon: int = 1,
    loss: str = "squared",
    harvey_correction: bool = True,
) -> dict[str, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    The null is that the two forecasts have equal expected loss. A negative statistic with
    a small p-value means the first model has lower loss.

    Overlapping multi-step forecasts are autocorrelated, so the long-run variance uses a
    Newey-West estimator with ``horizon - 1`` lags. The Harvey-Leybourne-Newbold
    small-sample correction is applied by default; without it the test over-rejects on the
    short samples this project has.

    Args:
        errors_model: Forecast errors of the candidate model.
        errors_benchmark: Forecast errors of the benchmark.
        horizon: Forecast horizon, used for the HAC lag truncation.
        loss: ``"squared"`` or ``"absolute"``.
        harvey_correction: Apply the HLN small-sample correction.

    Returns:
        ``dm_stat``, ``p_value``, ``mean_loss_differential``, ``n``.

    Raises:
        ValueError: If the error series differ in length or ``loss`` is unknown.
    """
    model = np.asarray(errors_model, dtype="float64")
    benchmark = np.asarray(errors_benchmark, dtype="float64")
    if model.shape != benchmark.shape:
        raise ValueError(
            f"error series must align: got {model.shape} and {benchmark.shape}. "
            "Compare forecasts over the same dates."
        )

    if loss == "squared":
        differential = model**2 - benchmark**2
    elif loss == "absolute":
        differential = np.abs(model) - np.abs(benchmark)
    else:
        raise ValueError(f"unknown loss {loss!r}; expected 'squared' or 'absolute'")

    differential = differential[np.isfinite(differential)]
    n = len(differential)
    if n < 10:
        return {
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "mean_loss_differential": float("nan"),
            "n": float(n),
        }

    mean_differential = float(np.mean(differential))
    centred = differential - mean_differential

    # Newey-West long-run variance with Bartlett weights.
    long_run_variance = float(np.dot(centred, centred) / n)
    for lag in range(1, horizon):
        if lag >= n:
            break
        autocovariance = float(np.dot(centred[lag:], centred[:-lag]) / n)
        weight = 1.0 - lag / horizon
        long_run_variance += 2.0 * weight * autocovariance

    if long_run_variance <= 0:
        return {
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "mean_loss_differential": mean_differential,
            "n": float(n),
        }

    dm_stat = mean_differential / np.sqrt(long_run_variance / n)

    if harvey_correction:
        correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        dm_stat *= correction

    p_value = float(2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1)))
    return {
        "dm_stat": float(dm_stat),
        "p_value": p_value,
        "mean_loss_differential": mean_differential,
        "n": float(n),
    }


def compare_models(
    frame: pd.DataFrame,
    target: str,
    price_features: list[str],
    ais_features: list[str],
    horizon: int = 1,
    min_train: int = 500,
    model_factory: Callable[[], Estimator] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run the full benchmark ladder for one target and horizon.

    The ladder is: zero forecast (random walk), training-mean forecast, price features
    only, then price + AIS features. The question the project exists to answer is whether
    the last step beats the one before it, so that comparison gets the Diebold-Mariano
    test.

    Args:
        frame: Feature table.
        target: Forward-return target column.
        price_features: Price-only feature columns.
        ais_features: AIS-derived feature columns.
        horizon: Forecast horizon in trading days.
        min_train: Minimum training rows.
        model_factory: Estimator factory for the two feature-based models; defaults to ridge.

    Returns:
        ``(scoreboard, predictions_by_model)``. The scoreboard is sorted by RMSE and
        carries the DM test of price+AIS against price-only.
    """
    factory = model_factory or ridge_model
    all_features = price_features + ais_features

    runs = {
        "zero_rw": walk_forward(frame, target, price_features, ZeroForecast, horizon, min_train),
        "train_mean": walk_forward(frame, target, price_features, MeanForecast, horizon, min_train),
        "price_only": walk_forward(frame, target, price_features, factory, horizon, min_train),
        "price_plus_ais": walk_forward(frame, target, all_features, factory, horizon, min_train),
    }

    rows = []
    for name, predictions in runs.items():
        metrics = evaluate(predictions)
        metrics["model"] = name
        rows.append(metrics)
    scoreboard = pd.DataFrame(rows).set_index("model")

    # Adding AIS features drops rows wherever they are NaN, so the models can end up
    # scored over different dates. The DM test below merges on date and is unaffected,
    # but the raw RMSE column is then not a like-for-like comparison.
    sample_sizes = scoreboard["n"].unique()
    if len(sample_sizes) > 1:
        logger.warning(
            "models were scored over different sample sizes %s - compare the RMSE column "
            "with care and rely on the Diebold-Mariano test, which aligns on date.",
            sorted(int(size) for size in sample_sizes),
        )

    # The comparison that matters: does adding AIS beat price history alone?
    aligned = runs["price_plus_ais"].merge(
        runs["price_only"], on="date", suffixes=("_ais", "_price")
    )
    if len(aligned) >= 10:
        test = diebold_mariano(aligned["error_ais"], aligned["error_price"], horizon=horizon)
        scoreboard.loc["price_plus_ais", "dm_stat_vs_price_only"] = test["dm_stat"]
        scoreboard.loc["price_plus_ais", "dm_p_vs_price_only"] = test["p_value"]
        logger.info(
            "stage=forecast.compare target=%s horizon=%d dm_stat=%.3f p=%.3f "
            "(negative stat + small p => AIS helps)",
            target,
            horizon,
            test["dm_stat"],
            test["p_value"],
        )
    else:
        logger.warning("not enough aligned predictions for a Diebold-Mariano test")

    scoreboard = scoreboard.sort_values("rmse")
    log_stage(logger, "forecast.compare_models", scoreboard.reset_index(), target=target)
    return scoreboard, runs
