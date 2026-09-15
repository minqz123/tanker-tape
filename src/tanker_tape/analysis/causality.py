"""Stationarity, VAR, Granger causality, and Jordà local projections.

Two things in here are easy to get wrong and are handled explicitly:

* **Direction.** Granger tests run in *both* directions by default. Prior work finds oil
  prices drive tanker routing decisions as much as the reverse, so a one-directional test
  that "confirms" AIS leads prices is very likely just measuring the feedback loop.
* **Levels vs changes.** Transit counts and price levels are non-stationary. Everything
  here expects changes, returns, or z-scores; :func:`stationarity_report` is the check you
  run first, not a formality.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss

from ..logging_utils import get_logger, log_stage

logger = get_logger(__name__)


def stationarity_report(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Run ADF and KPSS on each column.

    The two tests have opposite null hypotheses, which is the point: ADF's null is a unit
    root, KPSS's null is stationarity. Agreement is informative; disagreement usually means
    the series is fractionally integrated or has a structural break — and this sample has
    an obvious structural break in it.

    Returns:
        Columns ``column``, ``n_obs``, ``adf_stat``, ``adf_p``, ``kpss_stat``, ``kpss_p``,
        ``verdict``.
    """
    rows = []
    for column in columns:
        series = frame[column].dropna()
        if len(series) < 20:
            logger.warning("skipping stationarity test for %r: only %d obs", column, len(series))
            continue
        if series.nunique() <= 1:
            # ADF and KPSS both raise on a constant series. A constant column is a
            # real thing to find in this data - a chokepoint with no recorded
            # transits all period - so report it rather than crashing the report.
            logger.warning(
                "column %r is constant (value=%s); reporting it as such rather than testing",
                column,
                series.iloc[0],
            )
            rows.append(
                {
                    "column": column,
                    "n_obs": len(series),
                    "adf_stat": float("nan"),
                    "adf_p": float("nan"),
                    "kpss_stat": float("nan"),
                    "kpss_p": float("nan"),
                    "verdict": "constant",
                }
            )
            continue

        with warnings.catch_warnings():
            # adfuller warns about a future return-type change; KPSS warns when its
            # p-value is clipped at the edge of the lookup table. Neither affects the
            # statistics read here.
            warnings.simplefilter("ignore")
            adf_stat, adf_p = adfuller(series, autolag="AIC")[:2]
            kpss_stat, kpss_p = kpss(series, regression="c", nlags="auto")[:2]

        adf_stationary = adf_p < 0.05
        kpss_stationary = kpss_p > 0.05
        if adf_stationary and kpss_stationary:
            verdict = "stationary"
        elif not adf_stationary and not kpss_stationary:
            verdict = "non_stationary"
        else:
            verdict = "inconclusive"

        rows.append(
            {
                "column": column,
                "n_obs": len(series),
                "adf_stat": adf_stat,
                "adf_p": adf_p,
                "kpss_stat": kpss_stat,
                "kpss_p": kpss_p,
                "verdict": verdict,
            }
        )

    report = pd.DataFrame(rows)
    if not report.empty:
        non_stationary = report.loc[report["verdict"] != "stationary", "column"].tolist()
        if non_stationary:
            logger.warning(
                "these columns are not clearly stationary and should be differenced before "
                "VAR/Granger: %s",
                non_stationary,
            )
    log_stage(logger, "causality.stationarity", report)
    return report


def select_var_lag(
    frame: pd.DataFrame,
    columns: list[str],
    max_lags: int = 10,
    criterion: str = "aic",
) -> int:
    """Choose a VAR lag order by information criterion.

    Returns:
        The selected lag order (at least 1).
    """
    data = frame.loc[:, columns].dropna()
    if len(data) < 3 * (len(columns) + 1):
        raise ValueError(
            f"only {len(data)} complete rows across {columns} — too few to fit a VAR. "
            "A sparse weekly feature joined to daily prices is the usual cause; make sure "
            "build_feature_table's forward_fill is on before differencing."
        )

    # statsmodels raises an opaque error if the largest candidate model is unidentifiable,
    # so clamp the search to what this sample can actually support.
    feasible = max(1, len(data) // (len(columns) + 1) - 1)
    if feasible < max_lags:
        logger.warning(
            "max_lags reduced from %d to %d: %d observations across %d equations cannot "
            "identify the larger model",
            max_lags,
            feasible,
            len(data),
            len(columns),
        )
        max_lags = feasible

    model = VAR(data)
    selection = model.select_order(maxlags=max_lags)
    chosen = int(getattr(selection, criterion) or 1)
    logger.info(
        "stage=causality.select_lag criterion=%s chosen=%d aic=%s bic=%s n_obs=%d",
        criterion,
        chosen,
        selection.aic,
        selection.bic,
        len(data),
    )
    return max(chosen, 1)


def fit_var(frame: pd.DataFrame, columns: list[str], lags: int | None = None, max_lags: int = 10):
    """Fit a VAR on the given columns.

    Args:
        frame: Data frame containing the columns.
        columns: Endogenous variables.
        lags: Lag order; selected by AIC when omitted.
        max_lags: Upper bound for lag selection.

    Returns:
        The fitted ``VARResults``.
    """
    data = frame.loc[:, columns].dropna()
    order = lags if lags is not None else select_var_lag(frame, columns, max_lags=max_lags)
    results = VAR(data).fit(order)
    logger.info(
        "stage=causality.fit_var lags=%d n_obs=%d columns=%s", order, len(data), ",".join(columns)
    )
    return results


def granger_both_directions(
    frame: pd.DataFrame,
    series_a: str,
    series_b: str,
    max_lag: int = 5,
) -> pd.DataFrame:
    """Test Granger causality in both directions.

    Reporting only the direction that produced a small p-value is the main way this
    analysis goes wrong, so both are always returned.

    Args:
        frame: Frame containing both columns.
        series_a: First series name.
        series_b: Second series name.
        max_lag: Maximum lag to test.

    Returns:
        Columns ``cause``, ``effect``, ``lag``, ``f_stat``, ``p_value``.
    """
    data = frame.loc[:, [series_a, series_b]].dropna()
    rows = []

    for cause, effect in ((series_a, series_b), (series_b, series_a)):
        # statsmodels tests whether column 2 Granger-causes column 1.
        ordered = data.loc[:, [effect, cause]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = grangercausalitytests(ordered, maxlag=max_lag)
        for lag, (tests, _) in results.items():
            f_stat, p_value = tests["ssr_ftest"][0], tests["ssr_ftest"][1]
            rows.append(
                {
                    "cause": cause,
                    "effect": effect,
                    "lag": lag,
                    "f_stat": f_stat,
                    "p_value": p_value,
                }
            )

    report = pd.DataFrame(rows)
    log_stage(logger, "causality.granger", report, n_obs=len(data), max_lag=max_lag)

    both = report[report["p_value"] < 0.05].groupby("lag")["cause"].nunique()
    if (both > 1).any():
        logger.warning(
            "Granger causality is significant in BOTH directions at lag(s) %s - this is "
            "feedback, not evidence that AIS leads prices.",
            both[both > 1].index.tolist(),
        )
    return report


def local_projection(
    frame: pd.DataFrame,
    outcome: str,
    shock: str,
    horizons: int = 20,
    controls: list[str] | None = None,
    lags: int = 2,
    newey_west_lags: int | None = None,
) -> pd.DataFrame:
    """Jordà local projections: impulse response of ``outcome`` to a ``shock``.

    For each horizon *h* this estimates

        ``y[t+h] = alpha + beta_h * shock[t] + controls + lags + e[t+h]``

    and reports ``beta_h``. Overlapping horizons make the residuals autocorrelated by
    construction, so standard errors are Newey-West corrected.

    Args:
        frame: Data frame with the outcome, shock, and any controls.
        outcome: Outcome series name (a return or change, not a level).
        shock: Shock series name.
        horizons: Maximum horizon in periods.
        controls: Extra contemporaneous controls.
        lags: Number of own-lags of outcome and shock to include.
        newey_west_lags: HAC lag truncation; defaults to ``horizon + 1``.

    Returns:
        Columns ``horizon``, ``beta``, ``std_err``, ``t_stat``, ``p_value``,
        ``ci_low``, ``ci_high``, ``n_obs``.
    """
    controls = controls or []
    data = frame.copy()

    regressors = [shock]
    for lag in range(1, lags + 1):
        data[f"_{outcome}_lag{lag}"] = data[outcome].shift(lag)
        data[f"_{shock}_lag{lag}"] = data[shock].shift(lag)
        regressors.extend([f"_{outcome}_lag{lag}", f"_{shock}_lag{lag}"])
    regressors.extend(controls)

    rows = []
    for horizon in range(horizons + 1):
        data["_target"] = data[outcome].shift(-horizon)
        sample = data.loc[:, ["_target", *regressors]].dropna()
        if len(sample) < 30:
            logger.warning("horizon %d has only %d usable rows; skipping", horizon, len(sample))
            continue

        exog = sm.add_constant(sample.loc[:, regressors])
        hac_lags = newey_west_lags if newey_west_lags is not None else horizon + 1
        fitted = sm.OLS(sample["_target"], exog).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

        beta = float(fitted.params[shock])
        std_err = float(fitted.bse[shock])
        rows.append(
            {
                "horizon": horizon,
                "beta": beta,
                "std_err": std_err,
                "t_stat": float(fitted.tvalues[shock]),
                "p_value": float(fitted.pvalues[shock]),
                "ci_low": beta - 1.96 * std_err,
                "ci_high": beta + 1.96 * std_err,
                "n_obs": len(sample),
            }
        )

    report = pd.DataFrame(rows)
    log_stage(logger, "causality.local_projection", report, outcome=outcome, shock=shock)
    return report


def rolling_correlation(
    frame: pd.DataFrame,
    series_a: str,
    series_b: str,
    windows: tuple[int, ...] = (60, 120),
) -> pd.DataFrame:
    """Rolling correlations, to show how regime-dependent the relationship is.

    Returns:
        Columns ``date`` plus ``corr_{window}d`` for each window.
    """
    out = frame.loc[:, ["date"]].copy()
    for window in windows:
        out[f"corr_{window}d"] = (
            frame[series_a].rolling(window, min_periods=window // 2).corr(frame[series_b])
        )
    log_stage(logger, "causality.rolling_correlation", out, windows=",".join(map(str, windows)))
    return out


def rolling_beta(
    frame: pd.DataFrame,
    outcome: str,
    predictor: str,
    window: int = 120,
) -> pd.Series:
    """Rolling univariate OLS beta of ``outcome`` on ``predictor``."""
    covariance = frame[outcome].rolling(window, min_periods=window // 2).cov(frame[predictor])
    variance = frame[predictor].rolling(window, min_periods=window // 2).var()
    return covariance / variance.replace(0.0, np.nan)
