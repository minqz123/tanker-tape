"""Price ingestion: EIA API v2 (primary), FRED (fallback), yfinance (timely, unofficial).

Target construction is deliberately split into two families and named accordingly:

* ``ret_*`` / ``rv_*`` are **backward-looking** — computable at the close of date *t* and
  therefore legal as features.
* ``fwd_ret_*`` are **forward-looking targets**. They are the thing being predicted and
  must never appear on the feature side of a model.

Nothing in this module fills gaps by interpolation. A missing price stays missing so the
downstream alignment is honest about which days actually traded.
"""

from __future__ import annotations

import datetime as dt
import io
from typing import Any

import httpx
import numpy as np
import pandas as pd

from ..config import get_settings
from ..logging_utils import get_logger, log_stage

logger = get_logger(__name__)

EIA_BASE_URL = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# EIA caps a single response; page with offset until the response runs short.
EIA_PAGE_LENGTH = 5000

EIA_SERIES = {"brent": "RBRTE", "wti": "RWTC"}
FRED_SERIES = {"brent": "DCOILBRENTEU", "wti": "DCOILWTICO"}
YFINANCE_SERIES = {"brent": "BZ=F", "wti": "CL=F"}

TRADING_DAYS_PER_YEAR = 252


def fetch_eia_spot(
    series_id: str,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Fetch a daily spot price series from the EIA API v2.

    Args:
        series_id: EIA series, e.g. ``"RBRTE"`` (Brent) or ``"RWTC"`` (WTI).
        start: Inclusive start date.
        end: Inclusive end date.
        timeout: Per-request timeout in seconds.

    Returns:
        Columns ``date``, ``value``, ``series_id``, sorted ascending by date.

    Raises:
        RuntimeError: If ``EIA_API_KEY`` is unset or the API returns an error payload.
    """
    api_key = get_settings().require("eia_api_key")
    params: dict[str, Any] = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": series_id,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": EIA_PAGE_LENGTH,
    }
    if start is not None:
        params["start"] = str(start)
    if end is not None:
        params["end"] = str(end)

    rows: list[dict[str, Any]] = []
    offset = 0
    with httpx.Client(timeout=timeout) as client:
        while True:
            response = client.get(EIA_BASE_URL, params={**params, "offset": offset})
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(f"EIA API error for {series_id}: {payload['error']}")
            page = payload.get("response", {}).get("data", [])
            rows.extend(page)
            if len(page) < EIA_PAGE_LENGTH:
                break
            offset += EIA_PAGE_LENGTH

    frame = pd.DataFrame(rows)
    if frame.empty:
        logger.warning("EIA returned no rows for series=%s start=%s end=%s", series_id, start, end)
        return pd.DataFrame(columns=["date", "value", "series_id"])

    frame = frame.rename(columns={"period": "date"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["series_id"] = series_id
    frame = frame.loc[:, ["date", "value", "series_id"]].dropna(subset=["value"])
    frame = frame.sort_values("date").reset_index(drop=True)
    log_stage(logger, "prices.eia", frame, series_id=series_id)
    return frame


def fetch_fred_spot(series_id: str, timeout: float = 60.0) -> pd.DataFrame:
    """Fetch a series from FRED's public CSV endpoint.

    This is the no-key fallback for the same underlying EIA data. ``fredapi`` is not
    required; the CSV endpoint is unauthenticated.

    Args:
        series_id: FRED series, e.g. ``"DCOILBRENTEU"``.
        timeout: Request timeout in seconds.

    Returns:
        Columns ``date``, ``value``, ``series_id``.
    """
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(FRED_CSV_URL, params={"id": series_id})
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))

    date_column, value_column = frame.columns[0], frame.columns[1]
    frame = frame.rename(columns={date_column: "date", value_column: "value"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    # FRED encodes missing observations as ".".
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["series_id"] = series_id
    frame = frame.loc[:, ["date", "value", "series_id"]].dropna(subset=["value"])
    frame = frame.sort_values("date").reset_index(drop=True)
    log_stage(logger, "prices.fred", frame, series_id=series_id)
    return frame


def fetch_yfinance_close(
    ticker: str,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
) -> pd.DataFrame:
    """Fetch futures closes from yfinance.

    yfinance is an **unofficial** scraper of Yahoo Finance. Use it for timely closes and
    intraday sanity checks only; never as the series of record in the research report.

    Args:
        ticker: Yahoo ticker, e.g. ``"BZ=F"`` for Brent front-month.
        start: Inclusive start date.
        end: Exclusive end date.

    Returns:
        Columns ``date``, ``value``, ``series_id``, ``source_is_unofficial``.
    """
    try:
        import yfinance
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "yfinance is not installed. Install the optional extra: uv sync --extra prices"
        ) from exc

    logger.warning(
        "Using UNOFFICIAL source yfinance for ticker=%s - not for the series of record", ticker
    )
    raw = yfinance.download(
        ticker, start=start, end=end, progress=False, auto_adjust=False, multi_level_index=False
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "value", "series_id", "source_is_unofficial"])

    frame = raw.reset_index().rename(columns={"Date": "date", "Close": "value"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["series_id"] = ticker
    frame["source_is_unofficial"] = True
    frame = frame.loc[:, ["date", "value", "series_id", "source_is_unofficial"]]
    frame = frame.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
    log_stage(logger, "prices.yfinance", frame, ticker=ticker, unofficial=True)
    return frame


def fetch_spot(
    commodity: str,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
    prefer: str = "eia",
) -> pd.DataFrame:
    """Fetch a spot series with automatic fallback.

    Args:
        commodity: ``"brent"`` or ``"wti"``.
        start: Inclusive start date.
        end: Inclusive end date.
        prefer: ``"eia"`` or ``"fred"``.

    Returns:
        Columns ``date``, ``value``, ``series_id``, ``source``.
    """
    if commodity not in EIA_SERIES:
        raise ValueError(f"unknown commodity {commodity!r}; expected one of {sorted(EIA_SERIES)}")

    if prefer == "eia":
        try:
            frame = fetch_eia_spot(EIA_SERIES[commodity], start=start, end=end)
            frame["source"] = "eia"
            return frame
        except Exception as exc:  # noqa: BLE001 - fallback is the point
            logger.warning("EIA fetch failed for %s (%s); falling back to FRED", commodity, exc)

    frame = fetch_fred_spot(FRED_SERIES[commodity])
    frame["source"] = "fred"
    if start is not None:
        frame = frame[frame["date"] >= pd.to_datetime(start).date()]
    if end is not None:
        frame = frame[frame["date"] <= pd.to_datetime(end).date()]
    return frame.reset_index(drop=True)


def add_return_features(
    frame: pd.DataFrame,
    price_column: str,
    prefix: str,
    horizons: tuple[int, ...] = (1, 5, 20),
    vol_window: int = 20,
) -> pd.DataFrame:
    """Add backward-looking returns/volatility and forward-looking targets.

    Backward columns (``{prefix}_ret_{h}d``, ``{prefix}_rv_{w}d``) use only data up to and
    including the row's date, so they are safe as features. Forward columns
    (``{prefix}_fwd_ret_{h}d``, ``{prefix}_fwd_rv_{h}d``) are targets and are unsafe as
    features by construction.

    Volatility targets exist because they are the more plausible place for this project
    to find anything. A closed strait is a dispersion event as much as a level event,
    and volatility is more persistent than direction, so it survives a publication lag
    that would destroy a directional signal.

    Args:
        frame: Frame sorted ascending by date.
        price_column: Column holding the price level.
        prefix: Column name prefix, e.g. ``"brent"``.
        horizons: Return horizons in trading days.
        vol_window: Window for realized volatility, in trading days.

    Returns:
        A copy of ``frame`` with the added columns.
    """
    out = frame.copy()
    log_price = np.log(out[price_column])

    for horizon in horizons:
        out[f"{prefix}_ret_{horizon}d"] = log_price.diff(horizon)
        out[f"{prefix}_fwd_ret_{horizon}d"] = log_price.shift(-horizon) - log_price

    daily = log_price.diff()
    out[f"{prefix}_rv_{vol_window}d"] = daily.rolling(vol_window).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )

    for horizon in horizons:
        if horizon < 2:
            # Realized volatility over a single day is just that day's absolute
            # return; it carries no dispersion information worth forecasting.
            continue
        # std of the daily returns in (t, t+h]: roll over h days, then shift the
        # window back so it lands on t.
        out[f"{prefix}_fwd_rv_{horizon}d"] = daily.rolling(horizon).std().shift(-horizon) * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )

    return out


def build_price_panel(
    start: dt.date | str | None = "2015-01-01",
    end: dt.date | str | None = None,
    prefer: str = "eia",
) -> pd.DataFrame:
    """Build the daily price panel with Brent/WTI levels, returns, vol, and the spread.

    Args:
        start: Inclusive start date.
        end: Inclusive end date.
        prefer: Preferred spot source, ``"eia"`` or ``"fred"``.

    Returns:
        One row per date on which at least one series traded.
    """
    brent = fetch_spot("brent", start=start, end=end, prefer=prefer)
    wti = fetch_spot("wti", start=start, end=end, prefer=prefer)

    panel = (
        brent.loc[:, ["date", "value"]]
        .rename(columns={"value": "brent_spot"})
        .merge(
            wti.loc[:, ["date", "value"]].rename(columns={"value": "wti_spot"}),
            on="date",
            how="outer",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    panel = add_return_features(panel, "brent_spot", "brent")
    panel = add_return_features(panel, "wti_spot", "wti")
    panel["brent_wti_spread"] = panel["brent_spot"] - panel["wti_spot"]
    # Backward change is a legal feature; the forward change is a target. The spread is
    # where a Gulf-specific supply shock should show up most cleanly, because it is
    # differenced against a US benchmark that the same shock does not touch.
    panel["brent_wti_spread_chg_1d"] = panel["brent_wti_spread"].diff()
    for horizon in (1, 5, 20):
        panel[f"brent_wti_spread_fwd_chg_{horizon}d"] = (
            panel["brent_wti_spread"].shift(-horizon) - panel["brent_wti_spread"]
        )

    log_stage(
        logger,
        "prices.panel",
        panel,
        brent_missing=int(panel["brent_spot"].isna().sum()),
        wti_missing=int(panel["wti_spot"].isna().sum()),
    )
    return panel
