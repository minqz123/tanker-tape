# Project Brief: Brent Crude × AIS Shipping Signals ("Tanker Tape")

Hand-off document for Claude Code. Read fully before writing code. Where this brief names an endpoint, schema, or ID, **verify it against live docs before hardcoding** — several were gathered from web research on 2026-09-15 and may have changed.

---

## 1. Goal

Build a reproducible research + monitoring system that measures how physical oil flows observed in AIS (Automatic Identification System) ship-tracking data relate to Brent crude prices, and tests whether AIS-derived signals carry *incremental, out-of-sample* information about Brent returns and volatility.

Three deliverables:
1. **Data pipeline** — historical + live AIS-derived shipping metrics joined to daily Brent/WTI prices in a local analytical store.
2. **Research notebook/report** — event study, causality tests, and walk-forward forecasting with honest benchmarks.
3. **Dashboard** — chokepoint status, tanker flow metrics, and Brent overlay, updated automatically.

This is a research/portfolio project, not a trading system. Every output must say so.

## 2. Why now (market context as of mid-Sept 2026)

The 2026 Strait of Hormuz crisis is the natural experiment that makes this project interesting:

- The US-Israel war on Iran began in late February 2026; Hormuz traffic nearly stopped and Brent jumped from roughly $70 (Feb avg) to ~$94 by March 9 and briefly near $120 in March.
- A US–Iran memorandum to reopen Hormuz was signed June 17 and collapsed; the strait was effectively closed again in early July (IEA: Gulf loadings fell from ~20 mb/d to ~12 mb/d during July).
- Saudi Arabia's East-West pipeline was shut after drone strikes on Sept 10–11; Oman postponed GCC–Iran Hormuz talks; Brent traded ~$106–108 on Sept 14.
- IMF PortWatch explicitly warns of GPS jamming, AIS spoofing, and vessels going dark around Hormuz. **Data quality handling is a first-class requirement, not an afterthought.**

Encode these as a dated event table (`data/reference/events.csv`: date, event, category, source_url) — Claude Code should research and verify exact dates rather than trusting this summary.

## 3. Data sources

### 3.1 Prices (target variables)
| Source | Series | Notes |
|---|---|---|
| EIA API v2 | `petroleum/pri/spt`, series `RBRTE` (Brent spot), `RWTC` (WTI) | Free API key. Daily FOB spot back to 1987. Publishes with a lag. |
| FRED | `DCOILBRENTEU`, `DCOILWTICO` | Same EIA data; easy fallback via `fredapi` or CSV. |
| yfinance | `BZ=F` (Brent front-month), `CL=F` | Unofficial; use for timely closes and intraday checks only. Log that it's unofficial. |
| Optional | Tanker equities (FRO, DHT, INSW, TNK, STNG), `BNO` ETF | Secondary targets: do AIS signals explain tanker stocks better than crude? |

Derived targets: log returns (1d, 5d, 20d), realized volatility (20d), Brent–WTI spread, and, if a second-month contract can be sourced, the M1–M2 spread (backwardation is where supply shocks show up most clearly).

### 3.2 Historical AIS-derived data (the backbone)
**IMF PortWatch** (free, no key) — built on UN Global Platform AIS covering ~90k ships.
- *Daily Chokepoint Transit Calls & Trade Volume Estimates*: 28 chokepoints, by vessel type (tanker, container, dry bulk, etc.) with capacity estimates. Hormuz is `chokepoint6`.
- *Daily Port Activity & Trade Estimates*: ~2,065 ports, port calls and import/export volume estimates by vessel type.
- Updated **weekly, Tuesdays 9 AM ET**; data is **revised** (sampling changes, spoofing checks). Exposed via ArcGIS open-data pages with GeoServices/FeatureServer query endpoints — discover the exact REST URL from the dataset "API" tab at portwatch.imf.org/pages/data-and-methodology rather than guessing.

Chokepoints to pull at minimum: Hormuz, Bab el-Mandeb, Suez Canal, Cape of Good Hope, Malacca, Bosporus, Panama. Ports: major crude export terminals (e.g., Ras Tanura, Juaymah, Yanbu, Fujairah, Basrah, Kharg if present, Novorossiysk, Primorsk, Corpus Christi, Houston). Confirm port IDs from the PortWatch ports table.

### 3.3 Live raw AIS (our own collection)
**aisstream.io** (free API key via GitHub signup)
- WebSocket only: `wss://stream.aisstream.io/v0/stream`. Send a JSON subscription within 3 seconds: `APIKey`, `BoundingBoxes`, optional `FiltersShipMMSI`, `FilterMessageTypes` (use `PositionReport`, `ShipStaticData`, plus Class B equivalents).
- **Server-side only** (browser connections not permitted). Negotiate permessage-deflate; consume fast or messages are dropped. Reconnect with exponential backoff.
- No history — every day not collected is lost. **Start the collector in week 1.**
- Coverage is largely terrestrial receivers; Gulf coverage may be patchy. Validate our counts against PortWatch before trusting them.
- Ship type codes (ITU-R M.1371): tankers 80–89, cargo 70–79.

### 3.4 Other free historical raw AIS (optional, for method development)
NOAA MarineCadastre (US waters) and the Danish Maritime Authority publish free historical AIS CSVs. Useful for building/testing the transit-detection and laden-state logic on dense data before applying it to the live Gulf feed.

### 3.5 Paid upgrades (document only; do not build against)
Kpler, Vortexa, Spire Maritime, MarineTraffic, Signal Ocean: satellite AIS, cargo/volume attribution, dark-fleet detection. Design the ingestion layer with a provider interface so one could be swapped in.

## 4. Architecture

```
tanker-tape/
├── CLAUDE.md                 # condensed version of this brief + conventions
├── pyproject.toml            # uv-managed, Python 3.11+
├── .env.example              # EIA_API_KEY, FRED_API_KEY, AISSTREAM_API_KEY
├── src/tanker_tape/
│   ├── config.py             # pydantic-settings; zones/gates/ports defined in YAML
│   ├── ingest/
│   │   ├── prices.py         # EIA, FRED, yfinance
│   │   ├── portwatch.py      # paginated FeatureServer pulls, stores vintage
│   │   └── aisstream.py      # async websocket collector
│   ├── process/
│   │   ├── transits.py       # gate-line crossing detection
│   │   ├── vessel_state.py   # laden/ballast, waiting/anchored, dark gaps
│   │   └── features.py       # daily feature table
│   ├── analysis/
│   │   ├── event_study.py
│   │   ├── causality.py      # VAR, Granger, local projections
│   │   └── forecast.py       # walk-forward models + benchmarks
│   └── dashboard/app.py      # Streamlit
├── data/{raw,interim,processed,reference}/   # gitignored except reference
├── notebooks/                # exploration only; logic lives in src/
├── reports/                  # generated HTML/MD research report
└── tests/
```

Stack: pandas or polars, DuckDB over Parquet, statsmodels, scikit-learn, LightGBM, `websockets` (asyncio), shapely/geopandas for geometry, Streamlit + pydeck/plotly for the dashboard, pytest, ruff. Schedule with cron/systemd on an always-on box (cheap VPS or Raspberry Pi) for the collector; GitHub Actions is fine for the weekly PortWatch/price pulls.

## 5. Core logic specifications

**Transit detection.** Define each chokepoint as a gate line (shapely LineString) plus a surrounding zone polygon. A transit = consecutive position reports for one MMSI on opposite sides of the gate within a max time gap (e.g., 6h). Record direction (inbound/outbound). Deduplicate by MMSI+IMO; handle MMSI changes by joining on IMO from static data.

**Laden/ballast.** Laden ratio = reported draught / max draught observed for that vessel (or class-typical max by length). Flag laden if above a threshold (start ~0.75, calibrate). Draught is manually entered and often stale — treat as noisy, track a `draught_last_updated` age.

**Queue/waiting.** Count tankers in zone with SOG < 1 knot for > 6h, split anchored vs drifting. This "waiting fleet" is likely the most informative live metric during a closure.

**Data-quality flags.** Position jumps implying impossible speed (> 40 kn), vessels on land, identical positions for many MMSIs (spoofing clusters), and AIS gaps > 12h inside the zone ("went dark"). Never silently drop — flag and report counts daily.

**Vintage tracking.** Store each PortWatch pull with a `retrieved_at` timestamp. Analyses must be able to run on *as-of* data to avoid using revised values that weren't available at the time.

## 6. Feature table (daily, one row per date)

For each chokepoint/port group: tanker transit count, tanker capacity (DWT), laden share, waiting-fleet count, dark-gap count, and deviations from trailing baselines (z-score vs 28d and 90d, and vs same weeks prior year). Cross-route features: Cape of Good Hope vs Suez tanker share (rerouting), Gulf export-port calls total. Plus event dummies from `events.csv`. All features must be computable using only information available at the row's date (respect PortWatch's weekly publication lag).

## 7. Analysis plan (in this order)

1. **Descriptives & sanity checks.** Plot Hormuz tanker transits vs Brent 2019–present with event markers. Compare our aisstream counts to PortWatch on overlapping days; report correlation and bias.
2. **Stationarity.** ADF/KPSS on levels and changes; model returns and transit changes/z-scores, not raw levels.
3. **Event study.** Brent abnormal returns and transit changes in windows around each dated event (−5 to +20 trading days). Pre-war vs war regimes separately.
4. **Dynamic relationships.** VAR with lag selection by information criteria; Granger causality in *both* directions (prior research finds oil prices also drive tanker port-call behavior — reverse causality is expected). Jordà local projections for the impulse response of Brent returns to a transit shock.
5. **Rolling analysis.** 60/120-day rolling correlations and betas to show regime dependence.
6. **Forecasting (the honest test).** Walk-forward, expanding window, horizons 1d/5d/20d. Benchmarks: random walk / zero-return, AR(p), and price-only features. Candidates: ridge, LightGBM with AIS features added. Metrics: RMSE, MAE, directional accuracy, Diebold–Mariano test vs benchmark, plus volatility-forecast accuracy. Report results even if AIS adds nothing — markets price public information fast, and the likely finding is that AIS helps more for volatility/spreads and during regime shifts than for daily direction.
7. **Optional toy backtest.** Only if step 6 shows significance: simple rule on BZ=F with transaction costs and slippage, clearly labeled hypothetical.

Leakage checklist to enforce in code: no future baseline windows, no full-sample scaling, publication lags applied, as-of vintages used, hyperparameters tuned only inside training folds.

## 8. Dashboard

Streamlit app with: map of monitored zones with latest vessel positions (color by type/laden state); chokepoint status cards (today vs baseline, colored by z-score); Brent (spot + futures) overlaid with Hormuz tanker transits and event markers; waiting-fleet time series; data-quality panel (dark gaps, spoofing flags, collector uptime); a footer disclaimer that this is research, not investment advice.

## 9. Milestones

1. **Week 1:** scaffold repo, price ingestion, aisstream collector running 24/7 on Hormuz + Bab el-Mandeb + Suez boxes, writing Parquet hourly.
2. **Week 2:** PortWatch ingestion with vintages; events table; descriptive plots.
3. **Week 3:** transit/laden/queue logic with tests (use NOAA/DMA sample data for fixtures); validation vs PortWatch.
4. **Week 4:** feature table; event study; VAR/Granger/local projections.
5. **Week 5:** walk-forward forecasting and research report.
6. **Week 6:** dashboard, scheduling, README with findings and limitations.

## 10. Conventions for Claude Code

- Secrets only in `.env`; never commit keys. Provide `.env.example`.
- Business logic in `src/`, not notebooks. Type hints, docstrings, ruff clean.
- Unit tests for gate crossing, laden classification, dedup, baseline z-scores, and the no-lookahead guarantee.
- Log row counts and data-quality stats at every pipeline stage.
- Before building each ingestion module, fetch the live docs and confirm the endpoint and schema; note any deviations from this brief in `CLAUDE.md`.
- Ask before adding any paid data source.
- Final README must include a Limitations section: AIS spoofing and dark fleet, terrestrial coverage gaps, draught unreliability, PortWatch revisions, short war-regime sample, and the fact that correlation around a geopolitical shock is not a tradable edge.
