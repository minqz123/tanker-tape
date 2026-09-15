# CLAUDE.md — Tanker Tape

Condensed hand-off for agents working in this repo. The full brief lives in
`docs/PROJECT_BRIEF.md`; this file is the operational summary plus conventions.

## What this is

A reproducible research + monitoring system measuring how physical oil flows observed in
AIS ship-tracking data relate to Brent crude prices, and testing whether AIS-derived
signals carry **incremental, out-of-sample** information about Brent returns and volatility.

**This is a research/portfolio project, not a trading system.** Every output — report,
dashboard page, README — must say so.

## Deliverables

1. Data pipeline: historical + live AIS-derived metrics joined to daily Brent/WTI in DuckDB/Parquet.
2. Research report: event study, causality tests, walk-forward forecasting with honest benchmarks.
3. Dashboard: chokepoint status, tanker flow metrics, Brent overlay.

## Layout

```
src/tanker_tape/
  config.py           pydantic-settings; zones/gates/ports from config/*.yaml
  storage.py          Parquet + DuckDB helpers, vintage-aware writes
  ingest/prices.py    EIA v2, FRED, yfinance
  ingest/portwatch.py paginated ArcGIS FeatureServer pulls, stores vintage
  ingest/aisstream.py async websocket collector
  process/transits.py     gate-line crossing detection
  process/vessel_state.py laden/ballast, waiting/anchored, dark gaps, DQ flags
  process/ais_metrics.py  raw collected AIS -> daily per-zone metrics
  process/features.py     daily feature table
  analysis/               event_study, causality, forecast, report
  dashboard/app.py        Streamlit
config/                 zones.yaml (gates/polygons), ports.yaml
deploy/                 hardened systemd unit + deployment guide for the collector
.github/workflows/      CI on push; weekly PortWatch/price pull on Wednesdays
data/{raw,interim,processed,reference}/   gitignored except reference/
```

Business logic lives in `src/`, never in notebooks. Notebooks are for exploration only.

## Conventions

- Secrets only in `.env`; `.env.example` is committed, `.env` never is.
- Type hints and docstrings on public functions. `ruff check` and `ruff format` clean.
- Log row counts and data-quality stats at **every** pipeline stage (`log_stage_counts`).
- Never silently drop bad data — flag it, count it, report it.
- Ask before adding any paid data source.
- Before building against an endpoint, fetch the live docs and confirm schema; record any
  deviation in the "Endpoint verification log" below.

## Non-negotiable analysis rules (leakage)

These are enforced in code and covered by tests. Do not weaken them.

- No future baseline windows. Rolling baselines use `shift(1)` so the value at date *t* is
  computed from data strictly before *t*. See `features.rolling_zscore`.
- No full-sample scaling. Scalers fit inside the training fold only.
- Publication lags applied. PortWatch publishes weekly (Tuesdays 09:00 ET) and **revises**
  history, so a feature dated *t* may only use a vintage retrieved on or before *t*.
- As-of vintages used. Every PortWatch pull is stamped `retrieved_at`; analysis reads
  `as_of=` rather than "latest".
- Hyperparameters tuned only inside training folds.
- PortWatch-derived features are lagged to publication date and carried forward with an
  age column. **Self-collected AIS features (`ais_*`) are same-day and are never carried
  forward** — a gap there means the collector was down, and filling it would invent
  traffic that was never observed.

The regression test for this is `tests/test_no_lookahead.py`: perturbing a future
observation must not change any feature value at or before the perturbation date.

## Core logic specs

**Transit detection.** Each chokepoint is a gate LineString plus a zone Polygon. A transit is
two consecutive position reports for one MMSI on opposite sides of the gate within
`max_gap_hours` (default 6). Direction from the sign of the side function on the crossed
segment. Deduplicate on MMSI+IMO; MMSI changes are reconciled via IMO from static data.

**Laden/ballast.** `laden_ratio = reported_draught / max_draught_for_that_vessel`, falling back
to a class-typical max by length when the vessel has no history. Laden if ratio > 0.75
(calibrate). Draught is manually entered and often stale — treat as noisy and carry
`draught_age_hours`.

**Queue/waiting.** Tankers in zone with SOG < 1 kn for > 6h, split anchored vs drifting by
navigational status. Likely the most informative live metric during a closure.

**Data-quality flags.** Implied speed > 40 kn between fixes, positions on land, identical
positions shared across many MMSIs (spoofing clusters), AIS gaps > 12h inside a zone
("went dark"). Count and report daily; never drop silently.

**Vintage tracking.** Every PortWatch pull is written under a `retrieved_at` partition.

## Endpoint verification log

Checked 2026-09-15. **Two of these could not be reached from the build container** — the
egress proxy denied `portwatch.imf.org`, `www.eia.gov`, `aisstream.io`, and
`services9.arcgis.com`. Values below marked *unconfirmed* came from documentation and
third-party mirrors, not a live response. Run `tanker-tape verify-endpoints` from a machine
with open network access before trusting them, and update this log with what it prints.

| Source | Value used | Status |
|---|---|---|
| aisstream | `wss://stream.aisstream.io/v0/stream`; subscription `{APIKey, BoundingBoxes, FiltersShipMMSI?, FilterMessageTypes?}` within 3s; bounding boxes are `[[lat,lon],[lat,lon]]` | Confirmed against the published AsyncAPI spec |
| aisstream | Envelope `{MessageType, MetaData, Message:{<Type>:{...}}}`; `PositionReport.{UserID,Latitude,Longitude,Sog,Cog,NavigationalStatus,TrueHeading}`; `ShipStaticData.{UserID,ImoNumber,Name,Type,MaximumStaticDraught,Dimension,Destination}` | Confirmed against the AsyncAPI spec |
| PortWatch chokepoints | `https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Chokepoints_Data/FeatureServer/0/query` (dataset `42132aa4e2fc4d41bdaf9a445f688931_0`) | **Unconfirmed** — host blocked by egress policy |
| PortWatch ports | `https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Ports_Data/FeatureServer/0/query` (dataset `959214444157458aad969389b3ebe1a0_0`) | **Unconfirmed** — host blocked by egress policy |
| PortWatch paging | 1000 records per response; page with `resultOffset`, order by `ObjectId` ascending | **Unconfirmed** — documented limit, not observed |
| EIA | `https://api.eia.gov/v2/petroleum/pri/spt/data/` with `frequency=daily`, `data[0]=value`, `facets[series][]=RBRTE`, `length<=5000` | **Unconfirmed** — host blocked by egress policy |

Deviations from the brief found so far:

- The brief says Hormuz is `chokepoint6`. This could not be verified, so the ID is a
  **config value** in `config/zones.yaml` (`portwatch_id`), not a constant in code.
  `tanker-tape verify-endpoints` resolves chokepoint names to IDs from the live table and
  will tell you if it disagrees.
- Port IDs in `config/ports.yaml` are likewise unverified placeholders. Ports whose ID does
  not resolve are skipped with a loud warning rather than silently returning empty frames.

## Commands

```bash
uv sync --all-extras
uv run tanker-tape verify-endpoints        # do this first, on a networked machine
uv run tanker-tape ingest-prices --start 2015-01-01
uv run tanker-tape ingest-portwatch --dataset chokepoints
uv run tanker-tape collect-ais             # long-running; run under systemd/cron
uv run tanker-tape build-ais-metrics        # collected AIS -> daily metrics
uv run tanker-tape build-features
uv run tanker-tape report                  # research report -> reports/
uv run pytest
uv run ruff check .
```
