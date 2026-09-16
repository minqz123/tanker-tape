# Tanker Tape

**Do ship movements through the Strait of Hormuz carry information about Brent crude
prices that the price history does not already contain?**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/brent-vs-transits-dark.png">
  <img alt="Brent crude spot price above, Strait of Hormuz tanker transits below, sharing one timeline with dated crisis events marked" src="docs/figures/brent-vs-transits.png">
</picture>

*Regenerated from each weekly data pull ([workflow](.github/workflows/weekly-data.yml)).
Every series is pulled from EIA/FRED and IMF PortWatch — nothing here is simulated.*

> **Research project, not a trading system.** No performance claim is made anywhere in
> this repository, and none should be inferred. See [Limitations](#limitations).

---

## The question, stated precisely

Physical oil flows are observable: AIS receivers track roughly 90,000 ships, and a closed
strait shows up as tankers that stop transiting and start waiting. Brent is a liquid market
that reprices on news within minutes.

So the question is not "do chokepoint closures move oil prices" — they obviously do. It is
narrower and harder:

> Conditional on everything the price history already tells you, does an AIS-derived
> signal improve an **out-of-sample** forecast of Brent returns or volatility?

The honest prior is **no**, at least for daily direction. Public information is priced
fast, and the main AIS dataset publishes weekly — the news moves days before the data
does. The places worth looking are volatility, the Brent–WTI spread, and regime
transitions.

## Status

| | |
|---|---|
| Pipeline | Complete, wired end to end, 161 tests |
| Historical data | **Flowing.** Prices 2015–present; PortWatch chokepoints and ports pulling weekly |
| Live AIS collection | **Not started** — collector written and deployable, see [`deploy/`](deploy/) |
| Findings | **None yet.** No result is claimed until the analysis has run on collected data |

The findings section stays empty until there is something to put in it. A portfolio project
that reports a result it has not computed is worth less than one that reports nothing.

## Why this is an easy question to get wrong

Most of the work here is not modelling. It is making a false positive hard to produce.

Two failure modes dominate this class of problem, and neither announces itself.
**Look-ahead bias** has many vectors — restated data, feature scaling, a rolling window that
includes its own centre, a publication schedule nobody modelled. It does not throw an
error; it produces a *better* result, which is exactly why it survives review.
**Overfitting** is the more famous one: any sufficiently flexible model will find structure
in noise, and a short sample containing one dramatic crisis is close to an ideal
environment for manufacturing a spurious relationship.

So the design premise is falsification-first: build the pipeline so that a result which
*looks* good has had to survive machinery built specifically to kill it.

## Validation

Enforced in code and covered by tests, not asserted in prose.

| Control | Implementation |
|---|---|
| Backward-only baselines | Rolling z-scores use `shift(1)`; date *t* is excluded from its own mean and standard deviation |
| Publication lag | PortWatch publishes Tuesdays, so Friday's count is not a Friday feature — joins use `available_from`, never `date` |
| Point-in-time vintages | Every pull is stored immutably under `retrieved_at`; requesting a vintage older than any on disk raises rather than backfilling |
| No full-sample scaling | Scalers live inside a per-fold `Pipeline` |
| Horizon-length embargo | Predicting date *i* at horizon *h* trains only to *i − h*, because a 20-day target is not observable for 20 days |
| Naming discipline | `walk_forward` raises if a `fwd_*` target column is passed as a feature |
| Benchmark ladder | Zero forecast → training mean → price-only → price + AIS, with a Diebold–Mariano test on the last step |
| Bidirectional causality | Granger is tested both ways and warns when both are significant, since that is feedback rather than evidence |

**How the no-lookahead test works.** It computes the features, then *changes the future* and
recomputes. Any value at or before the perturbation that moves is reading data it could not
have had. That catches the whole class — a centred window, a full-sample scaler, a baseline
that forgot to shift — without needing to know which mistake was made. It also includes a
test asserting that a deliberately leaky implementation *would* fail, so the guard is
itself guarded.

**Three feature families, lagged differently on purpose:**

| Source | Lag | Carried forward? |
|---|---|---|
| PortWatch chokepoints and ports | publication date (weekly) | yes, with `*_age_days` |
| Cross-route shares (Cape vs Suez) | inherits the PortWatch lag | yes |
| Self-collected AIS (`ais_*`) | none — readable same day | **no** |

Carrying a PortWatch value forward asserts "last week's published figure is still the
latest one", which is true. Carrying self-collected AIS forward would assert "yesterday's
traffic also happened today", which is false — a gap there means the collector was down,
and filling it would invent observations.

## Methodology

**Data.** Brent and WTI spot from EIA API v2 with an unauthenticated FRED fallback;
yfinance for timely futures closes, marked unofficial and never the series of record. IMF
PortWatch supplies the historical backbone — daily transit calls for 28 chokepoints and
~2,065 ports, published weekly and **revised**. A websocket collector on aisstream.io
gathers raw positions for metrics PortWatch does not provide.

**Transit detection.** Each chokepoint is a gate line plus a zone polygon. A transit is two
consecutive fixes for one vessel on opposite sides of the gate within six hours; longer
gaps are flagged `ambiguous` and reported separately rather than counted or dropped.
Vessels key on IMO where known, so a mid-voyage MMSI change does not split one ship in two,
and repeat crossings within six hours collapse so a vessel loitering on the gate does not
read as a fleet.

**Laden/ballast.** `laden_ratio = reported draught / max draught for that vessel`. The
denominator is the vessel's own observed maximum where available, a class-typical value by
length otherwise, and `max_draught_source` travels with every row. Draught is hand-entered
and frequently stale, so `draught_age_hours` is carried too and the *known* count is
reported beside the share — 0.8 from four vessels is not the claim that 0.8 from eighty is.

**Waiting fleet.** Tankers inside a zone below 1 knot for over six hours, split anchored
versus drifting. Probably the most informative live metric during a closure: ships that
cannot transit pile up rather than disappear.

**Data quality.** Implied speeds over 40 knots, identical positions shared across many
MMSIs at one timestamp (the GPS-jamming signature), and AIS silences over 12 hours inside a
zone. Everything is flagged and counted; nothing is silently dropped. `hours_with_data`
records collector uptime, because a gap in collection and a genuine drop in traffic are
indistinguishable in a transit count without it.

**Analysis order.** Descriptives and sanity checks → stationarity (ADF and KPSS) → event
study on dated events, pre-war and war regimes separately → VAR, bidirectional Granger,
Jordà local projections → rolling correlations → walk-forward forecasting against the
benchmark ladder → a toy backtest *only if* the forecasting step is significant.

Dated events live in [`data/reference/events.csv`](data/reference/events.csv), each with a
source URL and a confidence rating; dates that sources disagree on are marked as such
rather than silently picked.

## Limitations

Read this before believing any number this project produces.

**AIS spoofing and dark vessels.** IMF PortWatch explicitly warns of GPS jamming and
spoofing around Hormuz, and those behaviours increase exactly when the situation gets
interesting — so the measurement error is *correlated with the signal*. Every transit count
from this region is a lower bound, and the gap widens under stress.

**Terrestrial coverage gaps.** Self-collected AIS comes largely from land-based receivers.
Gulf coverage is not uniform, so apparent changes can reflect receiver availability rather
than ship movements.

**Draught is unreliable.** The laden signal rests on a number a crew member types in and
often does not update between voyages.

**PortWatch revises.** Published history changes as sampling and spoofing checks are
updated. Any analysis reading "latest" instead of an as-of vintage is using numbers that
did not exist at the time.

**The war-regime sample is short and not independent.** A handful of events inside one
crisis are successive stages of the same conflict with overlapping windows. Conventional
t-statistics overstate confidence here, and no amount of careful coding fixes that.

**Structural break.** Pre-war and war are different price-formation regimes; pooling across
February 2026 averages two things that never coexisted.

**Gate geometry is approximate.** Gate lines were placed from published narrowest-point
geography, not calibrated against observed traffic.

**And the fundamental one: correlation around a geopolitical shock is not a tradable
edge.** When a strait closes, transits fall and prices rise because of a third thing that
was on every front page that morning. Recovering that relationship demonstrates the
pipeline works. It does not demonstrate an information advantage, and it would not have
made money, because the news moved faster than the weekly data.

## Architecture

```
src/tanker_tape/
├── config.py            pydantic-settings; zones/gates/ports from config/*.yaml
├── storage.py           Parquet + DuckDB, vintage-aware (as-of) reads
├── ingest/
│   ├── prices.py        EIA v2, FRED, yfinance
│   ├── portwatch.py     paginated ArcGIS pulls, server-side filtered, vintaged
│   └── aisstream.py     async websocket collector, hourly Parquet
├── process/
│   ├── transits.py      gate-line crossing detection
│   ├── vessel_state.py  laden/ballast, waiting fleet, dark gaps, quality flags
│   ├── ais_metrics.py   raw collected AIS -> daily per-zone metrics
│   └── features.py      daily feature table (the leakage controls live here)
├── analysis/
│   ├── event_study.py   abnormal returns around dated events
│   ├── causality.py     stationarity, VAR, Granger both ways, local projections
│   ├── forecast.py      walk-forward, benchmarks, Diebold-Mariano
│   ├── charts.py        figures for the report, README and dashboard
│   └── report.py        generates the research report
└── dashboard/app.py     Streamlit
```

`deploy/` holds a hardened systemd unit for the collector. `.github/workflows/` runs CI on
every push plus the weekly data pull, which regenerates the figure above and records the
live PortWatch ID maps into `data/reference/`.

## Reproducibility

```bash
uv sync --all-extras
cp .env.example .env                  # EIA / aisstream keys
uv run tanker-tape verify-endpoints   # confirms live URLs, prints real field names
uv run tanker-tape validate-gates     # geometry check, no network needed

uv run tanker-tape ingest-prices --start 2015-01-01
uv run tanker-tape ingest-portwatch --dataset chokepoints
uv run tanker-tape build-features
uv run tanker-tape charts             # figures -> docs/figures (light + dark)
uv run tanker-tape report             # research report -> reports/
uv run tanker-tape dashboard
```

The live collector runs separately and should be started first, since AIS has no backfill —
every uncollected day is lost permanently. See [`deploy/README.md`](deploy/README.md).

```bash
uv run pytest        # 161 tests
uv run ruff check .
```

## Roadmap

1. Start the live collector (the only irreversible clock in the project).
2. Calibrate gate geometry against observed tracks, and the 0.75 laden threshold.
3. Validate self-collected transit counts against PortWatch on overlapping days.
4. Run the event study and walk-forward forecast on real data; publish whatever comes out.
5. Extend targets to the Brent–WTI spread, realized volatility, and tanker equities.

## License

MIT
