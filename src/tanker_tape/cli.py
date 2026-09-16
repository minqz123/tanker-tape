"""Command-line interface.

Run ``tanker-tape verify-endpoints`` first on a machine with open network access: several
endpoints in this repo are documented but unconfirmed (see the verification log in
CLAUDE.md), and that command is what turns them into confirmed ones.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import typer
from rich.console import Console
from rich.table import Table

from . import DISCLAIMER, __version__
from .config import get_settings, live_zones, load_ports, load_zones
from .logging_utils import configure_logging, get_logger

app = typer.Typer(
    help="Brent crude x AIS shipping signals. Research project - not investment advice.",
    no_args_is_help=True,
)
console = Console()
logger = get_logger(__name__)


@app.callback()
def main(log_level: str = typer.Option("INFO", help="Logging level.")) -> None:
    """Configure logging for every subcommand."""
    configure_logging(log_level)


@app.command()
def version() -> None:
    """Print the version and the standing disclaimer."""
    console.print(f"tanker-tape {__version__}")
    console.print(f"[dim]{DISCLAIMER}[/dim]")


@app.command("verify-endpoints")
def verify_endpoints(
    check_prices: bool = typer.Option(True, help="Check the EIA/FRED price endpoints."),
    check_portwatch: bool = typer.Option(True, help="Check the PortWatch FeatureServers."),
) -> None:
    """Probe every external endpoint and report what it actually returns.

    This exists because the project brief's endpoints were gathered from research and may
    have changed. It prints the live field list so the schema-tolerant parsers in
    ``ingest/`` can be pointed at the right columns, and resolves the configured
    chokepoint/port IDs against the live tables.
    """
    table = Table(title="Endpoint verification", show_lines=True)
    table.add_column("Endpoint")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")

    if check_portwatch:
        from .ingest import portwatch

        for label, url in (
            ("PortWatch chokepoints", portwatch.CHOKEPOINTS_LAYER),
            ("PortWatch ports", portwatch.PORTS_LAYER),
        ):
            try:
                metadata = portwatch.fetch_layer_metadata(url)
                fields = ", ".join(field["name"] for field in metadata.get("fields", []))
                table.add_row(
                    label,
                    "[green]OK[/green]",
                    f"name={metadata.get('name')} "
                    f"maxRecordCount={metadata.get('maxRecordCount')}\nfields: {fields}",
                )
            except Exception as exc:  # noqa: BLE001 - this command reports failures
                table.add_row(label, "[red]FAIL[/red]", f"{type(exc).__name__}: {exc}")

    if check_prices:
        from .ingest import prices

        try:
            frame = prices.fetch_eia_spot("RBRTE", start=dt.date.today() - dt.timedelta(days=30))
            table.add_row(
                "EIA v2 petroleum/pri/spt",
                "[green]OK[/green]",
                f"rows={len(frame)} latest={frame['date'].max() if len(frame) else 'n/a'}",
            )
        except Exception as exc:  # noqa: BLE001
            table.add_row(
                "EIA v2 petroleum/pri/spt", "[red]FAIL[/red]", f"{type(exc).__name__}: {exc}"
            )

        try:
            frame = prices.fetch_fred_spot("DCOILBRENTEU")
            table.add_row(
                "FRED CSV fallback",
                "[green]OK[/green]",
                f"rows={len(frame)} latest={frame['date'].max() if len(frame) else 'n/a'}",
            )
        except Exception as exc:  # noqa: BLE001
            table.add_row("FRED CSV fallback", "[red]FAIL[/red]", f"{type(exc).__name__}: {exc}")

    console.print(table)
    console.print(
        "\n[yellow]Record what this printed in the endpoint verification log in "
        "CLAUDE.md, including any schema deviations.[/yellow]"
    )


@app.command("resolve-ids")
def resolve_ids(entity: str = typer.Argument("chokepoint", help="'chokepoint' or 'port'.")) -> None:
    """List the live PortWatch IDs and names, to confirm the values in ``config/``.

    The brief claims Hormuz is ``chokepoint6``; this is how you check rather than assume.
    """
    from .ingest import portwatch

    if entity not in {"chokepoint", "port"}:
        raise typer.BadParameter("entity must be 'chokepoint' or 'port'")

    layer = portwatch.CHOKEPOINTS_LAYER if entity == "chokepoint" else portwatch.PORTS_LAYER
    frame = portwatch.normalise_daily_table(portwatch.fetch_layer(layer), entity)
    lookup = portwatch.resolve_entity_ids(frame, entity)

    table = Table(title=f"PortWatch {entity} IDs")
    for column in lookup.columns:
        table.add_column(str(column))
    for row in lookup.itertuples(index=False):
        table.add_row(*(str(value) for value in row))
    console.print(table)


@app.command("ingest-prices")
def ingest_prices(
    start: str = typer.Option("2015-01-01", help="Inclusive start date."),
    end: str | None = typer.Option(None, help="Inclusive end date."),
    prefer: str = typer.Option("eia", help="Preferred source: 'eia' or 'fred'."),
) -> None:
    """Fetch Brent/WTI, build the price panel, and write it to the processed store."""
    from .ingest.prices import build_price_panel
    from .storage import write_processed

    panel = build_price_panel(start=start, end=end, prefer=prefer)
    path = write_processed(panel, "prices_daily")
    console.print(f"[green]Wrote {len(panel)} rows[/green] to {path}")


@app.command("ingest-portwatch")
def ingest_portwatch(
    dataset: str = typer.Option("chokepoints", help="'chokepoints' or 'ports'."),
) -> None:
    """Pull a PortWatch table and store it as a new immutable vintage."""
    from .ingest import portwatch

    if dataset == "chokepoints":
        frame = portwatch.ingest_chokepoints()
    elif dataset == "ports":
        frame = portwatch.ingest_ports()
    else:
        raise typer.BadParameter("dataset must be 'chokepoints' or 'ports'")

    console.print(f"[green]Stored vintage with {len(frame)} rows[/green]")


@app.command("collect-ais")
def collect_ais(
    max_reconnects: int | None = typer.Option(
        None, help="Stop after this many reconnect attempts (default: run forever)."
    ),
) -> None:
    """Run the live aisstream collector.

    There is no history available for this source: every day not collected is lost
    permanently. Run it under systemd or cron from day one.
    """
    from .ingest.aisstream import collect

    zones = live_zones()
    if not zones:
        console.print("[red]No zones have collect_live: true in config/zones.yaml[/red]")
        raise typer.Exit(1)

    console.print(f"Collecting: {', '.join(zone.name for zone in zones.values())}")
    try:
        asyncio.run(collect(max_reconnects=max_reconnects))
    except KeyboardInterrupt:
        console.print("\n[yellow]Collector stopped; buffered rows were flushed.[/yellow]")


@app.command("validate-gates")
def validate_gates() -> None:
    """Sanity-check the configured geometry without touching the network.

    Confirms each gate intersects its zone, and prints the aisstream bounding box that
    will actually be subscribed to, so a latitude/longitude swap is visible before it
    costs a week of collection.
    """
    table = Table(title="Zone geometry")
    table.add_column("Key")
    table.add_column("Name")
    table.add_column("Gate length (deg)")
    table.add_column("Zone bounds (lon/lat)")
    table.add_column("aisstream box (lat/lon)")
    table.add_column("Live")

    for key, zone in load_zones().items():
        min_lon, min_lat, max_lon, max_lat = zone.polygon.bounds
        table.add_row(
            key,
            zone.name,
            f"{zone.line.length:.3f}",
            f"[{min_lon:.2f}, {max_lon:.2f}] / [{min_lat:.2f}, {max_lat:.2f}]",
            str(zone.aisstream_bounding_box()),
            "yes" if zone.collect_live else "no",
        )

    console.print(table)
    console.print(
        f"[dim]{len(load_ports())} ports configured; "
        "IDs are unverified until resolve-ids is run.[/dim]"
    )


@app.command("build-features")
def build_features(
    as_of: str | None = typer.Option(
        None, help="Read the PortWatch vintage available on this date (YYYY-MM-DD)."
    ),
    respect_publication_lag: bool = typer.Option(
        True, help="Join AIS features on publication date. Disable for plots only."
    ),
) -> None:
    """Assemble the daily feature table from stored prices and PortWatch vintages."""
    from .process.features import build_feature_table
    from .storage import read_processed, read_vintage, write_processed

    cutoff = dt.date.fromisoformat(as_of) if as_of else None
    prices = read_processed("prices_daily")
    chokepoints = read_vintage("portwatch_chokepoints", as_of=cutoff)

    count_column = next(
        (
            column
            for column in ("n_transits", "n_tanker", "n_total")
            if column in chokepoints.columns
        ),
        None,
    )
    if count_column is None:
        console.print(
            "[red]Could not find a transit-count column in the PortWatch vintage. "
            f"Available: {sorted(chokepoints.columns)}[/red]"
        )
        raise typer.Exit(1)

    daily = chokepoints.rename(columns={"chokepoint_id": "zone_key"})

    # Ports are optional: that vintage may not have been pulled yet.
    port_daily, port_column = None, None
    try:
        port_daily = read_vintage("portwatch_ports", as_of=cutoff)
        port_column = next(
            (
                column
                for column in ("n_tanker", "n_total", "portcalls_tanker")
                if column in port_daily.columns
            ),
            None,
        )
        if port_column is None:
            console.print(
                "[yellow]Port vintage found but no recognised tanker measure; "
                f"available: {sorted(port_daily.columns)}[/yellow]"
            )
            port_daily = None
    except FileNotFoundError:
        console.print("[dim]No PortWatch port vintage yet; skipping port-group features.[/dim]")

    # Our own AIS metrics are optional too, and only exist once the collector has run.
    ais_daily = None
    try:
        ais_daily = read_processed("ais_daily")
    except FileNotFoundError:
        console.print(
            "[dim]No collected-AIS metrics yet; run `tanker-tape build-ais-metrics` "
            "once the collector has data.[/dim]"
        )

    table = build_feature_table(
        prices,
        daily,
        value_columns=(count_column,),
        respect_publication_lag=respect_publication_lag,
        port_daily=port_daily,
        port_value_column=port_column,
        ais_daily=ais_daily,
    )
    path = write_processed(table, "features_daily")
    console.print(f"[green]Wrote {len(table)} rows x {table.shape[1]} columns[/green] to {path}")


@app.command("build-ais-metrics")
def build_ais_metrics_command(
    start: str | None = typer.Option(None, help="First collection date to read (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, help="Last collection date to read (YYYY-MM-DD)."),
    laden_threshold: float = typer.Option(0.75, help="Laden ratio cutoff; calibrate this."),
) -> None:
    """Turn collected raw AIS into daily per-zone metrics.

    Produces transit counts, the waiting fleet, laden share, dark gaps and the
    data-quality/collector-uptime summary the dashboard reads. Run it after the
    collector has been going for a while; it needs raw partitions to work on.
    """
    from .process.ais_metrics import build_ais_metrics

    tables = build_ais_metrics(
        start=dt.date.fromisoformat(start) if start else None,
        end=dt.date.fromisoformat(end) if end else None,
        laden_threshold=laden_threshold,
    )

    if all(frame is None or frame.empty for frame in tables.values()):
        console.print(
            "[red]No metrics produced - no collected AIS found.[/red]\n"
            "Is the collector running? See deploy/README.md. There is no backfill "
            "for this source, so gaps cannot be recovered later."
        )
        raise typer.Exit(1)

    summary = Table(title="AIS metrics written")
    summary.add_column("Table")
    summary.add_column("Rows", justify="right")
    for name, frame in tables.items():
        stored = name if name.startswith("ais_") else f"ais_{name}"
        summary.add_row(stored, str(len(frame)) if frame is not None else "0")
    console.print(summary)


@app.command()
def charts(
    output: str = typer.Option("docs/figures", help="Directory to write PNGs into."),
) -> None:
    """Render the report figures in both light and dark from the feature table.

    Figures are generated from real pulled data only. Nothing here invents a series:
    with no feature table, this command tells you to build one rather than drawing
    something plausible.
    """
    from .analysis.charts import render_all
    from .process.features import load_events
    from .storage import read_processed

    try:
        table = read_processed("features_daily")
    except FileNotFoundError:
        console.print("[red]No feature table found. Run `tanker-tape build-features` first.[/red]")
        raise typer.Exit(1) from None

    try:
        events = load_events()
    except (FileNotFoundError, OSError):
        events = None

    written = render_all(table, events, output_dir=output)
    summary = Table(title="Figures written")
    summary.add_column("Figure")
    summary.add_column("Files")
    for name, paths in written.items():
        summary.add_row(name, "\n".join(str(path) for path in paths))
    console.print(summary)


@app.command()
def dashboard(port: int = typer.Option(8501, help="Port to serve on.")) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    from pathlib import Path

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run(["streamlit", "run", str(app_path), "--server.port", str(port)], check=False)


@app.command()
def status() -> None:
    """Show what is configured and what data is on disk."""
    settings = get_settings()
    from .storage import list_vintages

    table = Table(title="Tanker Tape status")
    table.add_column("Item")
    table.add_column("Value")

    table.add_row("Data dir", str(settings.data_dir))
    table.add_row("DuckDB", str(settings.duckdb_path))
    table.add_row("Zones configured", str(len(load_zones())))
    table.add_row("Zones collecting live", str(len(live_zones())))
    table.add_row("Ports configured", str(len(load_ports())))

    for key in ("eia_api_key", "fred_api_key", "aisstream_api_key"):
        table.add_row(key, "[green]set[/green]" if getattr(settings, key) else "[red]unset[/red]")

    for dataset in ("portwatch_chokepoints", "portwatch_ports"):
        vintages = list_vintages(dataset)
        latest = vintages[-1].isoformat() if vintages else "none"
        table.add_row(f"{dataset} vintages", f"{len(vintages)} (latest: {latest})")

    console.print(table)
    console.print(f"[dim]{DISCLAIMER}[/dim]")


@app.command()
def report(
    output: str | None = typer.Option(None, help="Output path; defaults to reports/."),
    horizons: str = typer.Option("1,5,20", help="Comma-separated forecast horizons in days."),
    min_train: int = typer.Option(500, help="Minimum training rows before walk-forward starts."),
) -> None:
    """Generate the research report from the built feature table.

    Sections are independently guarded, so a failure in one (say the event study)
    reports itself in place rather than losing the rest of the report.
    """
    from .analysis.report import generate_report
    from .storage import read_processed

    try:
        table = read_processed("features_daily")
    except FileNotFoundError:
        console.print("[red]No feature table found. Run `tanker-tape build-features` first.[/red]")
        raise typer.Exit(1) from None

    parsed = tuple(int(value) for value in horizons.split(",") if value.strip())
    path = generate_report(table, output_path=output, horizons=parsed, min_train=min_train)
    console.print(f"[green]Wrote report[/green] to {path}")


if __name__ == "__main__":
    app()
