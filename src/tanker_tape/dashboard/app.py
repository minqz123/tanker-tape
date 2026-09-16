"""Streamlit dashboard: chokepoint status, Brent overlay, and data quality.

Run with ``tanker-tape dashboard`` (or ``streamlit run src/tanker_tape/dashboard/app.py``).

The data-quality panel is not decoration. Around Hormuz, GPS jamming and AIS spoofing are
routine, so a transit count shown without its dark-gap and spoofing-flag counts is telling
only half the story.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from tanker_tape import DISCLAIMER
from tanker_tape.config import load_zones
from tanker_tape.process.features import load_events
from tanker_tape.storage import list_vintages, read_processed

st.set_page_config(page_title="Tanker Tape", page_icon="🛢️", layout="wide")


@st.cache_data(ttl=900)
def _load(name: str) -> pd.DataFrame | None:
    try:
        return read_processed(name)
    except FileNotFoundError:
        return None


@st.cache_data(ttl=900)
def _load_events() -> pd.DataFrame:
    try:
        return load_events()
    except (FileNotFoundError, OSError):
        return pd.DataFrame(columns=["date", "event", "category", "source_url"])


def _status_colour(z_score: float | None) -> str:
    """Colour a status card by how far the day sits from its trailing baseline."""
    if z_score is None or pd.isna(z_score):
        return "gray"
    if abs(z_score) >= 2.5:
        return "red"
    if abs(z_score) >= 1.5:
        return "orange"
    return "green"


def render_header() -> None:
    st.title("Tanker Tape")
    st.caption(
        "Physical oil flows from AIS ship tracking, against Brent crude. "
        "Research project — not investment advice."
    )


def render_chokepoint_cards(features: pd.DataFrame) -> None:
    st.subheader("Chokepoint status")
    zones = load_zones()
    latest = features.iloc[-1] if not features.empty else None
    if latest is None:
        st.info("No feature data yet. Run `tanker-tape build-features`.")
        return

    columns = st.columns(min(len(zones), 4) or 1)
    for index, (key, zone) in enumerate(zones.items()):
        target = columns[index % len(columns)]
        # Exclude the derived columns explicitly. A substring test alone can pick
        # "<zone>_n_transits_age_days" as the headline figure and show a staleness
        # count where the transit count belongs.
        count_column = next(
            (
                column
                for column in features.columns
                if column.startswith(f"{key}_")
                and not column.endswith("_age_days")
                and "_z28d" not in column
                and "_z90d" not in column
                and "_yoy_dev" not in column
            ),
            None,
        )
        z_column = next(
            (
                column
                for column in features.columns
                if column.startswith(f"{key}_") and column.endswith("_z28d")
            ),
            None,
        )
        value = latest.get(count_column) if count_column else None
        z_score = latest.get(z_column) if z_column else None

        with target:
            st.metric(
                label=zone.name,
                value="n/a" if value is None or pd.isna(value) else f"{value:,.0f}",
                delta=None if z_score is None or pd.isna(z_score) else f"z = {z_score:+.2f}",
            )
            st.caption(f":{_status_colour(z_score)}[vs 28-day baseline]")


def render_price_overlay(features: pd.DataFrame, events: pd.DataFrame) -> None:
    st.subheader("Brent and chokepoint traffic")
    if "brent_spot" not in features.columns:
        st.info("No Brent series in the feature table yet.")
        return

    traffic_columns = [
        column
        for column in features.columns
        if column.startswith("hormuz_") and "_z" not in column and "_yoy" not in column
    ]
    chosen = st.selectbox("Traffic series", traffic_columns) if traffic_columns else None

    frame = features.set_index("date")
    st.line_chart(frame[["brent_spot"]].dropna(), height=260)
    if chosen:
        st.line_chart(frame[[chosen]].dropna(), height=200)

    if not events.empty:
        st.caption("Event markers")
        st.dataframe(
            events.loc[:, ["date", "event", "category", "confidence"]]
            if "confidence" in events.columns
            else events,
            use_container_width=True,
            hide_index=True,
        )


def render_waiting_fleet(features: pd.DataFrame) -> None:
    st.subheader("Waiting fleet")
    waiting_columns = [column for column in features.columns if column.endswith("_n_waiting")]
    if not waiting_columns:
        st.info(
            "No waiting-fleet data yet. This comes from the live AIS collector "
            "(`tanker-tape collect-ais`), not from PortWatch."
        )
        return
    st.line_chart(features.set_index("date")[waiting_columns].dropna(how="all"), height=260)


def render_data_quality() -> None:
    st.subheader("Data quality")
    quality = _load("ais_quality_daily")

    left, right = st.columns(2)
    with left:
        st.markdown("**PortWatch vintages**")
        for dataset in ("portwatch_chokepoints", "portwatch_ports"):
            vintages = list_vintages(dataset)
            if vintages:
                age = (dt.datetime.now(dt.UTC) - vintages[-1]).days
                st.write(f"{dataset}: {len(vintages)} vintages, latest {age}d old")
                if age > 10:
                    st.warning(
                        f"{dataset} is {age} days stale — PortWatch publishes weekly, "
                        "so the ingest may be failing."
                    )
            else:
                st.write(f"{dataset}: none")

    with right:
        st.markdown("**AIS collector**")
        if quality is None or quality.empty:
            st.write("No collector quality data yet.")
        else:
            st.dataframe(quality.tail(14), use_container_width=True, hide_index=True)

    st.caption(
        "IMF PortWatch warns of GPS jamming, AIS spoofing, and vessels going dark around "
        "Hormuz. Treat counts from this region as a lower bound on real traffic."
    )


def main() -> None:
    render_header()
    features = _load("features_daily")
    events = _load_events()

    if features is None:
        st.warning(
            "No feature table found. Run `tanker-tape ingest-prices`, "
            "`tanker-tape ingest-portwatch`, then `tanker-tape build-features`."
        )
    else:
        features = features.copy()
        features["date"] = pd.to_datetime(features["date"])
        render_chokepoint_cards(features)
        st.divider()
        render_price_overlay(features, events)
        st.divider()
        render_waiting_fleet(features)

    st.divider()
    render_data_quality()
    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
