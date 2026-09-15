"""Configuration: environment settings plus the zone/gate/port geometry registry.

Geometry convention: every coordinate pair in ``config/*.yaml`` is ``[longitude,
latitude]`` to match shapely's ``(x, y)`` ordering. aisstream expects the opposite
(``[latitude, longitude]``), so that flip is confined to
:meth:`Zone.aisstream_bounding_box` and covered by a unit test.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from shapely.geometry import LineString, Polygon

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

Coordinate = tuple[float, float]


class Settings(BaseSettings):
    """Environment-backed settings. Secrets come from ``.env``; see ``.env.example``."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    eia_api_key: str | None = None
    fred_api_key: str | None = None
    aisstream_api_key: str | None = None

    tanker_tape_data_dir: Path = Path("data")
    tanker_tape_duckdb_path: Path = Path("data/tanker_tape.duckdb")
    tanker_tape_log_level: str = "INFO"

    @property
    def data_dir(self) -> Path:
        """Absolute path to the data lake root."""
        path = self.tanker_tape_data_dir
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def duckdb_path(self) -> Path:
        """Absolute path to the DuckDB catalogue file."""
        path = self.tanker_tape_duckdb_path
        return path if path.is_absolute() else REPO_ROOT / path

    def require(self, field: str) -> str:
        """Return a secret, raising a actionable error when it is unset.

        Args:
            field: Attribute name on this settings object, e.g. ``"eia_api_key"``.

        Raises:
            RuntimeError: If the value is missing or blank.
        """
        value = getattr(self, field, None)
        if not value:
            raise RuntimeError(
                f"{field.upper()} is not set. Copy .env.example to .env and fill it in."
            )
        return str(value)


class Zone(BaseModel):
    """A monitored chokepoint: a gate line to cross and a polygon to loiter in."""

    key: str
    name: str
    portwatch_id: str | None = None
    gate: list[Coordinate] = Field(min_length=2)
    zone: list[Coordinate] = Field(min_length=3)
    direction_positive: str
    direction_negative: str
    collect_live: bool = False

    @field_validator("gate", "zone")
    @classmethod
    def _validate_coordinates(cls, value: list[Coordinate]) -> list[Coordinate]:
        for lon, lat in value:
            if not -180.0 <= lon <= 180.0:
                raise ValueError(f"longitude {lon} out of range; expected [lon, lat] ordering")
            if not -90.0 <= lat <= 90.0:
                raise ValueError(f"latitude {lat} out of range; expected [lon, lat] ordering")
        return value

    @model_validator(mode="after")
    def _validate_geometry(self) -> Zone:
        if not self.polygon.is_valid:
            raise ValueError(
                f"zone polygon for {self.key!r} is invalid (probably self-intersecting)"
            )
        if not self.polygon.intersects(self.line):
            raise ValueError(
                f"gate line for {self.key!r} does not intersect its own zone polygon; "
                "one of the two is misplaced"
            )
        return self

    @functools.cached_property
    def line(self) -> LineString:
        """The gate as a shapely LineString in (lon, lat) space."""
        return LineString(self.gate)

    @functools.cached_property
    def polygon(self) -> Polygon:
        """The monitored zone as a shapely Polygon in (lon, lat) space."""
        return Polygon(self.zone)

    def direction_label(self, sign: Literal[1, -1]) -> str:
        """Human-readable direction for a crossing into (+1) or out of (-1) the positive side."""
        return self.direction_positive if sign > 0 else self.direction_negative

    def aisstream_bounding_box(self) -> list[list[float]]:
        """Zone bounds as an aisstream bounding box.

        aisstream takes ``[[lat, lon], [lat, lon]]`` corner pairs, which is the reverse
        of this project's internal ``(lon, lat)`` ordering.

        Returns:
            Two ``[latitude, longitude]`` corners: south-west then north-east.
        """
        min_lon, min_lat, max_lon, max_lat = self.polygon.bounds
        return [[min_lat, min_lon], [max_lat, max_lon]]


class Port(BaseModel):
    """A PortWatch port we track daily call/volume estimates for."""

    key: str
    name: str
    match_name: str
    country: str
    group: str
    portwatch_id: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@functools.cache
def load_zones(config_dir: Path | None = None) -> dict[str, Zone]:
    """Load and validate chokepoint definitions from ``config/zones.yaml``.

    Args:
        config_dir: Override the config directory (used by tests).

    Returns:
        Mapping of zone key to :class:`Zone`.
    """
    directory = config_dir or CONFIG_DIR
    raw = _load_yaml(directory / "zones.yaml")
    return {key: Zone(key=key, **body) for key, body in (raw.get("chokepoints") or {}).items()}


@functools.cache
def load_ports(config_dir: Path | None = None) -> dict[str, Port]:
    """Load and validate port definitions from ``config/ports.yaml``.

    Args:
        config_dir: Override the config directory (used by tests).

    Returns:
        Mapping of port key to :class:`Port`.
    """
    directory = config_dir or CONFIG_DIR
    raw = _load_yaml(directory / "ports.yaml")
    return {entry["key"]: Port(**entry) for entry in (raw.get("ports") or [])}


def live_zones(config_dir: Path | None = None) -> dict[str, Zone]:
    """Zones flagged ``collect_live: true`` — the boxes the AIS collector subscribes to."""
    return {key: zone for key, zone in load_zones(config_dir).items() if zone.collect_live}


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
