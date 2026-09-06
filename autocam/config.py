"""Single source of truth for every tunable constant.

`configs/default.json` holds the values; this module loads, validates, merges
and hashes them. Phase 2 (iOS) loads the *same* JSON file. Logic modules take a
``Config`` and never define a numeric default of their own.

Units, once, here:

* Anything spatial in ``control`` is a **fraction of source frame width** (fw).
  ``deadzone`` 0.03 on a 3840 px source is 115 px. Velocity is fw/s,
  acceleration fw/s^2. This is what makes the constants portable to Phase 2,
  where they are rescaled fw -> degrees using the horizontal FOV.
* ``lead_gain`` is in **seconds**: lead = lead_gain * centroid velocity (fw/s),
  so gain * velocity lands back in fw.
* ``lead_vel_tau`` is the time constant (s) of the exponential smoother applied
  to raw centroid velocity before it is used as lead.
* ``bbox_area_min`` / ``bbox_area_max`` are fractions of total frame **area**.
* ``detect_hz`` is detections per second of source time, not per frame.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 1

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent


class ConfigError(ValueError):
    """Raised when a config file is malformed or holds an out-of-range value."""


def _resolve_default_config() -> Path:
    """Find ``configs/default.json``.

    It lives at the repo root because Phase 2 bundles that same file as an app
    resource -- one file for both phases. ``AUTOCAM_DEFAULT_CONFIG`` overrides
    the search for anyone installing the package away from its source tree.
    """
    candidates = []
    env = os.environ.get("AUTOCAM_DEFAULT_CONFIG")
    if env:
        candidates.append(Path(env))
    candidates += [
        _REPO_ROOT / "configs" / "default.json",
        _PACKAGE_DIR / "configs" / "default.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ConfigError(
        "could not find configs/default.json; looked in "
        + ", ".join(str(c) for c in candidates)
        + ". Install from a source checkout (pip install -e .) or set "
        "AUTOCAM_DEFAULT_CONFIG."
    )


DEFAULT_CONFIG_PATH = _resolve_default_config()


def default_config_path() -> Path:
    """Re-resolve at call time so AUTOCAM_DEFAULT_CONFIG can be set late (tests)."""
    return _resolve_default_config()


@dataclass(frozen=True)
class IngestConfig:
    min_source_width: int
    min_source_height: int
    honour_rotation_metadata: bool


@dataclass(frozen=True)
class DetectConfig:
    backend: str
    model: str
    device: str
    person_class_id: int
    detect_hz: float
    infer_long_edge: int
    conf_min: float
    bbox_area_min: float
    bbox_area_max: float


@dataclass(frozen=True)
class CentroidConfig:
    mad_k: float
    min_players: int
    spread_metric: str


@dataclass(frozen=True)
class ControlConfig:
    deadzone: float
    omega_n: float
    zeta: float
    max_pan_vel: float
    max_pan_accel: float
    lead_gain: float
    lead_max: float
    lead_vel_tau: float


@dataclass(frozen=True)
class RenderConfig:
    out_width: int
    out_height: int
    crop_y_mode: str
    crop_y_frac: float
    codec: str
    crf: int
    preset: str
    pix_fmt: str
    copy_audio: bool
    preview_duration_s: float
    preview_height: int
    preview_crf: int
    preview_preset: str
    preview_copy_audio: bool


@dataclass(frozen=True)
class EventsConfig:
    attacking_third_debounce_s: float
    transition_vel_threshold: float
    transition_min_duration_s: float
    transition_debounce_s: float


@dataclass(frozen=True)
class QualityConfig:
    invalid_run_warn_s: float
    invalid_frame_pct_warn: float
    mean_players_warn: float


@dataclass(frozen=True)
class Phase2Config:
    """Consumed by Phase 2 only. Present here so the constants cannot diverge."""

    deadzone_scale: float
    pan_limit_deg: float
    fov_h_deg: float
    watchdog_ms: int
    notify_hz: int
    max_vel_centideg_per_s: int


@dataclass(frozen=True)
class Config:
    schema_version: int
    ingest: IngestConfig
    detect: DetectConfig
    centroid: CentroidConfig
    control: ControlConfig
    render: RenderConfig
    events: EventsConfig
    quality: QualityConfig
    phase2: Phase2Config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        """sha256 of the resolved config, canonicalised so it is reproducible."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def detection_fingerprint(self) -> str:
        """Hash of only the constants that change raw inference output.

        Used to decide whether a cached ``<video>.detections.json`` is still
        valid. Control/render/event tuning must never invalidate the cache --
        that is the difference between a 30-second and a 12-minute tuning cycle.
        """
        d = asdict(self.detect)
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


_SECTIONS: dict[str, type] = {
    "ingest": IngestConfig,
    "detect": DetectConfig,
    "centroid": CentroidConfig,
    "control": ControlConfig,
    "render": RenderConfig,
    "events": EventsConfig,
    "quality": QualityConfig,
    "phase2": Phase2Config,
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(section_cls: type, name: str, raw: dict) -> Any:
    expected = {f.name for f in fields(section_cls)}
    got = set(raw)
    unknown = got - expected
    if unknown:
        raise ConfigError(f"unknown key(s) in section '{name}': {sorted(unknown)}")
    missing = expected - got
    if missing:
        raise ConfigError(f"missing key(s) in section '{name}': {sorted(missing)}")
    kwargs = {}
    for f in fields(section_cls):
        value = raw[f.name]
        if f.type in ("int", int) and isinstance(value, bool):
            raise ConfigError(f"{name}.{f.name} must be a number, got bool")
        if f.type in ("float", float):
            value = float(value)
        elif f.type in ("int", int):
            if isinstance(value, float) and not value.is_integer():
                raise ConfigError(f"{name}.{f.name} must be an integer, got {value}")
            value = int(value)
        elif f.type in ("bool", bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{name}.{f.name} must be a bool, got {value!r}")
        kwargs[f.name] = value
    return section_cls(**kwargs)


def from_dict(raw: dict) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("config must be a JSON object")
    version = raw.get("schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"config schema_version {version!r}, expected {CONFIG_SCHEMA_VERSION}"
        )
    unknown = set(raw) - set(_SECTIONS) - {"schema_version"}
    if unknown:
        raise ConfigError(f"unknown top-level section(s): {sorted(unknown)}")
    sections = {}
    for name, cls in _SECTIONS.items():
        if name not in raw:
            raise ConfigError(f"missing section '{name}'")
        sections[name] = _build(cls, name, raw[name])
    cfg = Config(schema_version=version, **sections)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Range-check every constant. A silently absurd value is worse than a crash."""
    c = cfg.control
    checks: list[tuple[bool, str]] = [
        (cfg.detect.detect_hz > 0, "detect.detect_hz must be > 0"),
        (cfg.detect.infer_long_edge >= 320, "detect.infer_long_edge must be >= 320"),
        (0.0 < cfg.detect.conf_min < 1.0, "detect.conf_min must be in (0, 1)"),
        (
            0.0 <= cfg.detect.bbox_area_min < cfg.detect.bbox_area_max <= 1.0,
            "detect.bbox_area_min must be < bbox_area_max, both in [0, 1]",
        ),
        (cfg.centroid.mad_k > 0, "centroid.mad_k must be > 0"),
        (cfg.centroid.min_players >= 1, "centroid.min_players must be >= 1"),
        (
            cfg.centroid.spread_metric in ("iqr", "std"),
            "centroid.spread_metric must be 'iqr' or 'std'",
        ),
        (0.0 <= c.deadzone < 0.5, "control.deadzone must be in [0, 0.5)"),
        (c.omega_n > 0, "control.omega_n must be > 0"),
        (c.zeta > 0, "control.zeta must be > 0"),
        (c.max_pan_vel > 0, "control.max_pan_vel must be > 0"),
        (c.max_pan_accel > 0, "control.max_pan_accel must be > 0"),
        (c.lead_gain >= 0, "control.lead_gain must be >= 0"),
        (c.lead_max >= 0, "control.lead_max must be >= 0"),
        (c.lead_vel_tau > 0, "control.lead_vel_tau must be > 0"),
        (cfg.render.out_width > 0 and cfg.render.out_height > 0, "render output size must be > 0"),
        (
            cfg.render.crop_y_mode in ("pitch", "center", "fixed"),
            "render.crop_y_mode must be 'pitch', 'center' or 'fixed'",
        ),
        (0.0 <= cfg.render.crop_y_frac <= 1.0, "render.crop_y_frac must be in [0, 1]"),
        (0 <= cfg.render.crf <= 51, "render.crf must be in [0, 51]"),
        (cfg.render.preview_duration_s > 0, "render.preview_duration_s must be > 0"),
        (cfg.render.preview_height > 0, "render.preview_height must be > 0"),
        (
            cfg.events.transition_vel_threshold > 0,
            "events.transition_vel_threshold must be > 0",
        ),
        (
            cfg.events.attacking_third_debounce_s >= 0,
            "events.attacking_third_debounce_s must be >= 0",
        ),
        (cfg.phase2.deadzone_scale > 0, "phase2.deadzone_scale must be > 0"),
        (0 < cfg.phase2.fov_h_deg < 180, "phase2.fov_h_deg must be in (0, 180)"),
    ]
    problems = [msg for ok, msg in checks if not ok]
    if problems:
        raise ConfigError("; ".join(problems))


def load(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Load ``configs/default.json``, layer ``path`` over it, then ``overrides``.

    A user config need only contain the keys it changes.
    """
    with open(default_config_path(), encoding="utf-8") as fh:
        raw = json.load(fh)
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            user = json.load(fh)
        user.pop("schema_version", None)
        raw = deep_merge(raw, user)
    if overrides:
        raw = deep_merge(raw, overrides)
    return from_dict(raw)


def dump(cfg: Config, path: str | Path) -> None:
    """Write a fully resolved config -- the artefact Phase 2 consumes."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2, sort_keys=False)
        fh.write("\n")


__all__ = [
    "Config",
    "ConfigError",
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_CONFIG_PATH",
    "default_config_path",
    "deep_merge",
    "dump",
    "from_dict",
    "load",
    "validate",
]
