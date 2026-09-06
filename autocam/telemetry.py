"""Stage G -- the JSON sidecar.

The schema is stable across phases: a downstream consumer must not be able to
tell whether a file came from the Phase 1 virtual camera or the Phase 2
motorized mount. Version it; never silently change a field's meaning.

The events are **proxies, not analysis**. They exist to produce candidate clip
timestamps for a human to review, and they say so in the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import __version__
from .calibrate import Pitch
from .centroid import Track
from .config import Config
from .control import ControlTrace
from .ingest import SourceInfo

TELEMETRY_SCHEMA_VERSION = 1

EVENTS_NOTE = (
    "Events are crude heuristics over the smoothed pan trace, not match analysis. "
    "Treat them as candidate timestamps for manual review; do not consume them "
    "unsupervised."
)


def telemetry_path_for(video: str | Path) -> Path:
    return Path(str(video) + ".telemetry.json")


def _window_confidence(valid: np.ndarray, index: int, fps: float) -> float:
    """Crude confidence: how much of the surrounding second was well-detected."""
    half = max(1, int(round(fps)))
    lo = max(0, index - half)
    hi = min(len(valid), index + half + 1)
    if hi <= lo:
        return 0.0
    frac = float(np.mean(valid[lo:hi]))
    return round(float(np.clip(frac, 0.0, 1.0)), 2)


def detect_attacking_third_entries(
    px: np.ndarray, valid: np.ndarray, cfg: Config, pitch: Pitch | None, fps: float
) -> list[dict]:
    """Smoothed pan centre crossing a third-line derived from the pitch quad."""
    if pitch is None or len(px) == 0:
        return []
    lo_x, hi_x = pitch.third_lines_x()
    debounce = cfg.events.attacking_third_debounce_s * fps
    events: list[dict] = []
    last_fire = {"left": -np.inf, "right": -np.inf}

    inside_right = px[0] > hi_x
    inside_left = px[0] < lo_x
    for i in range(1, len(px)):
        now_right = px[i] > hi_x
        now_left = px[i] < lo_x
        for direction, entered in (("right", now_right and not inside_right),
                                   ("left", now_left and not inside_left)):
            if entered and (i - last_fire[direction]) >= debounce:
                last_fire[direction] = i
                events.append(
                    {
                        "t": round(i / fps, 3),
                        "n": int(i),
                        "type": "attacking_third_entry",
                        "direction": direction,
                        "confidence": _window_confidence(valid, i, fps),
                        "heuristic": True,
                    }
                )
        inside_right, inside_left = now_right, now_left
    return events


def detect_transitions(
    cx_vel: np.ndarray, valid: np.ndarray, cfg: Config, frame_width: int, fps: float
) -> list[dict]:
    """Sustained high centroid velocity -- a counter-attack, probably."""
    if len(cx_vel) == 0:
        return []
    threshold = cfg.events.transition_vel_threshold * frame_width
    min_len = max(1, int(round(cfg.events.transition_min_duration_s * fps)))
    debounce = cfg.events.transition_debounce_s * fps

    fast = np.abs(cx_vel) >= threshold
    events: list[dict] = []
    last_fire = -np.inf
    run_start = None
    for i, flag in enumerate(np.append(fast, False)):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            if (i - run_start) >= min_len and (run_start - last_fire) >= debounce:
                last_fire = run_start
                segment = cx_vel[run_start:i]
                peak = float(segment[np.argmax(np.abs(segment))])
                events.append(
                    {
                        "t": round(run_start / fps, 3),
                        "n": int(run_start),
                        "type": "transition",
                        "direction": "right" if peak > 0 else "left",
                        "duration_s": round((i - run_start) / fps, 3),
                        "peak_vel_fw_per_s": round(peak / frame_width, 4),
                        "confidence": _window_confidence(valid, run_start, fps),
                        "heuristic": True,
                    }
                )
            run_start = None
    return events


def build(
    video: str | Path,
    info: SourceInfo,
    cfg: Config,
    pitch: Pitch | None,
    track: Track,
    trace: ControlTrace,
    crops: np.ndarray,
    quality: dict,
    extra: dict | None = None,
) -> dict:
    """Assemble the sidecar document."""
    fps = info.fps
    n_frames = len(trace.px)
    frames = []
    for i in range(n_frames):
        x, y, w, h = (int(v) for v in crops[i])
        frames.append(
            {
                "n": i,
                "t": round(i / fps, 4),
                "cx": round(float(track.cx[i]), 2),
                "spread": round(float(track.spread[i]), 2),
                "px": round(float(trace.px[i]), 2),
                "crop": [x, y, w, h],
                "n_players": int(track.n_players[i]),
                "valid": bool(track.valid[i]),
            }
        )

    events = detect_attacking_third_entries(trace.px, track.valid, cfg, pitch, fps)
    events += detect_transitions(trace.cx_vel, track.valid, cfg, info.width, fps)
    events.sort(key=lambda e: e["t"])

    doc = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "generated_by": {"tool": "soccer-autocam", "version": __version__, "phase": 1},
        "source": {
            "path": str(video),
            "width": info.width,
            "height": info.height,
            "fps": fps,
            "frames": int(info.frames),
            "rotation": info.rotation,
        },
        "config_hash": cfg.hash(),
        "pitch_quad": [[round(float(x), 1), round(float(y), 1)] for x, y in pitch.quad]
        if pitch is not None
        else None,
        "frames": frames,
        "events": events,
        "events_note": EVENTS_NOTE,
        "quality": quality,
    }
    if extra:
        doc.update(extra)
    return doc


def save(doc: dict, video: str | Path) -> Path:
    path = telemetry_path_for(video)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"))
        fh.write("\n")
    return path


def load(video: str | Path) -> dict:
    with open(telemetry_path_for(video), encoding="utf-8") as fh:
        return json.load(fh)


__all__ = [
    "EVENTS_NOTE",
    "TELEMETRY_SCHEMA_VERSION",
    "build",
    "detect_attacking_third_entries",
    "detect_transitions",
    "load",
    "save",
    "telemetry_path_for",
]
