"""Stage D -- robust centroid and spread.

The system does not track the ball. It tracks the *median* x of detected player
foot-points inside the pitch polygon.

Median, not mean: a goalkeeper stranded at the far end is a legitimate detection
and would drag a mean toward an empty half of the pitch. MAD rejection on top of
that removes the stragglers the median alone still leans toward.

``spread`` (the IQR of surviving x positions) is computed and carried through
telemetry even though v1.0 does not zoom -- it is the input the v1.1 variable
zoom needs, and recording it now costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibrate import Pitch
from .config import Config
from .detect import DetectionCache
from .ingest import SourceInfo


@dataclass(frozen=True)
class Timestep:
    """One detection instant, after filtering and robust aggregation."""

    n: int
    t: float
    cx: float
    spread: float
    n_players: int
    n_raw: int
    valid: bool


@dataclass
class Track:
    """Per-frame centroid track, interpolated up from the detection timesteps."""

    fps: float
    frames: int
    cx: np.ndarray            # (frames,) source-pixel x of the play centroid
    spread: np.ndarray        # (frames,) IQR of player x, source pixels
    n_players: np.ndarray     # (frames,) int, surviving detections at the nearest timestep
    valid: np.ndarray         # (frames,) bool
    timesteps: list[Timestep] = field(default_factory=list)

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.frames, dtype=float) / self.fps


def foot_points(boxes: np.ndarray) -> np.ndarray:
    """Bottom-centre of each bbox -- where the player meets the ground.

    Using the foot point rather than the box centre is what makes the pitch
    polygon test meaningful: a tall spectator's box centre can sit over the
    pitch while their feet are clearly on the touchline.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 5)
    if len(boxes) == 0:
        return np.zeros((0, 2), dtype=float)
    x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    y = boxes[:, 3]
    return np.stack([x, y], axis=1)


def filter_boxes(
    boxes: np.ndarray, cfg: Config, pitch: Pitch | None, frame_area: float
) -> np.ndarray:
    """Drop boxes by confidence, area fraction, and pitch containment."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 5)
    if len(boxes) == 0:
        return boxes
    keep = boxes[:, 4] >= cfg.detect.conf_min

    w = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    area_frac = (w * h) / float(frame_area)
    keep &= area_frac >= cfg.detect.bbox_area_min
    keep &= area_frac <= cfg.detect.bbox_area_max

    if pitch is not None:
        feet = foot_points(boxes)
        keep &= pitch.contains(feet[:, 0], feet[:, 1])
    return boxes[keep]


def robust_centroid(xs: np.ndarray, cfg: Config) -> tuple[float, float, int]:
    """Median x after MAD rejection, plus spread. Returns ``(cx, spread, n)``."""
    xs = np.asarray(xs, dtype=float).ravel()
    if xs.size == 0:
        return float("nan"), float("nan"), 0

    median = float(np.median(xs))
    mad = float(np.median(np.abs(xs - median)))
    if mad > 0:
        survivors = xs[np.abs(xs - median) <= cfg.centroid.mad_k * mad]
        if survivors.size == 0:  # every point rejected: keep the originals
            survivors = xs
    else:
        # Degenerate spread (identical or near-identical positions): nothing to
        # reject, and dividing by a zero MAD would discard the whole cluster.
        survivors = xs

    cx = float(np.median(survivors))
    if cfg.centroid.spread_metric == "std":
        spread = float(np.std(survivors))
    else:
        q75, q25 = np.percentile(survivors, [75, 25])
        spread = float(q75 - q25)
    return cx, spread, int(survivors.size)


def analyse_timesteps(
    cache: DetectionCache, cfg: Config, pitch: Pitch | None, info: SourceInfo
) -> list[Timestep]:
    """Filter + aggregate every cached detection instant."""
    frame_area = float(info.width) * float(info.height)
    out: list[Timestep] = []
    for entry in cache.timesteps:
        raw = np.asarray(entry["boxes"], dtype=float).reshape(-1, 5)
        kept = filter_boxes(raw, cfg, pitch, frame_area)
        feet = foot_points(kept)
        cx, spread, n = robust_centroid(feet[:, 0], cfg)
        valid = n >= cfg.centroid.min_players
        out.append(
            Timestep(
                n=int(entry["n"]),
                t=float(entry["t"]),
                cx=cx,
                spread=spread,
                n_players=n,
                n_raw=int(len(raw)),
                valid=bool(valid),
            )
        )
    return out


def build_track(
    timesteps: list[Timestep], cfg: Config, info: SourceInfo, frames: int | None = None
) -> Track:
    """Interpolate timesteps up to a per-frame track.

    Invalid timesteps (fewer than ``centroid.min_players`` survivors) **hold the
    previous valid target** rather than contributing a low-confidence position.
    Interpolation is then linear between consecutive timesteps, so recovery from
    a held run is a one-timestep ramp -- which the deadzone and the velocity and
    acceleration clamps in Stage E are there to absorb.
    """
    fps = info.fps
    frames = int(frames if frames is not None else max(info.frames, 1))

    if not timesteps:
        centre = info.width / 2.0
        return Track(
            fps=fps,
            frames=frames,
            cx=np.full(frames, centre),
            spread=np.zeros(frames),
            n_players=np.zeros(frames, dtype=int),
            valid=np.zeros(frames, dtype=bool),
            timesteps=[],
        )

    ns = np.array([ts.n for ts in timesteps], dtype=float)
    valid_ts = np.array([ts.valid for ts in timesteps], dtype=bool)
    counts = np.array([ts.n_players for ts in timesteps], dtype=int)

    first_valid = int(np.argmax(valid_ts)) if valid_ts.any() else -1
    if first_valid < 0:
        held_cx = np.full(len(timesteps), info.width / 2.0)
        held_spread = np.zeros(len(timesteps))
    else:
        held_cx = np.empty(len(timesteps))
        held_spread = np.empty(len(timesteps))
        last_cx = timesteps[first_valid].cx
        last_spread = timesteps[first_valid].spread
        for i, ts in enumerate(timesteps):
            if ts.valid and np.isfinite(ts.cx):
                last_cx, last_spread = ts.cx, ts.spread
            held_cx[i] = last_cx
            held_spread[i] = last_spread

    frame_idx = np.arange(frames, dtype=float)
    cx = np.interp(frame_idx, ns, held_cx)
    spread = np.interp(frame_idx, ns, held_spread)

    # Counts and validity are per-timestep facts; step them, never blend them.
    nearest = np.clip(np.searchsorted(ns, frame_idx, side="left"), 0, len(ns) - 1)
    prev = np.clip(nearest - 1, 0, len(ns) - 1)
    take_prev = np.abs(frame_idx - ns[prev]) <= np.abs(ns[nearest] - frame_idx)
    nearest = np.where(take_prev, prev, nearest)
    n_players = counts[nearest]

    # A frame that lands exactly on a timestep takes that timestep's validity;
    # an interpolated frame is only valid if both timesteps bracketing it are.
    left = np.clip(np.searchsorted(ns, frame_idx, side="right") - 1, 0, len(ns) - 1)
    right = np.clip(left + 1, 0, len(ns) - 1)
    on_timestep = frame_idx == ns[left]
    frame_valid = np.where(on_timestep, valid_ts[left], valid_ts[left] & valid_ts[right])
    frame_valid &= frame_idx >= ns[0]

    return Track(
        fps=fps,
        frames=frames,
        cx=cx,
        spread=spread,
        n_players=n_players,
        valid=frame_valid,
        timesteps=list(timesteps),
    )


def quality_report(track: Track, cfg: Config) -> dict:
    """Fail visible: the numbers that say whether this run should be trusted."""
    frames = max(track.frames, 1)
    invalid = ~track.valid
    invalid_pct = 100.0 * float(invalid.sum()) / frames

    longest = 0
    run = 0
    for flag in invalid:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    longest_s = longest / track.fps if track.fps else 0.0

    counted = [ts.n_players for ts in track.timesteps]
    mean_players = float(np.mean(counted)) if counted else 0.0

    warnings: list[str] = []
    if longest_s > cfg.quality.invalid_run_warn_s:
        warnings.append(
            f"longest run with fewer than {cfg.centroid.min_players} players was "
            f"{longest_s:.1f}s (> {cfg.quality.invalid_run_warn_s:.1f}s): "
            "framing was held, not tracked, for that stretch"
        )
    if invalid_pct > cfg.quality.invalid_frame_pct_warn:
        warnings.append(
            f"{invalid_pct:.1f}% of frames were low-confidence "
            f"(> {cfg.quality.invalid_frame_pct_warn:.1f}%)"
        )
    if mean_players < cfg.quality.mean_players_warn:
        warnings.append(
            f"mean {mean_players:.1f} players detected per timestep "
            f"(< {cfg.quality.mean_players_warn:.1f}): check camera height and the pitch quad"
        )

    return {
        "invalid_frame_pct": round(invalid_pct, 2),
        "longest_invalid_run_s": round(longest_s, 2),
        "mean_players_detected": round(mean_players, 2),
        "timesteps": len(track.timesteps),
        "invalid_timesteps": int(sum(1 for ts in track.timesteps if not ts.valid)),
        "warnings": warnings,
    }


__all__ = [
    "Timestep",
    "Track",
    "analyse_timesteps",
    "build_track",
    "filter_boxes",
    "foot_points",
    "quality_report",
    "robust_centroid",
]
