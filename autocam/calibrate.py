"""Stage B -- pitch calibration: the four-corner quadrilateral and its mask.

Every detection whose bbox foot-point falls outside this polygon is discarded.
That single filter removes the dominant framing failure -- the crop drifting
toward spectators, warming-up substitutes, or the match on the next pitch over.
There is no heuristic substitute for it.

The quad also gives us a homography onto a unit pitch rectangle, which is how
the attacking-third lines in Stage G are derived rather than guessed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PITCH_SCHEMA_VERSION = 1


class CalibrationError(RuntimeError):
    pass


def pitch_path_for(video: str | Path) -> Path:
    return Path(str(video) + ".pitch.json")


def order_quad(points: np.ndarray) -> np.ndarray:
    """Order four clicked points as TL, TR, BR, BL so clicks can be in any order.

    Sorts counter-clockwise about the centroid in image coordinates (y down),
    which is clockwise on screen, then rotates so the top-left-most point leads.
    """
    pts = np.asarray(points, dtype=float).reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    order = np.argsort(angles)
    ordered = pts[order]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def polygon_area(quad: np.ndarray) -> float:
    x, y = np.asarray(quad, dtype=float).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def points_in_polygon(xs: np.ndarray, ys: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Vectorised crossing-number test. Returns a bool array over the points."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    quad = np.asarray(quad, dtype=float)
    inside = np.zeros(xs.shape, dtype=bool)
    n = len(quad)
    for i in range(n):
        x1, y1 = quad[i]
        x2, y2 = quad[(i + 1) % n]
        straddles = (y1 > ys) != (y2 > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (x2 - x1) * (ys - y1) / (y2 - y1) + x1
        inside ^= straddles & (xs < x_cross)
    return inside


def compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography mapping the four ``src`` points onto the four ``dst`` points."""
    src = np.asarray(src, dtype=float).reshape(4, 2)
    dst = np.asarray(dst, dtype=float).reshape(4, 2)
    a = np.zeros((8, 8))
    b = np.zeros(8)
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        a[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        a[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i] = u
        b[2 * i + 1] = v
    try:
        h = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as exc:
        raise CalibrationError("degenerate pitch quad: cannot build homography") from exc
    return np.append(h, 1.0).reshape(3, 3)


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    out = homo @ h.T
    w = out[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return out[:, :2] / w


# The unit pitch: (0,0) is the top-left corner of the quad, (1,1) the bottom-right.
_UNIT_QUAD = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


@dataclass(frozen=True)
class Pitch:
    """A calibrated pitch quadrilateral in source (display-oriented) pixels."""

    quad: np.ndarray            # (4, 2) ordered TL, TR, BR, BL
    source_width: int
    source_height: int
    created_at: float = 0.0
    note: str = ""

    @property
    def image_to_unit(self) -> np.ndarray:
        return compute_homography(self.quad, _UNIT_QUAD)

    @property
    def unit_to_image(self) -> np.ndarray:
        return compute_homography(_UNIT_QUAD, self.quad)

    def contains(self, xs, ys) -> np.ndarray:
        return points_in_polygon(np.asarray(xs), np.asarray(ys), self.quad)

    def centre_y(self) -> float:
        return float(self.quad[:, 1].mean())

    def third_lines_x(self) -> tuple[float, float]:
        """Image x of the two third-lines, sampled across the middle of the pitch.

        Perspective makes a third-line a slanted segment, not a vertical one; we
        take its x at the pitch's mid-depth. Crude, but derived from the quad
        rather than from a guess, and the events it feeds are candidates for
        manual review anyway.
        """
        pts = apply_homography(self.unit_to_image, [[1 / 3, 0.5], [2 / 3, 0.5]])
        lo, hi = float(pts[0, 0]), float(pts[1, 0])
        return (lo, hi) if lo <= hi else (hi, lo)

    def to_dict(self) -> dict:
        return {
            "schema_version": PITCH_SCHEMA_VERSION,
            "quad": [[float(x), float(y)] for x, y in self.quad],
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
            "created_at": self.created_at,
            "note": self.note,
        }


def make_pitch(
    corners, source_width: int, source_height: int, note: str = ""
) -> Pitch:
    quad = order_quad(np.asarray(corners, dtype=float))
    area = polygon_area(quad)
    if area <= 0 or not np.isfinite(quad).all():
        raise CalibrationError("degenerate pitch quad (zero area or non-finite corner)")
    frame_area = float(source_width) * float(source_height)
    if frame_area > 0 and area < 0.02 * frame_area:
        raise CalibrationError(
            f"pitch quad covers {100 * area / frame_area:.1f}% of the frame -- "
            "that is almost certainly a mis-click, not a pitch"
        )
    return Pitch(
        quad=quad,
        source_width=int(source_width),
        source_height=int(source_height),
        created_at=time.time(),
        note=note,
    )


def full_frame_pitch(source_width: int, source_height: int) -> Pitch:
    """Whole-frame quad. Escape hatch for headless runs and tests only.

    This disables the spectator filter, which is the point of calibration. It
    exists so the pipeline is scriptable, not because it is a reasonable
    production setting.
    """
    w, h = float(source_width), float(source_height)
    return make_pitch(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        source_width,
        source_height,
        note="full-frame: pitch filter disabled",
    )


def parse_corners(text: str) -> np.ndarray:
    """Parse ``x1,y1,x2,y2,x3,y3,x4,y4`` into a (4, 2) array."""
    parts = [p for p in text.replace(";", ",").replace(" ", ",").split(",") if p]
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise CalibrationError(f"could not parse corners: {text!r}") from exc
    if len(values) != 8:
        raise CalibrationError(f"expected 8 numbers (4 corners), got {len(values)}")
    return np.asarray(values, dtype=float).reshape(4, 2)


def save(pitch: Pitch, video: str | Path) -> Path:
    path = pitch_path_for(video)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(pitch.to_dict(), fh, indent=2)
        fh.write("\n")
    return path


def load(video: str | Path) -> Pitch:
    path = pitch_path_for(video)
    if not path.exists():
        raise CalibrationError(
            f"no calibration at {path}. Run `autocam calibrate {video}` first."
        )
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema_version") != PITCH_SCHEMA_VERSION:
        raise CalibrationError(
            f"{path}: schema_version {data.get('schema_version')!r}, "
            f"expected {PITCH_SCHEMA_VERSION}"
        )
    return Pitch(
        quad=order_quad(np.asarray(data["quad"], dtype=float)),
        source_width=int(data["source_width"]),
        source_height=int(data["source_height"]),
        created_at=float(data.get("created_at", 0.0)),
        note=str(data.get("note", "")),
    )


def load_if_present(video: str | Path) -> Pitch | None:
    return load(video) if pitch_path_for(video).exists() else None


def check_matches_source(pitch: Pitch, width: int, height: int) -> None:
    if (pitch.source_width, pitch.source_height) != (width, height):
        raise CalibrationError(
            f"calibration was captured at {pitch.source_width}x{pitch.source_height} "
            f"but the source is {width}x{height}. Re-run calibrate."
        )


def pick_corners_interactive(image: np.ndarray, title: str = "") -> np.ndarray:
    """Four-click corner picker. Requires matplotlib (``pip install -e .[gui]``)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise CalibrationError(
            "interactive calibration needs matplotlib: pip install 'soccer-autocam[gui]' "
            "-- or pass --corners / --full-frame"
        ) from exc

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(image)
    ax.set_title(
        (title + "\n" if title else "")
        + "Click the FOUR PITCH CORNERS (any order). "
        "Backspace undoes, Enter accepts."
    )
    ax.set_axis_off()
    fig.tight_layout()
    picked = plt.ginput(n=4, timeout=0, show_clicks=True)
    plt.close(fig)
    if len(picked) != 4:
        raise CalibrationError(f"expected 4 corners, got {len(picked)}")
    return np.asarray(picked, dtype=float)


__all__ = [
    "CalibrationError",
    "PITCH_SCHEMA_VERSION",
    "Pitch",
    "apply_homography",
    "check_matches_source",
    "compute_homography",
    "full_frame_pitch",
    "load",
    "load_if_present",
    "make_pitch",
    "order_quad",
    "parse_corners",
    "pick_corners_interactive",
    "pitch_path_for",
    "points_in_polygon",
    "polygon_area",
    "save",
]
