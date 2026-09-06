"""Stage A -- probe, decode, frame iteration.

Two things here are load-bearing:

1. **Rotation metadata is honoured before any coordinate math.** libav hands us
   the stored (unrotated) pixels; only the ffmpeg *CLI* auto-rotates. An iPhone
   clip carrying a 90-degree display matrix decodes as portrait-shaped pixels,
   and every x coordinate downstream would be wrong in a way that looks like a
   tracking bug. ``SourceInfo.width``/``height`` are always *display* dimensions
   and ``to_rgb`` always returns display-oriented pixels.
2. **Sub-4K sources are rejected** per the capture assumptions, unless the
   caller explicitly lowers ``ingest.min_source_*`` (the tests do).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from .config import Config


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceInfo:
    """Everything downstream needs to know about the source file."""

    path: str
    width: int          # display width, post-rotation
    height: int         # display height, post-rotation
    fps: float
    frames: int
    duration_s: float
    rotation: int       # degrees clockwise to apply to decoded pixels: 0/90/180/270
    has_audio: bool
    probe_backend: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames": self.frames,
            "duration_s": self.duration_s,
            "rotation": self.rotation,
            "has_audio": self.has_audio,
        }


@dataclass(frozen=True)
class FrameRef:
    """A decoded frame plus its position in the source timeline."""

    n: int
    t: float
    frame: "av.VideoFrame"


def _normalise_rotation(value: float | int | str | None) -> int:
    if value is None:
        return 0
    try:
        deg = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    deg %= 360
    if deg % 90 != 0:
        # Non-right-angle display matrices are not something a phone produces;
        # refusing is better than silently mis-cropping.
        raise IngestError(f"unsupported rotation metadata: {value!r}")
    return deg


def _probe_ffprobe(path: str) -> SourceInfo | None:
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    cmd = [exe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise IngestError(f"ffprobe failed on {path}: {exc}") from exc
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise IngestError(f"no video stream in {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    rotation = 0
    tag = (video.get("tags") or {}).get("rotate")
    if tag is not None:
        rotation = _normalise_rotation(tag)
    for sd in video.get("side_data_list") or []:
        if "rotation" in sd:
            # ffprobe reports the display-matrix rotation counter-clockwise.
            rotation = _normalise_rotation(-float(sd["rotation"]))
            break

    stored_w, stored_h = int(video["width"]), int(video["height"])
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate")) or 0.0
    duration = float(video.get("duration") or (data.get("format", {}).get("duration") or 0.0))
    frames = int(video.get("nb_frames") or 0)
    if frames <= 0 and fps > 0 and duration > 0:
        frames = int(round(duration * fps))
    if duration <= 0 and fps > 0 and frames > 0:
        duration = frames / fps
    w, h = (stored_h, stored_w) if rotation in (90, 270) else (stored_w, stored_h)
    return SourceInfo(path, w, h, fps, frames, duration, rotation, has_audio, "ffprobe")


def _parse_rate(value) -> float:
    if not value:
        return 0.0
    try:
        if isinstance(value, str) and "/" in value:
            num, den = value.split("/")
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _probe_pyav(path: str) -> SourceInfo:
    with av.open(path) as container:
        if not container.streams.video:
            raise IngestError(f"no video stream in {path}")
        stream = container.streams.video[0]
        has_audio = bool(container.streams.audio)
        rotation = _extract_rotation_pyav(stream)
        stored_w = int(stream.codec_context.width)
        stored_h = int(stream.codec_context.height)
        rate = stream.average_rate or stream.guessed_rate or Fraction(0, 1)
        fps = float(rate)
        duration = 0.0
        if stream.duration is not None and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / av.time_base
        frames = int(stream.frames or 0)
        if frames <= 0 and fps > 0 and duration > 0:
            frames = int(round(duration * fps))
        if duration <= 0 and fps > 0 and frames > 0:
            duration = frames / fps
    w, h = (stored_h, stored_w) if rotation in (90, 270) else (stored_w, stored_h)
    return SourceInfo(path, w, h, fps, frames, duration, rotation, has_audio, "pyav")


def _extract_rotation_pyav(stream) -> int:
    tag = stream.metadata.get("rotate")
    if tag is not None:
        return _normalise_rotation(tag)
    side = getattr(stream, "side_data", None)
    if side:
        for key in ("DISPLAYMATRIX", "Display Matrix"):
            try:
                entry = side.get(key)
            except (TypeError, AttributeError):
                entry = None
            if entry is None:
                continue
            rot = getattr(entry, "rotation", None)
            if rot is not None:
                return _normalise_rotation(-float(rot))
    return 0


def probe(path: str | Path) -> SourceInfo:
    """Resolution, fps, duration and rotation. ffprobe when present, PyAV otherwise."""
    path = str(path)
    if not Path(path).exists():
        raise IngestError(f"no such file: {path}")
    info = _probe_ffprobe(path)
    if info is None or info.fps <= 0 or info.width <= 0:
        info = _probe_pyav(path)
    if info.fps <= 0:
        raise IngestError(f"could not determine frame rate for {path}")
    return info


def check_resolution(info: SourceInfo, cfg: Config) -> None:
    """Reject sub-4K sources. Framing quality below this is not recoverable."""
    if info.width < cfg.ingest.min_source_width or info.height < cfg.ingest.min_source_height:
        raise IngestError(
            f"source is {info.width}x{info.height}; "
            f"minimum is {cfg.ingest.min_source_width}x{cfg.ingest.min_source_height}. "
            "Re-record at 4K, or lower ingest.min_source_* in the config if you know why."
        )


def apply_rotation(img: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate decoded pixels into display orientation.

    ``rotation`` is degrees *clockwise* to apply, so np.rot90 (counter-clockwise)
    is called ``4 - rotation/90`` times.
    """
    if rotation == 0:
        return img
    if rotation not in (90, 180, 270):
        raise IngestError(f"unsupported rotation: {rotation}")
    return np.ascontiguousarray(np.rot90(img, k=(4 - rotation // 90) % 4))


def to_rgb(frame: "av.VideoFrame", rotation: int = 0) -> np.ndarray:
    """Decoded frame -> display-oriented HxWx3 uint8 RGB."""
    return apply_rotation(frame.to_ndarray(format="rgb24"), rotation)


def to_rgb_scaled(
    frame: "av.VideoFrame", rotation: int, long_edge: int
) -> tuple[np.ndarray, float, float]:
    """Display-oriented RGB downscaled so its long edge is <= ``long_edge``.

    Returns ``(image, scale_x, scale_y)`` where multiplying an x/y coordinate in
    the returned image by the scale factor maps it back to source pixels.
    Scaling is done by libswscale on the decoded frame (cheap) rather than in
    numpy, and rotation is applied afterwards on the smaller array.
    """
    stored_w, stored_h = frame.width, frame.height
    disp_w, disp_h = (stored_h, stored_w) if rotation in (90, 270) else (stored_w, stored_h)
    longest = max(disp_w, disp_h)
    if long_edge <= 0 or longest <= long_edge:
        img = to_rgb(frame, rotation)
        return img, 1.0, 1.0
    ratio = long_edge / float(longest)
    # Keep dimensions even; some swscale paths dislike odd sizes.
    target_disp_w = max(2, int(round(disp_w * ratio / 2)) * 2)
    target_disp_h = max(2, int(round(disp_h * ratio / 2)) * 2)
    if rotation in (90, 270):
        target_stored_w, target_stored_h = target_disp_h, target_disp_w
    else:
        target_stored_w, target_stored_h = target_disp_w, target_disp_h
    small = frame.reformat(
        width=target_stored_w, height=target_stored_h, format="rgb24"
    ).to_ndarray()
    img = apply_rotation(small, rotation)
    return img, disp_w / float(img.shape[1]), disp_h / float(img.shape[0])


def iter_frames(
    path: str | Path,
    info: SourceInfo | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
) -> Iterator[FrameRef]:
    """Yield frames in ``[start_s, end_s)``, with source-absolute index and time.

    Frames are yielded undecoded-to-numpy so callers that only need every Nth
    frame (detection) do not pay ``to_ndarray`` on the ones they discard.
    """
    path = str(path)
    info = info or probe(path)
    fps = info.fps
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        if start_s and start_s > 0:
            # Seek backwards to the nearest keyframe; frames before start_s are
            # decoded and dropped so the yielded indices stay source-absolute.
            offset = int(start_s / float(time_base))
            container.seek(offset, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * time_base)
            if start_s is not None and t < start_s - 0.5 / fps:
                continue
            if end_s is not None and t >= end_s:
                break
            yield FrameRef(n=int(round(t * fps)), t=t, frame=frame)


def frame_at(path: str | Path, index: int = 0, info: SourceInfo | None = None) -> np.ndarray:
    """Decode a single frame as display-oriented RGB (used by calibration)."""
    info = info or probe(path)
    target_t = index / info.fps if info.fps else 0.0
    for ref in iter_frames(path, info, start_s=max(0.0, target_t - 1e-6)):
        return to_rgb(ref.frame, info.rotation)
    raise IngestError(f"could not decode frame {index} of {path}")


def frame_count(info: SourceInfo) -> int:
    if info.frames > 0:
        return info.frames
    if info.duration_s > 0 and info.fps > 0:
        return int(math.floor(info.duration_s * info.fps))
    raise IngestError(f"could not determine frame count for {info.path}")


__all__ = [
    "FrameRef",
    "IngestError",
    "SourceInfo",
    "apply_rotation",
    "check_resolution",
    "frame_at",
    "frame_count",
    "iter_frames",
    "probe",
    "to_rgb",
    "to_rgb_scaled",
]
