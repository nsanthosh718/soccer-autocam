"""Stage F -- crop and encode.

v1.0 is **pan only**: a fixed 1920x1080 window slid horizontally across the 4K
source. No scaling in the crop path, so no resampling artefacts, and nothing to
tune but the control law. Variable zoom is v1.1 (see the build spec) and would
require crop-then-scale here.

Vertical placement is fixed for the whole render. With a camera at 10 ft near
the halfway line the pitch sits in a band well below the frame's vertical
centre, so ``crop_y_mode: "pitch"`` centres the window on the calibrated quad
rather than on the frame -- otherwise a third of every output frame is sky.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av
import numpy as np

from . import ingest
from .calibrate import Pitch
from .config import Config
from .ingest import SourceInfo


class RenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderPlan:
    """Everything fixed for a render pass, resolved before a frame is decoded."""

    out_width: int
    out_height: int
    crop_width: int
    crop_height: int
    crop_y: int
    start_s: float
    end_s: float | None
    fps: float
    crf: int
    preset: str
    copy_audio: bool
    scaled: bool


def crop_y_for(cfg: Config, info: SourceInfo, pitch: Pitch | None) -> int:
    """Top edge of the crop window, in source pixels."""
    crop_h = cfg.render.out_height
    if crop_h > info.height:
        raise RenderError(
            f"crop height {crop_h} exceeds source height {info.height}"
        )
    mode = cfg.render.crop_y_mode
    if mode == "pitch" and pitch is not None:
        centre = pitch.centre_y()
    elif mode == "fixed":
        centre = cfg.render.crop_y_frac * info.height
    else:  # "center", or "pitch" with no calibration available
        centre = info.height / 2.0
    return int(round(float(np.clip(centre - crop_h / 2.0, 0, info.height - crop_h))))


def crop_rects(
    px: np.ndarray, cfg: Config, info: SourceInfo, pitch: Pitch | None
) -> np.ndarray:
    """(N, 4) int array of ``[x, y, w, h]`` crop rectangles, one per frame.

    x is forced even: an odd left edge would shift the chroma plane by half a
    sample on every frame and shimmer on flat colour.
    """
    crop_w = cfg.render.out_width
    crop_h = cfg.render.out_height
    if crop_w > info.width:
        raise RenderError(f"crop width {crop_w} exceeds source width {info.width}")
    y = crop_y_for(cfg, info, pitch)
    px = np.asarray(px, dtype=float).ravel()
    x = np.rint(px - crop_w / 2.0)
    x = np.clip(x, 0, info.width - crop_w)
    x = (x // 2) * 2
    n = px.size
    return np.stack(
        [
            x.astype(int),
            np.full(n, y, dtype=int),
            np.full(n, crop_w, dtype=int),
            np.full(n, crop_h, dtype=int),
        ],
        axis=1,
    )


def make_plan(
    cfg: Config,
    info: SourceInfo,
    pitch: Pitch | None,
    start_s: float | None,
    end_s: float | None,
    preview: bool,
) -> RenderPlan:
    r = cfg.render
    start = float(start_s or 0.0)
    if preview:
        end = start + r.preview_duration_s if end_s is None else min(end_s, start + r.preview_duration_s)
        out_h = r.preview_height
        out_w = int(round(r.out_width * out_h / r.out_height / 2)) * 2
        crf, preset, audio = r.preview_crf, r.preview_preset, r.preview_copy_audio
        scaled = (out_w, out_h) != (r.out_width, r.out_height)
    else:
        end = end_s
        out_w, out_h = r.out_width, r.out_height
        crf, preset, audio = r.crf, r.preset, r.copy_audio
        scaled = False
    return RenderPlan(
        out_width=out_w,
        out_height=out_h,
        crop_width=r.out_width,
        crop_height=r.out_height,
        crop_y=crop_y_for(cfg, info, pitch),
        start_s=start,
        end_s=end,
        fps=info.fps,
        crf=crf,
        preset=preset,
        copy_audio=audio,
        scaled=scaled,
    )


def default_output_path(video: str | Path, preview: bool) -> Path:
    video = Path(video)
    suffix = ".preview.mp4" if preview else ".autocam.mp4"
    return video.with_name(video.stem + suffix)


class _AudioPassthrough:
    """Remuxes the source audio stream into the output, never re-encoding it.

    The output stream must be declared *before* the first video packet is muxed:
    the muxer writes its header on that first packet, and a stream added
    afterwards has no time base -- libav then divides by zero and takes the
    process with it.
    """

    def __init__(self, video: str | Path, output: "av.container.OutputContainer", plan: RenderPlan):
        self.container = None
        self.in_stream = None
        self.out_stream = None
        self.plan = plan
        try:
            self.container = av.open(str(video))
            if not self.container.streams.audio:
                self.close()
                return
            self.in_stream = self.container.streams.audio[0]
            self.out_stream = output.add_stream_from_template(self.in_stream)
        except Exception as exc:  # noqa: BLE001 - audio never fails a render
            print(f"[render] audio not copied ({exc}); video is unaffected")
            self.close()

    @property
    def active(self) -> bool:
        return self.out_stream is not None

    def mux_into(self, output: "av.container.OutputContainer") -> bool:
        if not self.active:
            return False
        try:
            tb = float(self.in_stream.time_base)
            start_ticks = int(self.plan.start_s / tb) if self.plan.start_s else 0
            end_ticks = int(self.plan.end_s / tb) if self.plan.end_s is not None else None
            if start_ticks:
                self.container.seek(start_ticks, stream=self.in_stream, backward=True)
            for packet in self.container.demux(self.in_stream):
                if packet.pts is None or packet.dts is None:
                    continue
                if packet.pts < start_ticks:
                    continue
                if end_ticks is not None and packet.pts >= end_ticks:
                    break
                packet.pts -= start_ticks
                packet.dts -= start_ticks
                packet.stream = self.out_stream
                output.mux(packet)
            return True
        except Exception as exc:  # noqa: BLE001 - audio never fails a render
            print(f"[render] audio not copied ({exc}); video is unaffected")
            return False
        finally:
            self.close()

    def close(self) -> None:
        if self.container is not None:
            self.container.close()
            self.container = None


def render(
    video: str | Path,
    px: np.ndarray,
    cfg: Config,
    info: SourceInfo,
    pitch: Pitch | None = None,
    out_path: str | Path | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    preview: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, RenderPlan, int]:
    """Crop each frame to the control law's window and encode. Returns the path,
    the plan, and the number of frames written."""
    plan = make_plan(cfg, info, pitch, start_s, end_s, preview)
    rects = crop_rects(px, cfg, info, pitch)
    if len(rects) == 0:
        raise RenderError("no crop rectangles: the control trace is empty")
    out_path = Path(out_path) if out_path else default_output_path(video, preview)

    rate = Fraction(info.fps).limit_denominator(90000)
    written = 0
    total = int(round(((plan.end_s or info.duration_s) - plan.start_s) * info.fps))

    output = av.open(str(out_path), mode="w")
    audio = None
    try:
        stream = output.add_stream(cfg.render.codec, rate=rate)
        stream.width = plan.out_width
        stream.height = plan.out_height
        stream.pix_fmt = cfg.render.pix_fmt
        stream.options = {"crf": str(plan.crf), "preset": plan.preset}
        audio = (
            _AudioPassthrough(video, output, plan)
            if plan.copy_audio and info.has_audio
            else None
        )
        # Deliberately not setting stream.time_base: libx264 installs its own
        # (1/15360) when the encoder opens, and a stream time base assigned
        # beforehand makes the muxer reject packets once the GOP fills up.
        frame_time_base = Fraction(1, 1) / rate

        for ref in ingest.iter_frames(video, info, plan.start_s, plan.end_s):
            idx = min(ref.n, len(rects) - 1)
            x, y, w, h = rects[idx]
            rgb = ingest.to_rgb(ref.frame, info.rotation)
            crop = np.ascontiguousarray(rgb[y : y + h, x : x + w])
            out_frame = av.VideoFrame.from_ndarray(crop, format="rgb24")
            if plan.scaled:
                out_frame = out_frame.reformat(
                    width=plan.out_width, height=plan.out_height, format=cfg.render.pix_fmt
                )
            else:
                out_frame = out_frame.reformat(format=cfg.render.pix_fmt)
            out_frame.pts = written
            out_frame.time_base = frame_time_base
            for packet in stream.encode(out_frame):
                output.mux(packet)
            written += 1
            if progress is not None and written % 30 == 0:
                progress(written, max(total, written))

        for packet in stream.encode():
            output.mux(packet)

        if audio is not None:
            audio.mux_into(output)
    finally:
        if audio is not None:
            audio.close()
        output.close()

    if written == 0:
        raise RenderError(
            f"no frames written -- is --start ({plan.start_s}s) past the end of the file?"
        )
    return out_path, plan, written


def sample_frames(
    video: str | Path,
    count: int = 100,
    seed: int = 1234,
    out_dir: str | Path | None = None,
) -> dict:
    """Export ``count`` uniformly-random frames of a rendered file as PNGs.

    Exit criterion 1 is a manual count -- a human looks for the ball in each
    sampled frame. The spec requires the sampled indices to be logged so the
    measurement is reproducible; this writes them alongside the images, with the
    seed, so a re-run samples exactly the same frames.
    """
    video = Path(video)
    info = ingest.probe(video)
    total = ingest.frame_count(info)
    if count > total:
        count = total
    rng = np.random.default_rng(seed)
    indices = sorted(int(i) for i in rng.choice(total, size=count, replace=False))

    out_dir = Path(out_dir) if out_dir else video.with_name(video.stem + ".sample")
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(indices)
    written: list[dict] = []
    for ref in ingest.iter_frames(video, info):
        if ref.n not in wanted:
            continue
        image = ingest.to_rgb(ref.frame, info.rotation)
        path = out_dir / f"frame_{ref.n:08d}.png"
        with av.open(str(path), mode="w", format="image2") as container:
            stream = container.add_stream("png", rate=1)
            stream.width, stream.height = image.shape[1], image.shape[0]
            stream.pix_fmt = "rgb24"
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        written.append({"n": ref.n, "t": round(ref.t, 4), "file": path.name})
        wanted.discard(ref.n)
        if not wanted:
            break

    manifest = {
        "schema_version": 1,
        "video": str(video),
        "frames_total": total,
        "seed": seed,
        "requested": count,
        "indices": indices,
        "frames": written,
        "note": (
            "Exit criterion 1: count how many of these frames show the ball inside "
            "the crop. Ball-in-frame rate must be >= 92%."
        ),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


class Stopwatch:
    """Trivial wall-clock timer for the run report (exit criterion 4)."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.t0


__all__ = [
    "RenderError",
    "RenderPlan",
    "Stopwatch",
    "crop_rects",
    "crop_y_for",
    "default_output_path",
    "make_plan",
    "render",
    "sample_frames",
]
