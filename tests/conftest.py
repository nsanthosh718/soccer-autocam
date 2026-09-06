"""Synthetic match footage so the whole pipeline is exercisable without torch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from autocam import config as config_mod

WIDTH, HEIGHT, FPS = 640, 360, 30.0
DURATION_S = 12.0
N_PLAYERS = 14

# A pitch band that excludes the top of the frame (where the spectators are).
PITCH_QUAD = [[40.0, 300.0], [600.0, 300.0], [560.0, 140.0], [80.0, 140.0]]


def true_centre(t: float) -> float:
    """Play parks in the middle, breaks right over 2 s, holds, drifts back."""
    if t < 3.0:
        return 320.0
    if t < 5.0:
        return 320.0 + (500.0 - 320.0) * (t - 3.0) / 2.0
    if t < 8.0:
        return 500.0
    return 500.0 - (500.0 - 260.0) * min(1.0, (t - 8.0) / 3.0)


def _draw(img: np.ndarray, x0: int, y0: int, w: int, h: int, value: int = 20) -> None:
    x0 = max(0, min(WIDTH - w, x0))
    y0 = max(0, min(HEIGHT - h, y0))
    img[y0 : y0 + h, x0 : x0 + w] = value


def synthetic_frame(t: float, rng: np.random.Generator) -> np.ndarray:
    """One frame: a player cluster on the pitch, a spectator off it, a foreground blob."""
    img = np.full((HEIGHT, WIDTH, 3), 210, dtype=np.uint8)
    centre = true_centre(t)

    # Player cluster: tight spread, deterministic per-player offsets.
    offsets = np.linspace(-70, 70, N_PLAYERS)
    for i, off in enumerate(offsets):
        # 8 px of jitter, seeded, so the centroid is not perfectly static.
        jitter = 8.0 * np.sin(2.0 * np.pi * (t * 0.7 + i / N_PLAYERS))
        x = int(round(centre + off + jitter))
        y = 200 + (i % 5) * 14
        _draw(img, x, y, 6, 14)

    # Stranded goalkeeper: a legitimate detection far from play. A mean would
    # chase it; the median must not.
    _draw(img, 600, 240, 6, 14)

    # Spectator on the touchline, above the pitch quad -> pitch filter must drop it.
    _draw(img, 40, 96, 8, 20)

    # Foreground blob walking past the lens, inside the quad -> area filter drops it.
    _draw(img, 560, 170, 60, 120)
    return img


def write_synthetic_video(path: Path, duration_s: float = DURATION_S) -> Path:
    rng = np.random.default_rng(0)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(int(FPS), 1))
    stream.width, stream.height = WIDTH, HEIGHT
    stream.pix_fmt = "yuv420p"
    # Lossless-ish: the synthetic detector thresholds on luma, and heavy
    # quantisation of hard-edged blobs would make detections flicker.
    stream.options = {"crf": "12", "preset": "veryfast"}
    n_frames = int(round(duration_s * FPS))
    for n in range(n_frames):
        img = synthetic_frame(n / FPS, rng)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = n
        frame.time_base = Fraction(1, int(FPS))
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def write_av_clip(path: Path, duration_s: float = 2.0) -> Path:
    """A short clip that also has an audio track, to exercise audio passthrough."""
    container = av.open(str(path), mode="w")
    video = container.add_stream("libx264", rate=Fraction(int(FPS), 1))
    video.width, video.height, video.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    video.options = {"crf": "20", "preset": "veryfast"}
    audio = container.add_stream("aac", rate=48000)

    n_frames = int(round(duration_s * FPS))
    rng = np.random.default_rng(1)
    for n in range(n_frames):
        frame = av.VideoFrame.from_ndarray(synthetic_frame(n / FPS, rng), format="rgb24")
        frame.pts = n
        frame.time_base = Fraction(1, int(FPS))
        for packet in video.encode(frame):
            container.mux(packet)

    sample_rate = 48000
    total = int(sample_rate * duration_s)
    pos = 0
    while pos < total:
        count = min(1024, total - pos)
        t = (np.arange(pos, pos + count) / sample_rate).astype(np.float32)
        tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(np.stack([tone, tone]), format="fltp", layout="stereo")
        frame.sample_rate = sample_rate
        frame.pts = pos
        frame.time_base = Fraction(1, sample_rate)
        for packet in audio.encode(frame):
            container.mux(packet)
        pos += count

    for packet in video.encode():
        container.mux(packet)
    for packet in audio.encode():
        container.mux(packet)
    container.close()
    return path


@dataclass
class Fixture:
    video: Path
    cfg: "config_mod.Config"
    quad: list


# Everything that retargets the shipped defaults at the small synthetic source.
# Nothing here changes the control law -- those constants are tested as shipped.
TEST_OVERRIDES = {
    "ingest": {"min_source_width": WIDTH, "min_source_height": HEIGHT},
    "detect": {
        "backend": "synthetic",
        "infer_long_edge": WIDTH,
        "bbox_area_min": 1.0e-4,
        "bbox_area_max": 0.02,
    },
    "render": {
        "out_width": 320,
        "out_height": 180,
        "preview_duration_s": 2.0,
        "preview_height": 90,
    },
}


def make_config(**extra) -> "config_mod.Config":
    """Default config, retargeted at the small synthetic source."""
    overrides = {section: dict(values) for section, values in TEST_OVERRIDES.items()}
    for section, values in extra.items():
        overrides.setdefault(section, {}).update(values)
    return config_mod.load(overrides=overrides)


def write_test_config(path: Path) -> Path:
    """The same overrides as a --config file, for exercising the CLI."""
    path.write_text(json.dumps(TEST_OVERRIDES, indent=2))
    return path


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("footage") / "match.mp4"
    return write_synthetic_video(path)


@pytest.fixture(scope="session")
def av_clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("footage_av") / "clip_with_audio.mp4"
    return write_av_clip(path)


@pytest.fixture
def cfg():
    return make_config()


@pytest.fixture
def config_file(tmp_path) -> Path:
    return write_test_config(tmp_path / "test-config.json")
