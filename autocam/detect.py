"""Stage C -- sparse person detection with an on-disk cache.

Two design points:

* Detection runs at ``detect.detect_hz`` (10 Hz), not per frame. At 30 fps that
  is every third frame; the centroid is interpolated between. Player clusters do
  not move fast enough to need more, and inference is the whole runtime budget.
* **Raw** detections are cached to ``<video>.detections.json``. The pitch-polygon
  and bbox-area filters are applied at *load* time, in ``centroid.py``, so that
  re-tuning any of them -- or any control constant -- is a seconds-long cycle
  instead of a re-run of inference.

The YOLO backend imports torch/ultralytics lazily, so every other module (and
the whole test suite) runs on a machine that has neither.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from . import ingest
from .config import Config
from .ingest import SourceInfo

DETECTIONS_SCHEMA_VERSION = 1


class DetectionError(RuntimeError):
    pass


class Detector(Protocol):
    """Anything that turns one RGB image into person boxes.

    Boxes are ``(N, 5)`` float arrays of ``[x1, y1, x2, y2, conf]`` in the
    coordinate system of the image it was handed (i.e. downscaled space);
    ``run_detection`` rescales them to source pixels.
    """

    def detect(self, image: np.ndarray) -> np.ndarray: ...

    def describe(self) -> dict: ...


class YoloDetector:
    """ultralytics YOLO11, COCO weights, person class only."""

    def __init__(self, cfg: Config):
        self.cfg = cfg.detect
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise DetectionError(
                    "the yolo backend needs ultralytics + torch: "
                    "pip install 'soccer-autocam[detect]'"
                ) from exc
            self._model = YOLO(self.cfg.model)
        return self._model

    def detect(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover - needs torch
        model = self._load()
        results = model.predict(
            source=image,
            imgsz=self.cfg.infer_long_edge,
            conf=self.cfg.conf_min,
            classes=[self.cfg.person_class_id],
            device=self.cfg.device,
            verbose=False,
        )
        if not results:
            return np.zeros((0, 5), dtype=float)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 5), dtype=float)
        xyxy = np.asarray(boxes.xyxy.cpu(), dtype=float)
        conf = np.asarray(boxes.conf.cpu(), dtype=float).reshape(-1, 1)
        return np.hstack([xyxy, conf])

    def describe(self) -> dict:
        return {
            "backend": "yolo",
            "model": self.cfg.model,
            "device": self.cfg.device,
            "conf_min": self.cfg.conf_min,
            "infer_long_edge": self.cfg.infer_long_edge,
        }


class SyntheticBlobDetector:
    """Connected-component detector for synthetic footage.

    Not a substitute for YOLO on real video -- it exists so the ingest ->
    detect -> centroid -> control -> render -> telemetry path can be exercised
    end-to-end without torch, in CI and in this repo's tests.
    """

    def __init__(self, cfg: Config, luma_threshold: int = 110, min_pixels: int = 12):
        self.cfg = cfg.detect
        self.luma_threshold = luma_threshold
        self.min_pixels = min_pixels

    def detect(self, image: np.ndarray) -> np.ndarray:
        from scipy import ndimage

        luma = image.astype(np.float32).mean(axis=2)
        mask = luma < self.luma_threshold
        labels, count = ndimage.label(mask)
        if count == 0:
            return np.zeros((0, 5), dtype=float)
        out = []
        for sl_y, sl_x in ndimage.find_objects(labels):
            h = sl_y.stop - sl_y.start
            w = sl_x.stop - sl_x.start
            if h * w < self.min_pixels:
                continue
            out.append([sl_x.start, sl_y.start, sl_x.stop, sl_y.stop, 0.9])
        if not out:
            return np.zeros((0, 5), dtype=float)
        return np.asarray(out, dtype=float)

    def describe(self) -> dict:
        return {
            "backend": "synthetic",
            "luma_threshold": self.luma_threshold,
            "min_pixels": self.min_pixels,
            "conf_min": self.cfg.conf_min,
            "infer_long_edge": self.cfg.infer_long_edge,
        }


_BACKENDS: dict[str, Callable[[Config], Detector]] = {
    "yolo": YoloDetector,
    "synthetic": SyntheticBlobDetector,
}


def get_detector(cfg: Config) -> Detector:
    backend = cfg.detect.backend
    if backend not in _BACKENDS:
        raise DetectionError(
            f"unknown detect.backend {backend!r}; known: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[backend](cfg)


def detections_path_for(video: str | Path) -> Path:
    return Path(str(video) + ".detections.json")


def detection_step(fps: float, detect_hz: float) -> int:
    """Frames between detections. Never 0, never faster than every frame."""
    if detect_hz <= 0:
        raise DetectionError("detect.detect_hz must be > 0")
    return max(1, int(round(fps / detect_hz)))


@dataclass
class DetectionCache:
    source: dict
    detector: dict
    fingerprint: str
    step: int
    timesteps: list[dict]

    @property
    def n_timesteps(self) -> int:
        return len(self.timesteps)

    def to_dict(self) -> dict:
        return {
            "schema_version": DETECTIONS_SCHEMA_VERSION,
            "source": self.source,
            "detector": self.detector,
            "fingerprint": self.fingerprint,
            "step": self.step,
            "timesteps": self.timesteps,
        }


def run_detection(
    video: str | Path,
    cfg: Config,
    info: SourceInfo | None = None,
    progress: Callable[[int, int], None] | None = None,
    detector: Detector | None = None,
) -> DetectionCache:
    """Detect at ``detect_hz`` across the whole file and return the raw cache."""
    info = info or ingest.probe(video)
    detector = detector or get_detector(cfg)
    step = detection_step(info.fps, cfg.detect.detect_hz)
    total = ingest.frame_count(info)
    timesteps: list[dict] = []

    for ref in ingest.iter_frames(video, info):
        if ref.n % step:
            continue
        image, sx, sy = ingest.to_rgb_scaled(
            ref.frame, info.rotation, cfg.detect.infer_long_edge
        )
        boxes = np.asarray(detector.detect(image), dtype=float).reshape(-1, 5)
        if len(boxes):
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            boxes = boxes[boxes[:, 4] >= cfg.detect.conf_min]
        timesteps.append(
            {
                "n": int(ref.n),
                "t": round(float(ref.t), 4),
                "boxes": [
                    [round(v, 1) for v in box[:4]] + [round(float(box[4]), 3)]
                    for box in boxes
                ],
            }
        )
        if progress is not None:
            progress(ref.n, total)

    return DetectionCache(
        source=info.to_dict(),
        detector=detector.describe(),
        fingerprint=cfg.detection_fingerprint(),
        step=step,
        timesteps=timesteps,
    )


def save_cache(cache: DetectionCache, video: str | Path) -> Path:
    path = detections_path_for(video)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cache.to_dict(), fh, separators=(",", ":"))
        fh.write("\n")
    return path


def load_cache(video: str | Path) -> DetectionCache:
    path = detections_path_for(video)
    if not path.exists():
        raise DetectionError(
            f"no cached detections at {path}. Run `autocam detect {video}` first."
        )
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema_version") != DETECTIONS_SCHEMA_VERSION:
        raise DetectionError(
            f"{path}: schema_version {data.get('schema_version')!r}, "
            f"expected {DETECTIONS_SCHEMA_VERSION}"
        )
    return DetectionCache(
        source=data["source"],
        detector=data["detector"],
        fingerprint=data.get("fingerprint", ""),
        step=int(data.get("step", 1)),
        timesteps=data["timesteps"],
    )


def cache_is_valid(cache: DetectionCache, cfg: Config, info: SourceInfo) -> tuple[bool, str]:
    """Is this cache reusable for the current config and source file?"""
    if cache.fingerprint != cfg.detection_fingerprint():
        return False, "detection constants changed since the cache was written"
    src = cache.source
    if (src.get("width"), src.get("height")) != (info.width, info.height):
        return False, "source resolution differs from the cached run"
    if abs(float(src.get("fps", 0.0)) - info.fps) > 1e-6:
        return False, "source frame rate differs from the cached run"
    if int(src.get("frames", 0)) != info.frames:
        return False, "source frame count differs from the cached run"
    return True, ""


def load_or_run(
    video: str | Path,
    cfg: Config,
    info: SourceInfo,
    progress: Callable[[int, int], None] | None = None,
    force: bool = False,
    detector: Detector | None = None,
) -> tuple[DetectionCache, bool]:
    """Return ``(cache, reused)``, re-running inference only when it is needed."""
    if not force and detections_path_for(video).exists():
        cache = load_cache(video)
        ok, why = cache_is_valid(cache, cfg, info)
        if ok:
            return cache, True
        print(f"[detect] re-running inference: {why}")
    cache = run_detection(video, cfg, info, progress=progress, detector=detector)
    save_cache(cache, video)
    return cache, False


__all__ = [
    "DETECTIONS_SCHEMA_VERSION",
    "DetectionCache",
    "DetectionError",
    "Detector",
    "SyntheticBlobDetector",
    "YoloDetector",
    "cache_is_valid",
    "detection_step",
    "detections_path_for",
    "get_detector",
    "load_cache",
    "load_or_run",
    "run_detection",
    "save_cache",
]
