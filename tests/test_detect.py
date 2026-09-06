import json

import numpy as np
import pytest

from autocam import detect
from autocam import ingest
from conftest import make_config


def test_detection_step_matches_detect_hz():
    assert detect.detection_step(30.0, 10.0) == 3
    assert detect.detection_step(60.0, 10.0) == 6
    assert detect.detection_step(59.94, 10.0) == 6
    assert detect.detection_step(10.0, 30.0) == 1        # never faster than every frame


def test_unknown_backend_is_refused():
    cfg = make_config(detect={"backend": "magic"})
    with pytest.raises(detect.DetectionError, match="unknown detect.backend"):
        detect.get_detector(cfg)


def test_synthetic_detector_finds_the_blobs():
    cfg = make_config()
    image = np.full((100, 200, 3), 210, dtype=np.uint8)
    image[40:60, 30:40] = 20
    image[50:70, 150:158] = 10
    boxes = detect.get_detector(cfg).detect(image)
    assert boxes.shape == (2, 5)
    assert sorted(np.round(boxes[:, 0]).tolist()) == [30.0, 150.0]


def test_run_detection_samples_at_detect_hz(synthetic_video):
    cfg = make_config()
    info = ingest.probe(synthetic_video)
    cache = detect.run_detection(synthetic_video, cfg, info)
    assert cache.step == 3
    assert cache.n_timesteps == 120                     # 12 s at 10 Hz
    assert [ts["n"] for ts in cache.timesteps[:3]] == [0, 3, 6]
    assert all(len(ts["boxes"]) > 10 for ts in cache.timesteps)


def test_cache_roundtrip_and_reuse(tmp_path, synthetic_video):
    import shutil

    cfg = make_config()
    video = tmp_path / "clip.mp4"
    shutil.copy(synthetic_video, video)
    info = ingest.probe(video)

    cache, reused = detect.load_or_run(video, cfg, info)
    assert not reused
    assert detect.detections_path_for(video).exists()

    again, reused = detect.load_or_run(video, cfg, info)
    assert reused
    assert again.n_timesteps == cache.n_timesteps


def test_control_tuning_does_not_invalidate_the_cache(tmp_path, synthetic_video):
    """The whole point of caching: re-tuning must not re-run inference."""
    import shutil

    video = tmp_path / "clip.mp4"
    shutil.copy(synthetic_video, video)
    info = ingest.probe(video)
    detect.load_or_run(video, make_config(), info)

    retuned = make_config(control={"omega_n": 1.6, "deadzone": 0.01})
    _, reused = detect.load_or_run(video, retuned, info)
    assert reused


def test_changing_detection_constants_invalidates_the_cache(tmp_path, synthetic_video):
    import shutil

    video = tmp_path / "clip.mp4"
    shutil.copy(synthetic_video, video)
    info = ingest.probe(video)
    detect.load_or_run(video, make_config(), info)

    cache = detect.load_cache(video)
    ok, why = detect.cache_is_valid(cache, make_config(detect={"conf_min": 0.6}), info)
    assert not ok
    assert "detection constants" in why


def test_resolution_change_invalidates_the_cache(tmp_path, synthetic_video):
    import shutil

    video = tmp_path / "clip.mp4"
    shutil.copy(synthetic_video, video)
    info = ingest.probe(video)
    detect.load_or_run(video, make_config(), info)

    other = ingest.SourceInfo("clip.mp4", 3840, 2160, info.fps, info.frames, 12.0, 0, False, "test")
    ok, why = detect.cache_is_valid(detect.load_cache(video), make_config(), other)
    assert not ok and "resolution" in why


def test_missing_cache_explains_itself(tmp_path):
    with pytest.raises(detect.DetectionError, match="autocam detect"):
        detect.load_cache(tmp_path / "nope.mp4")


def test_bad_cache_schema_is_refused(tmp_path):
    video = tmp_path / "clip.mp4"
    detect.detections_path_for(video).write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(detect.DetectionError, match="schema_version"):
        detect.load_cache(video)


def test_cache_stores_raw_boxes_before_pitch_filtering(synthetic_video):
    """Raw means raw: the spectator and the foreground blob must still be in there."""
    cfg = make_config()
    info = ingest.probe(synthetic_video)
    cache = detect.run_detection(synthetic_video, cfg, info)
    boxes = np.asarray(cache.timesteps[0]["boxes"], dtype=float)
    feet_y = boxes[:, 3]
    assert (feet_y < 140).any()          # the touchline spectator, above the pitch quad
