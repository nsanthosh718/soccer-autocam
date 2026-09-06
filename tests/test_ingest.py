import numpy as np
import pytest

from autocam import ingest
from autocam.config import load as load_config
from conftest import FPS, HEIGHT, WIDTH, make_config


def test_probe_reads_the_synthetic_source(synthetic_video):
    info = ingest.probe(synthetic_video)
    assert (info.width, info.height) == (WIDTH, HEIGHT)
    assert info.fps == pytest.approx(FPS)
    assert info.frames == 360
    assert info.rotation == 0
    assert info.duration_s == pytest.approx(12.0, abs=0.05)


def test_probe_rejects_a_missing_file(tmp_path):
    with pytest.raises(ingest.IngestError, match="no such file"):
        ingest.probe(tmp_path / "absent.mp4")


def test_sub_4k_sources_are_rejected(synthetic_video):
    info = ingest.probe(synthetic_video)
    with pytest.raises(ingest.IngestError, match="minimum is 3840x2160"):
        ingest.check_resolution(info, load_config())
    ingest.check_resolution(info, make_config())     # lowered on purpose in tests


@pytest.mark.parametrize("rotation,expected", [(0, (4, 6)), (90, (6, 4)), (180, (4, 6)), (270, (6, 4))])
def test_apply_rotation_shapes(rotation, expected):
    img = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    out = ingest.apply_rotation(img, rotation)
    assert out.shape[:2] == expected


def test_apply_rotation_is_clockwise():
    """A 90-degree display matrix must move the top-left pixel to the top-right."""
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[0, 0] = [255, 0, 0]
    out = ingest.apply_rotation(img, 90)
    assert out[0, -1].tolist() == [255, 0, 0]


def test_rotation_roundtrips_to_identity():
    img = np.random.default_rng(0).integers(0, 255, (5, 9, 3), dtype=np.uint8)
    once = ingest.apply_rotation(img, 90)
    assert np.array_equal(ingest.apply_rotation(once, 270), img)


def test_unsupported_rotation_is_refused():
    with pytest.raises(ingest.IngestError):
        ingest._normalise_rotation(45)
    assert ingest._normalise_rotation(-90) == 270
    assert ingest._normalise_rotation(None) == 0


def test_iter_frames_indices_are_source_absolute(synthetic_video):
    info = ingest.probe(synthetic_video)
    refs = list(ingest.iter_frames(synthetic_video, info, start_s=2.0, end_s=3.0))
    assert len(refs) == pytest.approx(30, abs=1)
    assert refs[0].n == 60
    assert refs[0].t == pytest.approx(2.0, abs=1e-6)
    assert all(b.n == a.n + 1 for a, b in zip(refs, refs[1:]))


def test_iter_frames_whole_file(synthetic_video):
    info = ingest.probe(synthetic_video)
    assert sum(1 for _ in ingest.iter_frames(synthetic_video, info)) == 360


def test_frame_at_decodes_display_oriented_pixels(synthetic_video):
    img = ingest.frame_at(synthetic_video, 0)
    assert img.shape == (HEIGHT, WIDTH, 3)
    assert img.dtype == np.uint8


def test_to_rgb_scaled_reports_usable_scale_factors(synthetic_video):
    info = ingest.probe(synthetic_video)
    ref = next(iter(ingest.iter_frames(synthetic_video, info)))
    small, sx, sy = ingest.to_rgb_scaled(ref.frame, info.rotation, 320)
    assert max(small.shape[:2]) <= 320
    assert small.shape[1] * sx == pytest.approx(WIDTH, abs=1.0)
    assert small.shape[0] * sy == pytest.approx(HEIGHT, abs=1.0)


def test_to_rgb_scaled_is_a_noop_when_already_small(synthetic_video):
    info = ingest.probe(synthetic_video)
    ref = next(iter(ingest.iter_frames(synthetic_video, info)))
    img, sx, sy = ingest.to_rgb_scaled(ref.frame, info.rotation, 4096)
    assert (sx, sy) == (1.0, 1.0)
    assert img.shape == (HEIGHT, WIDTH, 3)


def test_frame_count_falls_back_to_duration():
    info = ingest.SourceInfo("x", 3840, 2160, 30.0, 0, 10.0, 0, False, "test")
    assert ingest.frame_count(info) == 300
