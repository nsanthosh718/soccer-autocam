import numpy as np
import pytest

from autocam import calibrate as cal

RECT = [[100.0, 100.0], [900.0, 100.0], [900.0, 500.0], [100.0, 500.0]]


def test_order_quad_is_click_order_independent():
    expected = cal.order_quad(np.array(RECT))
    for roll in range(4):
        rolled = np.roll(np.array(RECT), roll, axis=0)
        assert np.allclose(cal.order_quad(rolled), expected)
    shuffled = np.array(RECT)[[2, 0, 3, 1]]
    assert np.allclose(cal.order_quad(shuffled), expected)


def test_order_quad_starts_top_left():
    quad = cal.order_quad(np.array(RECT))
    assert np.allclose(quad[0], [100.0, 100.0])
    assert np.allclose(quad[2], [900.0, 500.0])


def test_points_in_polygon():
    quad = np.array(RECT)
    xs = np.array([500.0, 50.0, 500.0, 950.0])
    ys = np.array([300.0, 300.0, 50.0, 300.0])
    assert cal.points_in_polygon(xs, ys, quad).tolist() == [True, False, False, False]


def test_points_in_trapezoid_respects_slanted_edge():
    quad = cal.order_quad(np.array([[0, 400], [1000, 400], [800, 100], [200, 100]]))
    # (150, 150) is outside the slanted left edge but inside the bounding box.
    inside = cal.points_in_polygon(np.array([500.0, 150.0]), np.array([150.0, 150.0]), quad)
    assert inside.tolist() == [True, False]


def test_homography_roundtrip():
    pitch = cal.make_pitch([[100, 900], [1800, 880], [1700, 400], [200, 420]], 1920, 1080)
    unit = cal.apply_homography(pitch.image_to_unit, pitch.quad)
    assert np.allclose(unit, [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-9)
    back = cal.apply_homography(pitch.unit_to_image, unit)
    assert np.allclose(back, pitch.quad, atol=1e-6)


def test_third_lines_on_a_rectangle_are_exact_thirds():
    pitch = cal.make_pitch(RECT, 1000, 600)
    lo, hi = pitch.third_lines_x()
    assert lo == pytest.approx(100 + 800 / 3, abs=1e-6)
    assert hi == pytest.approx(100 + 1600 / 3, abs=1e-6)
    assert lo < hi


def test_centre_y_is_quad_mean():
    pitch = cal.make_pitch(RECT, 1000, 600)
    assert pitch.centre_y() == pytest.approx(300.0)


def test_degenerate_quad_is_rejected():
    with pytest.raises(cal.CalibrationError):
        cal.make_pitch([[0, 0], [0, 0], [0, 0], [0, 0]], 1920, 1080)


def test_tiny_quad_is_rejected_as_a_misclick():
    with pytest.raises(cal.CalibrationError, match="mis-click"):
        cal.make_pitch([[10, 10], [40, 10], [40, 40], [10, 40]], 1920, 1080)


def test_full_frame_pitch_contains_everything():
    pitch = cal.full_frame_pitch(640, 360)
    assert pitch.contains(np.array([1.0, 320.0, 638.0]), np.array([1.0, 180.0, 358.0])).all()
    assert pitch.note


def test_parse_corners():
    quad = cal.parse_corners("1,2, 3,4; 5,6 7,8")
    assert quad.shape == (4, 2)
    assert quad[3].tolist() == [7.0, 8.0]


@pytest.mark.parametrize("text", ["1,2,3", "a,b,c,d,e,f,g,h", ""])
def test_parse_corners_rejects_garbage(text):
    with pytest.raises(cal.CalibrationError):
        cal.parse_corners(text)


def test_save_load_roundtrip(tmp_path):
    video = tmp_path / "match.mp4"
    video.write_bytes(b"")
    pitch = cal.make_pitch(RECT, 1000, 600)
    path = cal.save(pitch, video)
    assert path.name == "match.mp4.pitch.json"
    loaded = cal.load(video)
    assert np.allclose(loaded.quad, pitch.quad)
    assert (loaded.source_width, loaded.source_height) == (1000, 600)


def test_load_missing_calibration_explains_itself(tmp_path):
    with pytest.raises(cal.CalibrationError, match="autocam calibrate"):
        cal.load(tmp_path / "nope.mp4")


def test_source_mismatch_is_caught(tmp_path):
    pitch = cal.make_pitch(RECT, 1000, 600)
    with pytest.raises(cal.CalibrationError, match="Re-run calibrate"):
        cal.check_matches_source(pitch, 3840, 2160)
