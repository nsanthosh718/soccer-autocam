import numpy as np
import pytest

from autocam import calibrate as cal
from autocam import centroid
from autocam.config import load as load_config
from autocam.detect import DetectionCache
from autocam.ingest import SourceInfo

INFO = SourceInfo("m.mp4", 3840, 2160, 30.0, 300, 10.0, 0, False, "test")
PITCH = cal.make_pitch([[200, 1800], [3600, 1800], [3500, 900], [300, 900]], 3840, 2160)


def box(cx, foot_y, w=40, h=110, conf=0.9):
    return [cx - w / 2, foot_y - h, cx + w / 2, foot_y, conf]


def test_foot_points_are_bottom_centre():
    pts = centroid.foot_points(np.array([box(1000, 1500)]))
    assert pts[0].tolist() == [1000.0, 1500.0]


def test_filter_drops_low_confidence():
    cfg = load_config()
    boxes = np.array([box(1000, 1500, conf=0.9), box(1100, 1500, conf=0.1)])
    kept = centroid.filter_boxes(boxes, cfg, PITCH, 3840 * 2160)
    assert len(kept) == 1


def test_filter_drops_out_of_area_boxes():
    cfg = load_config()
    frame_area = 3840 * 2160
    too_big = box(1000, 1500, w=1200, h=1900)      # spectator across the lens
    too_small = box(1200, 1500, w=2, h=4)          # far-field noise
    ok = box(1400, 1500)
    kept = centroid.filter_boxes(np.array([too_big, too_small, ok]), cfg, PITCH, frame_area)
    assert len(kept) == 1
    assert kept[0][0] == pytest.approx(1400 - 20)


def test_pitch_filter_drops_touchline_detections():
    """The single most important filter: feet off the pitch, detection gone."""
    cfg = load_config()
    on_pitch = box(1000, 1500)
    spectator = box(1000, 700)       # above the quad's top edge
    kept = centroid.filter_boxes(np.array([on_pitch, spectator]), cfg, PITCH, 3840 * 2160)
    assert len(kept) == 1
    assert centroid.foot_points(kept)[0][1] == 1500


def test_median_ignores_the_stranded_goalkeeper():
    cfg = load_config()
    xs = np.array([1000, 1010, 1020, 1030, 1040, 1050, 3600.0])
    cx, _, n = centroid.robust_centroid(xs, cfg)
    assert cx == pytest.approx(1025.0)
    assert n == 6                      # the keeper was rejected by MAD
    assert abs(cx - float(np.mean(xs))) > 300


def test_mad_rejection_keeps_a_tight_cluster_intact():
    cfg = load_config()
    xs = np.array([1000.0, 1001.0, 1002.0, 1003.0])
    cx, _, n = centroid.robust_centroid(xs, cfg)
    assert n == 4
    assert cx == pytest.approx(1001.5)


def test_identical_positions_do_not_zero_out_the_cluster():
    """A zero MAD must not reject every point."""
    cfg = load_config()
    xs = np.full(8, 1500.0)
    cx, spread, n = centroid.robust_centroid(xs, cfg)
    assert (cx, spread, n) == (1500.0, 0.0, 8)


def test_spread_is_the_interquartile_range():
    cfg = load_config()
    xs = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    _, spread, _ = centroid.robust_centroid(xs, cfg)
    assert spread == pytest.approx(200.0)


def test_empty_detection_set():
    cfg = load_config()
    cx, spread, n = centroid.robust_centroid(np.array([]), cfg)
    assert n == 0 and np.isnan(cx) and np.isnan(spread)


def _cache(timesteps):
    return DetectionCache(source=INFO.to_dict(), detector={}, fingerprint="", step=3,
                          timesteps=timesteps)


def test_timestep_below_min_players_is_invalid():
    cfg = load_config()
    ts = _cache([
        {"n": 0, "t": 0.0, "boxes": [box(1000 + 30 * i, 1500) for i in range(8)]},
        {"n": 3, "t": 0.1, "boxes": [box(1000, 1500), box(1030, 1500)]},
    ])
    out = centroid.analyse_timesteps(ts, cfg, PITCH, INFO)
    assert [t.valid for t in out] == [True, False]
    assert out[0].n_players >= cfg.centroid.min_players


def test_build_track_holds_through_an_invalid_run():
    """Invalid timesteps hold the last good target; they never emit a jump."""
    cfg = load_config()
    good = centroid.Timestep(n=0, t=0.0, cx=1000.0, spread=50.0, n_players=10, n_raw=10, valid=True)
    bad = [centroid.Timestep(n=3 * i, t=0.1 * i, cx=3000.0, spread=10.0,
                             n_players=1, n_raw=1, valid=False) for i in range(1, 5)]
    recovered = centroid.Timestep(n=15, t=0.5, cx=1200.0, spread=60.0,
                                  n_players=12, n_raw=12, valid=True)
    track = centroid.build_track([good] + bad + [recovered], cfg, INFO, frames=16)
    assert track.cx[0] == pytest.approx(1000.0)
    assert track.cx[12] == pytest.approx(1000.0)   # held, not dragged to 3000
    assert track.cx[15] == pytest.approx(1200.0)
    assert not track.valid[6]
    assert track.valid[0]


def test_build_track_interpolates_between_timesteps():
    cfg = load_config()
    steps = [
        centroid.Timestep(n=0, t=0.0, cx=1000.0, spread=100.0, n_players=10, n_raw=10, valid=True),
        centroid.Timestep(n=3, t=0.1, cx=1300.0, spread=160.0, n_players=10, n_raw=10, valid=True),
    ]
    track = centroid.build_track(steps, cfg, INFO, frames=4)
    assert track.cx.tolist() == pytest.approx([1000.0, 1100.0, 1200.0, 1300.0])
    assert track.spread[1] == pytest.approx(120.0)


def test_build_track_with_no_timesteps_centres_the_frame():
    cfg = load_config()
    track = centroid.build_track([], cfg, INFO, frames=5)
    assert (track.cx == INFO.width / 2).all()
    assert not track.valid.any()


def test_quality_report_flags_a_long_hold():
    cfg = load_config()
    track = centroid.Track(
        fps=30.0, frames=300,
        cx=np.full(300, 1920.0), spread=np.zeros(300),
        n_players=np.full(300, 2), valid=np.array([True] * 100 + [False] * 200),
        timesteps=[centroid.Timestep(n=0, t=0.0, cx=1920.0, spread=0.0,
                                     n_players=2, n_raw=2, valid=False)],
    )
    report = centroid.quality_report(track, cfg)
    assert report["invalid_frame_pct"] == pytest.approx(200 / 3, abs=0.01)
    assert report["longest_invalid_run_s"] == pytest.approx(200 / 30, abs=0.01)
    assert any("held, not tracked" in w for w in report["warnings"])
    assert any("mean" in w for w in report["warnings"])
