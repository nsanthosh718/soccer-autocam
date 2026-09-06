import numpy as np
import pytest

from autocam import control
from autocam.config import load as load_config

W, CROP, FPS = 3840, 1920, 30.0

# The crop centre can only travel within [960, 2880] on a 3840-wide source.
# Step tests deliberately stay clear of those walls so they exercise the control
# law rather than the edge clamp.
START, GOAL = 1440.0, 2400.0            # a quarter-frame-width step


def step_input(hold_frames, jump_to, total, start=START):
    return np.concatenate([np.full(hold_frames, start), np.full(total - hold_frames, jump_to)])


def test_static_play_produces_no_pan():
    """Deadzone: jitter around a fixed centroid must not move the crop at all."""
    cfg = load_config()
    rng = np.random.default_rng(7)
    cx = 1920.0 + rng.normal(0.0, 12.0, 600)
    trace = control.run(cx, cfg, W, CROP, FPS, initial_px=1920.0)
    assert np.ptp(trace.px) == pytest.approx(0.0, abs=1e-9)


def test_no_overshoot_on_a_step():
    """zeta = 1.0 is chosen precisely so this holds."""
    cfg = load_config()
    trace = control.run(step_input(30, GOAL, 400), cfg, W, CROP, FPS, initial_px=START)
    assert trace.px.max() <= GOAL + 1e-6
    assert trace.px.max() < W - CROP / 2      # not the edge clamp doing the work


def test_pan_is_monotone_toward_a_step():
    cfg = load_config()
    trace = control.run(step_input(30, GOAL, 400), cfg, W, CROP, FPS, initial_px=START)
    assert (np.diff(trace.px) >= -1e-9).all()


def test_transition_is_tracked_within_the_exit_criterion():
    """Exit criterion 3: reach the new play location within 1.5 s, no overshoot."""
    cfg = load_config()
    tol = cfg.control.deadzone * W      # the controller parks inside the deadzone
    trace = control.run(step_input(30, GOAL, 400), cfg, W, CROP, FPS, initial_px=START)
    settle = control.settling_time(trace.px[30:], GOAL, FPS, tol)
    assert settle is not None
    assert settle <= 1.5


def test_velocity_clamp_is_respected():
    cfg = load_config()
    trace = control.run(step_input(10, 2800.0, 300), cfg, W, CROP, FPS, initial_px=1000.0)
    assert np.abs(trace.vx).max() <= cfg.control.max_pan_vel * W + 1e-6


def test_acceleration_clamp_is_respected():
    cfg = load_config()
    trace = control.run(step_input(10, GOAL, 300), cfg, W, CROP, FPS, initial_px=START)
    accel = np.abs(np.diff(trace.vx)) * FPS
    assert accel.max() <= cfg.control.max_pan_accel * W + 1e-6


def test_crop_never_leaves_the_source_frame():
    cfg = load_config()
    cx = np.concatenate([np.full(150, -500.0), np.full(150, 5000.0)])
    trace = control.run(cx, cfg, W, CROP, FPS, initial_px=1920.0)
    assert trace.px.min() >= CROP / 2 - 1e-9
    assert trace.px.max() <= W - CROP / 2 + 1e-9


def test_lead_offsets_the_target_in_the_direction_of_travel():
    cfg = load_config()
    velocity = 400.0                       # px/s, sustained
    n = 300
    cx = 1920.0 + velocity * np.arange(n) / FPS
    trace = control.run(cx, cfg, W, CROP, FPS, initial_px=1920.0)
    lead = trace.target[-1] - cx[-1]
    assert lead == pytest.approx(cfg.control.lead_gain * velocity, rel=0.02)
    assert lead > 0


def test_lead_is_capped():
    cfg = load_config()
    velocity = 4000.0                      # absurdly fast; lead must saturate
    n = 200
    cx = 500.0 + velocity * np.arange(n) / FPS
    trace = control.run(cx, cfg, W, CROP, FPS, initial_px=500.0)
    leads = trace.target - cx
    assert leads.max() <= cfg.control.lead_max * W + 1e-6


def test_lead_reverses_with_the_play():
    cfg = load_config()
    n = 300
    cx = 2500.0 - 400.0 * np.arange(n) / FPS
    trace = control.run(cx, cfg, W, CROP, FPS, initial_px=2500.0)
    assert trace.target[-1] - cx[-1] < 0


def test_underdamped_would_overshoot():
    """Sanity check that zeta is actually load-bearing, not decorative."""
    cfg = load_config(overrides={"control": {"zeta": 0.3, "max_pan_vel": 5.0,
                                             "max_pan_accel": 50.0, "deadzone": 0.0,
                                             "lead_gain": 0.0}})
    trace = control.run(step_input(10, GOAL, 300), cfg, W, CROP, FPS, initial_px=START)
    assert trace.px.max() > GOAL


def test_crop_wider_than_source_is_rejected():
    cfg = load_config()
    with pytest.raises(control.ControlError):
        control.PanController(cfg, frame_width=1920, crop_width=3840)


def test_empty_track_returns_empty_trace():
    cfg = load_config()
    trace = control.run(np.array([]), cfg, W, CROP, FPS)
    assert len(trace.px) == 0


def test_no_sustained_oscillation_after_a_transition():
    """Exit criterion 2: inspect the settled px trace for periodic components."""
    cfg = load_config()
    trace = control.run(step_input(30, GOAL, 900), cfg, W, CROP, FPS, initial_px=START)
    settled = trace.px[300:]
    reversals = np.sum(np.diff(np.sign(np.diff(settled))) != 0)
    assert reversals == 0
    assert np.ptp(settled) < 1e-6
