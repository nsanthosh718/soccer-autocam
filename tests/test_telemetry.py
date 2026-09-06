import numpy as np
import pytest

from autocam import calibrate as cal
from autocam import centroid, control, render, telemetry
from autocam.config import load as load_config
from autocam.ingest import SourceInfo

FPS = 30.0
INFO = SourceInfo("m.mp4", 3840, 2160, FPS, 900, 30.0, 0, False, "test")
PITCH = cal.make_pitch([[200, 1800], [3600, 1800], [3500, 900], [300, 900]], 3840, 2160)
ALL_VALID = np.ones(900, dtype=bool)


def test_attacking_third_entry_fires_on_a_crossing():
    cfg = load_config()
    lo, hi = PITCH.third_lines_x()
    px = np.concatenate([np.full(300, (lo + hi) / 2), np.full(600, hi + 200)])
    events = telemetry.detect_attacking_third_entries(px, ALL_VALID, cfg, PITCH, FPS)
    assert len(events) == 1
    assert events[0]["direction"] == "right"
    assert events[0]["t"] == pytest.approx(300 / FPS, abs=1 / FPS)
    assert events[0]["heuristic"] is True


def test_attacking_third_entry_is_debounced():
    """Play rattling across the line must not emit an event per crossing."""
    cfg = load_config()
    lo, hi = PITCH.third_lines_x()
    mid, deep = (lo + hi) / 2, hi + 200
    # Cross into the right third four times inside the 5 s debounce window.
    px = np.concatenate([np.tile(np.concatenate([np.full(15, mid), np.full(15, deep)]), 5),
                         np.full(750, mid)])
    events = telemetry.detect_attacking_third_entries(px, ALL_VALID, cfg, PITCH, FPS)
    assert len(events) == 1


def test_attacking_third_entry_distinguishes_direction():
    cfg = load_config()
    lo, hi = PITCH.third_lines_x()
    px = np.concatenate([np.full(300, (lo + hi) / 2), np.full(600, lo - 200)])
    events = telemetry.detect_attacking_third_entries(px, ALL_VALID, cfg, PITCH, FPS)
    assert [e["direction"] for e in events] == ["left"]


def test_no_pitch_means_no_third_line_events():
    cfg = load_config()
    assert telemetry.detect_attacking_third_entries(np.zeros(10), ALL_VALID[:10], cfg, None, FPS) == []


def test_transition_needs_sustained_velocity():
    cfg = load_config()
    fast = cfg.events.transition_vel_threshold * INFO.width * 1.5
    brief = np.zeros(900)
    brief[100:115] = fast                      # 0.5 s: too short
    assert telemetry.detect_transitions(brief, ALL_VALID, cfg, INFO.width, FPS) == []

    sustained = np.zeros(900)
    sustained[100:160] = fast                  # 2.0 s
    events = telemetry.detect_transitions(sustained, ALL_VALID, cfg, INFO.width, FPS)
    assert len(events) == 1
    assert events[0]["type"] == "transition"
    assert events[0]["direction"] == "right"
    assert events[0]["duration_s"] == pytest.approx(2.0, abs=0.05)


def test_transition_below_threshold_is_ignored():
    cfg = load_config()
    slow = np.full(900, cfg.events.transition_vel_threshold * INFO.width * 0.5)
    assert telemetry.detect_transitions(slow, ALL_VALID, cfg, INFO.width, FPS) == []


def test_transition_direction_follows_the_sign():
    cfg = load_config()
    vel = np.zeros(900)
    vel[100:200] = -cfg.events.transition_vel_threshold * INFO.width * 2
    events = telemetry.detect_transitions(vel, ALL_VALID, cfg, INFO.width, FPS)
    assert events[0]["direction"] == "left"


def test_confidence_reflects_local_detection_quality():
    cfg = load_config()
    valid = np.ones(900, dtype=bool)
    valid[280:320] = False
    lo, hi = PITCH.third_lines_x()
    px = np.concatenate([np.full(300, (lo + hi) / 2), np.full(600, hi + 200)])
    events = telemetry.detect_attacking_third_entries(px, valid, cfg, PITCH, FPS)
    assert events[0]["confidence"] < 0.5


def _document():
    cfg = load_config()
    steps = [centroid.Timestep(n=i * 3, t=i * 0.1, cx=1900.0 + i, spread=400.0,
                               n_players=12, n_raw=14, valid=True) for i in range(20)]
    track = centroid.build_track(steps, cfg, INFO, frames=60)
    trace = control.run(track.cx, cfg, INFO.width, cfg.render.out_width, FPS)
    crops = render.crop_rects(trace.px, cfg, INFO, PITCH)
    quality = centroid.quality_report(track, cfg)
    return cfg, telemetry.build("m.mp4", INFO, cfg, PITCH, track, trace, crops, quality)


def test_document_carries_every_field_the_schema_promises():
    cfg, doc = _document()
    assert doc["schema_version"] == telemetry.TELEMETRY_SCHEMA_VERSION
    assert set(doc) >= {"source", "config_hash", "pitch_quad", "frames", "events", "quality"}
    assert doc["config_hash"] == cfg.hash()
    assert doc["source"]["width"] == 3840
    assert len(doc["pitch_quad"]) == 4

    frame = doc["frames"][0]
    assert set(frame) == {"n", "t", "cx", "spread", "px", "crop", "n_players", "valid"}
    assert len(frame["crop"]) == 4
    assert isinstance(frame["valid"], bool)

    assert set(doc["quality"]) >= {"invalid_frame_pct", "longest_invalid_run_s",
                                   "mean_players_detected"}


def test_events_are_labelled_as_heuristics():
    """The spec is explicit: these are candidates for review, not analysis."""
    _, doc = _document()
    assert "manual review" in doc["events_note"]
    assert all(e["heuristic"] for e in doc["events"])


def test_save_and_load_roundtrip(tmp_path):
    _, doc = _document()
    video = tmp_path / "m.mp4"
    path = telemetry.save(doc, video)
    assert path.name == "m.mp4.telemetry.json"
    assert telemetry.load(video)["config_hash"] == doc["config_hash"]
