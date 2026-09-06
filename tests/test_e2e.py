"""End-to-end: a synthetic match through the real CLI, start to finish."""

import json
import shutil

import numpy as np
import pytest
from click.testing import CliRunner

from autocam import calibrate as cal
from autocam import detect, ingest, telemetry
from autocam.cli import main
from conftest import FPS, PITCH_QUAD, WIDTH, true_centre

CORNERS = ",".join(str(v) for pair in PITCH_QUAD for v in pair)


@pytest.fixture
def clip(tmp_path, synthetic_video):
    path = tmp_path / "match.mp4"
    shutil.copy(synthetic_video, path)
    return path


def output_of(result) -> str:
    """Click 8.2+ keeps stderr separate; operator-facing errors land there."""
    return result.output + (result.stderr if result.stderr_bytes else "")


def invoke(runner, *args):
    result = runner.invoke(main, [str(a) for a in args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture
def rendered(clip, config_file):
    runner = CliRunner()
    invoke(runner, "calibrate", clip, "--corners", CORNERS, "--config", config_file)
    invoke(runner, "detect", clip, "--config", config_file)
    result = invoke(runner, "render", clip, "--config", config_file)
    return clip, result


def test_pipeline_produces_every_artefact(rendered):
    clip, _ = rendered
    assert cal.pitch_path_for(clip).exists()
    assert detect.detections_path_for(clip).exists()
    assert telemetry.telemetry_path_for(clip).exists()
    assert clip.with_name("match.autocam.mp4").exists()


def test_output_video_is_the_configured_crop(rendered):
    clip, _ = rendered
    source = ingest.probe(clip)
    out = ingest.probe(clip.with_name("match.autocam.mp4"))
    assert (out.width, out.height) == (320, 180)
    assert out.fps == pytest.approx(source.fps)
    assert out.frames == pytest.approx(source.frames, abs=2)


def test_telemetry_matches_the_schema_and_the_source(rendered):
    clip, _ = rendered
    doc = telemetry.load(clip)
    info = ingest.probe(clip)

    assert doc["schema_version"] == telemetry.TELEMETRY_SCHEMA_VERSION
    assert doc["source"]["width"] == info.width
    assert doc["source"]["fps"] == pytest.approx(info.fps)
    assert len(doc["frames"]) == info.frames
    assert len(doc["pitch_quad"]) == 4
    assert doc["quality"]["mean_players_detected"] > 10

    for frame in doc["frames"][::37]:
        x, y, w, h = frame["crop"]
        assert 0 <= x and x + w <= info.width
        assert 0 <= y and y + h <= info.height


def test_framing_keeps_the_play_in_shot(rendered):
    """The automated stand-in for exit criterion 1, on footage whose ground truth
    we actually know. Real matches still need the manual 100-frame count."""
    clip, _ = rendered
    doc = telemetry.load(clip)
    inside = 0
    for frame in doc["frames"]:
        x, _, w, _ = frame["crop"]
        if x <= true_centre(frame["t"]) <= x + w:
            inside += 1
    assert inside / len(doc["frames"]) >= 0.92


def test_centroid_follows_the_scripted_play(rendered):
    clip, _ = rendered
    doc = telemetry.load(clip)
    errors = [abs(f["cx"] - true_centre(f["t"])) for f in doc["frames"]]
    assert float(np.median(errors)) < 15.0


def test_pan_trace_is_smooth(rendered):
    """No frame-to-frame jump larger than the configured velocity clamp allows."""
    clip, _ = rendered
    doc = telemetry.load(clip)
    px = np.array([f["px"] for f in doc["frames"]])
    max_step = 0.18 * WIDTH / FPS + 1e-6
    assert np.abs(np.diff(px)).max() <= max_step


def test_static_play_does_not_wander(rendered):
    """Frames 0-90 are the parked-in-midfield stretch: the crop must not move."""
    clip, _ = rendered
    doc = telemetry.load(clip)
    px = np.array([f["px"] for f in doc["frames"][30:90]])
    assert np.ptp(px) < 1.0


def test_transition_is_reported(rendered):
    clip, _ = rendered
    doc = telemetry.load(clip)
    kinds = {e["type"] for e in doc["events"]}
    assert "transition" in kinds
    transition = next(e for e in doc["events"] if e["type"] == "transition")
    assert transition["direction"] == "right"
    assert 2.0 <= transition["t"] <= 6.0
    assert transition["heuristic"] is True


def test_rerender_reuses_the_detection_cache(clip, config_file):
    """The tuning loop: change a control constant, re-render, no inference."""
    runner = CliRunner()
    invoke(runner, "calibrate", clip, "--corners", CORNERS, "--config", config_file)
    invoke(runner, "detect", clip, "--config", config_file)
    mtime = detect.detections_path_for(clip).stat().st_mtime_ns

    result = invoke(runner, "render", clip, "--config", config_file,
                    "--preview-only", "--set", "control.omega_n=1.5")
    assert detect.detections_path_for(clip).stat().st_mtime_ns == mtime
    assert clip.with_name("match.preview.mp4").exists()
    assert "run report" in output_of(result)


def test_preview_is_short_and_small(clip, config_file):
    runner = CliRunner()
    invoke(runner, "calibrate", clip, "--corners", CORNERS, "--config", config_file)
    invoke(runner, "detect", clip, "--config", config_file)
    invoke(runner, "render", clip, "--config", config_file, "--preview-only")
    out = ingest.probe(clip.with_name("match.preview.mp4"))
    assert (out.width, out.height) == (160, 90)
    assert out.frames == pytest.approx(60, abs=2)      # preview_duration_s = 2.0


def test_render_without_calibration_refuses_and_says_why(clip, config_file):
    runner = CliRunner()
    invoke(runner, "detect", clip, "--config", config_file)
    result = runner.invoke(main, ["render", str(clip), "--config", str(config_file)])
    assert result.exit_code != 0
    assert "autocam calibrate" in output_of(result)


def test_render_without_detections_refuses_and_says_why(clip, config_file):
    runner = CliRunner()
    invoke(runner, "calibrate", clip, "--corners", CORNERS, "--config", config_file)
    result = runner.invoke(main, ["render", str(clip), "--config", str(config_file)])
    assert result.exit_code != 0
    assert "autocam detect" in output_of(result)


def test_sub_4k_source_is_refused_by_default(clip):
    runner = CliRunner()
    result = runner.invoke(main, ["detect", str(clip)])
    assert result.exit_code != 0
    assert "3840x2160" in output_of(result)


def test_run_command_does_the_whole_thing(clip, config_file):
    runner = CliRunner()
    result = invoke(runner, "run", clip, "--config", config_file, "--full-frame",
                    "--preview-only")
    assert cal.load(clip).note                       # full-frame calibration
    assert clip.with_name("match.preview.mp4").exists()
    assert "FILTER DISABLED" in output_of(result)


def test_calibrate_refuses_to_clobber_without_force(clip, config_file):
    runner = CliRunner()
    invoke(runner, "calibrate", clip, "--corners", CORNERS, "--config", config_file)
    quad = cal.load(clip).quad.copy()
    invoke(runner, "calibrate", clip, "--full-frame", "--config", config_file)
    assert np.allclose(cal.load(clip).quad, quad)
    invoke(runner, "calibrate", clip, "--full-frame", "--config", config_file, "--force")
    assert not np.allclose(cal.load(clip).quad, quad)


def test_config_command_writes_the_phase_2_input(tmp_path, config_file):
    runner = CliRunner()
    out = tmp_path / "tuned.json"
    invoke(runner, "config", "--config", config_file, "--set", "control.omega_n=1.2",
           "--out", out)
    written = json.loads(out.read_text())
    assert written["control"]["omega_n"] == 1.2
    assert written["phase2"]["deadzone_scale"] == 2.0     # both phases, one file


def test_bad_override_is_rejected_loudly(clip, config_file):
    runner = CliRunner()
    result = runner.invoke(main, ["detect", str(clip), "--config", str(config_file),
                                  "--set", "control.zeta=0"])
    assert result.exit_code != 0
    assert "config error" in output_of(result)
