import numpy as np
import pytest

from autocam import calibrate as cal
from autocam import ingest, render
from autocam.config import load as load_config
from conftest import HEIGHT, PITCH_QUAD, WIDTH, make_config

INFO_4K = ingest.SourceInfo("m.mp4", 3840, 2160, 30.0, 900, 30.0, 0, False, "test")
PITCH_4K = cal.make_pitch([[200, 1800], [3600, 1800], [3500, 900], [300, 900]], 3840, 2160)


def test_crop_is_clamped_inside_the_frame():
    cfg = load_config()
    rects = render.crop_rects(np.array([-1000.0, 1920.0, 99999.0]), cfg, INFO_4K, PITCH_4K)
    assert rects[0][0] == 0
    assert rects[2][0] == 3840 - 1920
    assert (rects[:, 0] + rects[:, 2] <= 3840).all()


def test_crop_x_is_even():
    """An odd left edge shifts chroma by half a sample and shimmers."""
    cfg = load_config()
    rects = render.crop_rects(np.array([1921.0, 1922.5, 1923.0]), cfg, INFO_4K, PITCH_4K)
    assert (rects[:, 0] % 2 == 0).all()


def test_crop_y_follows_the_pitch_not_the_frame_centre():
    cfg = load_config()
    pitch_y = render.crop_y_for(cfg, INFO_4K, PITCH_4K)
    centre_y = render.crop_y_for(load_config(overrides={"render": {"crop_y_mode": "center"}}),
                                 INFO_4K, PITCH_4K)
    assert pitch_y == int(round(PITCH_4K.centre_y() - 1080 / 2))
    assert centre_y == (2160 - 1080) // 2
    assert pitch_y != centre_y


def test_crop_y_is_clamped_to_the_frame():
    cfg = load_config(overrides={"render": {"crop_y_mode": "fixed", "crop_y_frac": 0.99}})
    y = render.crop_y_for(cfg, INFO_4K, PITCH_4K)
    assert 0 <= y <= 2160 - 1080


def test_crop_wider_than_the_source_is_refused():
    cfg = load_config()
    small = ingest.SourceInfo("m.mp4", 1280, 720, 30.0, 90, 3.0, 0, False, "test")
    with pytest.raises(render.RenderError, match="exceeds source width"):
        render.crop_rects(np.array([640.0]), cfg, small, None)


def test_preview_plan_downscales_and_truncates():
    cfg = load_config()
    plan = render.make_plan(cfg, INFO_4K, PITCH_4K, start_s=10.0, end_s=None, preview=True)
    assert (plan.out_width, plan.out_height) == (1280, 720)
    assert plan.crop_width == 1920                # crop at full size, then scale
    assert plan.end_s == pytest.approx(70.0)
    assert plan.scaled is True
    assert plan.copy_audio is False               # previews skip audio for speed


def test_full_plan_is_pan_only_with_no_scaling():
    cfg = load_config()
    plan = render.make_plan(cfg, INFO_4K, PITCH_4K, None, None, preview=False)
    assert (plan.out_width, plan.out_height) == (1920, 1080)
    assert plan.scaled is False
    assert plan.copy_audio is True


def test_default_output_paths(tmp_path):
    video = tmp_path / "match.mp4"
    assert render.default_output_path(video, False).name == "match.autocam.mp4"
    assert render.default_output_path(video, True).name == "match.preview.mp4"


def test_render_writes_a_decodable_file(tmp_path, synthetic_video):
    cfg = make_config()
    info = ingest.probe(synthetic_video)
    pitch = cal.make_pitch(PITCH_QUAD, WIDTH, HEIGHT)
    px = np.linspace(200.0, 440.0, info.frames)
    out, plan, written = render.render(
        synthetic_video, px, cfg, info, pitch,
        out_path=tmp_path / "out.mp4", start_s=1.0, end_s=3.0,
    )
    assert out.exists()
    assert written == pytest.approx(60, abs=1)

    result = ingest.probe(out)
    assert (result.width, result.height) == (cfg.render.out_width, cfg.render.out_height)
    assert result.fps == pytest.approx(info.fps)


def test_preview_render_is_downscaled_and_short(tmp_path, synthetic_video):
    cfg = make_config()                            # preview: 2 s at 90 px tall
    info = ingest.probe(synthetic_video)
    pitch = cal.make_pitch(PITCH_QUAD, WIDTH, HEIGHT)
    out, plan, written = render.render(
        synthetic_video, np.full(info.frames, 320.0), cfg, info, pitch,
        out_path=tmp_path / "preview.mp4", preview=True,
    )
    assert (plan.out_width, plan.out_height) == (160, 90)
    assert written == pytest.approx(60, abs=1)
    result = ingest.probe(out)
    assert (result.width, result.height) == (160, 90)


def test_render_past_the_end_is_an_explicit_error(tmp_path, synthetic_video):
    cfg = make_config()
    info = ingest.probe(synthetic_video)
    with pytest.raises(render.RenderError, match="past the end"):
        render.render(synthetic_video, np.full(info.frames, 320.0), cfg, info, None,
                      out_path=tmp_path / "empty.mp4", start_s=999.0)


def test_rendered_pixels_come_from_the_requested_window(tmp_path, synthetic_video):
    """The crop is a real slice of the source, not a resample of the whole frame."""
    cfg = make_config()
    info = ingest.probe(synthetic_video)
    px = np.full(info.frames, 420.0)
    out, _, _ = render.render(synthetic_video, px, cfg, info, None,
                              out_path=tmp_path / "slice.mp4", start_s=0.0, end_s=0.5)
    rects = render.crop_rects(px, cfg, info, None)
    x, y, w, h = rects[0]
    source = ingest.frame_at(synthetic_video, 0, info)[y:y + h, x:x + w]
    rendered = ingest.frame_at(out, 0)
    assert rendered.shape == source.shape
    # h.264 at crf 20 is lossy; compare structure, not exact bytes.
    assert np.mean(np.abs(rendered.astype(float) - source.astype(float))) < 12.0


def test_audio_is_copied_through_untouched(tmp_path, av_clip):
    """Audio is remuxed, never re-encoded, and never worth failing a render over."""
    import av

    cfg = make_config()
    info = ingest.probe(av_clip)
    assert info.has_audio
    out, plan, _ = render.render(
        av_clip, np.full(info.frames, 320.0), cfg, info, None,
        out_path=tmp_path / "with_audio.mp4",
    )
    assert plan.copy_audio is True
    with av.open(str(out)) as container:
        assert container.streams.audio
        assert container.streams.audio[0].codec_context.name == "aac"
        assert container.streams.audio[0].duration > 0


def test_missing_audio_is_not_an_error(tmp_path, synthetic_video):
    cfg = make_config()
    info = ingest.probe(synthetic_video)
    assert not info.has_audio
    out, _, written = render.render(
        synthetic_video, np.full(info.frames, 320.0), cfg, info, None,
        out_path=tmp_path / "silent.mp4", end_s=1.0,
    )
    assert written > 0


def test_sample_frames_is_reproducible(tmp_path, synthetic_video):
    """Exit criterion 1 must be re-measurable: same seed, same frames."""
    first = render.sample_frames(synthetic_video, count=10, seed=7, out_dir=tmp_path / "a")
    second = render.sample_frames(synthetic_video, count=10, seed=7, out_dir=tmp_path / "b")
    assert first["indices"] == second["indices"]
    assert len(set(first["indices"])) == 10
    assert len(first["frames"]) == 10
    assert (tmp_path / "a" / "manifest.json").exists()

    different = render.sample_frames(synthetic_video, count=10, seed=8, out_dir=tmp_path / "c")
    assert different["indices"] != first["indices"]


def test_sampled_pngs_are_readable(tmp_path, synthetic_video):
    manifest = render.sample_frames(synthetic_video, count=3, seed=3, out_dir=tmp_path / "s")
    for entry in manifest["frames"]:
        image = ingest.frame_at(tmp_path / "s" / entry["file"], 0)
        assert image.shape == (HEIGHT, WIDTH, 3)


def test_sampling_more_than_the_file_holds_is_clamped(tmp_path, synthetic_video):
    manifest = render.sample_frames(synthetic_video, count=10_000, seed=1, out_dir=tmp_path / "all")
    assert manifest["requested"] == manifest["frames_total"] == 360
