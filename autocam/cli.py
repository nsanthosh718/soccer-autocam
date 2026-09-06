"""Command line entrypoint.

    autocam calibrate <video>     capture/refresh the pitch quad
    autocam detect    <video>     run + cache detections
    autocam render    <video>     apply the control law and encode
    autocam run       <video>     calibrate if needed, detect, render
    autocam config                print/write the resolved config (Phase 2 input)

``--preview-only`` renders a short segment at 720p. It is the flag you will use
most; it reuses the cached detections, so a constant-tuning cycle is seconds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click

from . import calibrate as cal
from . import centroid, config as config_mod, control, detect as detect_mod
from . import ingest, render as render_mod, telemetry as telemetry_mod
from .config import Config, ConfigError

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

# Every failure the pipeline raises deliberately. They carry a message written
# for the operator, so surface that -- never a traceback.
DOMAIN_ERRORS = (
    ConfigError,
    cal.CalibrationError,
    control.ControlError,
    detect_mod.DetectionError,
    ingest.IngestError,
    render_mod.RenderError,
)


class AutocamGroup(click.Group):
    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except DOMAIN_ERRORS as exc:
            raise click.ClickException(str(exc)) from exc


def _parse_set(values: tuple[str, ...]) -> dict:
    """``--set control.omega_n=1.1`` -> ``{"control": {"omega_n": 1.1}}``."""
    out: dict = {}
    for item in values:
        if "=" not in item:
            raise click.BadParameter(f"expected section.key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def _load_config(path: str | None, overrides: tuple[str, ...]) -> Config:
    try:
        return config_mod.load(path, _parse_set(overrides))
    except ConfigError as exc:
        raise click.ClickException(f"config error: {exc}") from exc


def _progress(label: str):
    state = {"last": 0.0}

    def report(done: int, total: int) -> None:
        now = time.monotonic()
        if now - state["last"] < 0.5 and done < total:
            return
        state["last"] = now
        pct = 100.0 * done / max(total, 1)
        click.echo(f"\r[{label}] {pct:5.1f}%  ({done}/{total})", nl=False, err=True)

    return report


def _end_progress() -> None:
    click.echo("", err=True)


def _require_pitch(video: Path, info: ingest.SourceInfo) -> cal.Pitch:
    pitch = cal.load_if_present(video)
    if pitch is None:
        raise click.ClickException(
            f"no pitch calibration for {video}. Run `autocam calibrate {video}` first "
            "(or `--full-frame` if you really want no pitch filter)."
        )
    cal.check_matches_source(pitch, info.width, info.height)
    return pitch


def _probe(video: Path, cfg: Config, allow_low_res: bool) -> ingest.SourceInfo:
    info = ingest.probe(video)
    try:
        ingest.check_resolution(info, cfg)
    except ingest.IngestError:
        if not allow_low_res:
            raise
        click.secho(
            f"[warn] source is {info.width}x{info.height}, below the 4K requirement; "
            "proceeding because --allow-low-res was given",
            fg="yellow",
            err=True,
        )
    return info


video_arg = click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
config_opt = click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
                          help="JSON layered over configs/default.json.")
set_opt = click.option("--set", "overrides", multiple=True, metavar="SECTION.KEY=VALUE",
                       help="Override one constant. Repeatable.")
low_res_opt = click.option("--allow-low-res", is_flag=True,
                           help="Proceed on a sub-4K source (warns loudly).")


@click.group(cls=AutocamGroup, context_settings=CONTEXT_SETTINGS)
@click.version_option(package_name="soccer-autocam", prog_name="autocam")
def main() -> None:
    """Auto-framed soccer footage from a fixed 4K camera."""


@main.command()
@video_arg
@config_opt
@set_opt
@low_res_opt
@click.option("--corners", help='Non-interactive: "x1,y1,x2,y2,x3,y3,x4,y4" in source pixels.')
@click.option("--full-frame", is_flag=True,
              help="Use the whole frame as the pitch. Disables the spectator filter.")
@click.option("--frame", "frame_index", default=0, show_default=True,
              help="Which frame to calibrate against.")
@click.option("--force", is_flag=True, help="Overwrite an existing calibration.")
def calibrate(video, config_path, overrides, allow_low_res, corners, full_frame, frame_index, force):
    """Capture the four pitch corners and persist <video>.pitch.json."""
    cfg = _load_config(config_path, overrides)
    info = _probe(video, cfg, allow_low_res)

    existing = cal.pitch_path_for(video)
    if existing.exists() and not force:
        click.echo(f"{existing} already exists; pass --force to replace it.")
        return

    if corners:
        pitch = cal.make_pitch(cal.parse_corners(corners), info.width, info.height)
    elif full_frame:
        pitch = cal.full_frame_pitch(info.width, info.height)
        click.secho(
            "[warn] full-frame calibration: detections will NOT be filtered to the "
            "pitch, so spectators and adjacent matches can pull the framing.",
            fg="yellow", err=True,
        )
    else:
        image = ingest.frame_at(video, frame_index, info)
        pitch = cal.make_pitch(
            cal.pick_corners_interactive(image, title=str(video)), info.width, info.height
        )

    path = cal.save(pitch, video)
    click.echo(f"pitch quad -> {path}")
    for x, y in pitch.quad:
        click.echo(f"  ({x:8.1f}, {y:8.1f})")
    click.echo(f"  covers {100 * cal.polygon_area(pitch.quad) / (info.width * info.height):.1f}% of frame")


@main.command()
@video_arg
@config_opt
@set_opt
@low_res_opt
@click.option("--force", is_flag=True, help="Re-run inference even if the cache is valid.")
def detect(video, config_path, overrides, allow_low_res, force):
    """Run detection at detect_hz and cache raw boxes to <video>.detections.json."""
    cfg = _load_config(config_path, overrides)
    info = _probe(video, cfg, allow_low_res)
    click.echo(
        f"{video}: {info.width}x{info.height} @ {info.fps:.3f} fps, "
        f"{info.frames} frames, rotation {info.rotation} deg"
    )
    t0 = time.monotonic()
    cache, reused = detect_mod.load_or_run(
        video, cfg, info, progress=_progress("detect"), force=force
    )
    _end_progress()
    if reused:
        click.echo(f"reused cached detections ({cache.n_timesteps} timesteps)")
        return
    boxes = sum(len(ts["boxes"]) for ts in cache.timesteps)
    click.echo(
        f"{cache.n_timesteps} timesteps, {boxes} raw boxes "
        f"(every {cache.step} frames) in {time.monotonic() - t0:.1f}s "
        f"-> {detect_mod.detections_path_for(video)}"
    )


def _compute(video: Path, cfg: Config, info: ingest.SourceInfo, pitch: cal.Pitch, frames: int):
    """Detections -> timesteps -> per-frame track -> control trace."""
    cache = detect_mod.load_cache(video)
    ok, why = detect_mod.cache_is_valid(cache, cfg, info)
    if not ok:
        raise click.ClickException(
            f"cached detections are stale ({why}). Re-run `autocam detect {video}`."
        )
    timesteps = centroid.analyse_timesteps(cache, cfg, pitch, info)
    track = centroid.build_track(timesteps, cfg, info, frames=frames)
    trace = control.run(track.cx, cfg, info.width, cfg.render.out_width, info.fps)
    return track, trace


def _print_report(video, info, cfg, quality, plan, out_path, written, seconds, pitch):
    click.echo("")
    click.secho("=== run report ===", bold=True)
    click.echo(f"source            {video} ({info.width}x{info.height} @ {info.fps:.3f} fps)")
    click.echo(f"pitch quad        {'full frame (FILTER DISABLED)' if pitch.note else 'calibrated'}")
    click.echo(f"config hash       {cfg.hash()[:16]}")
    click.echo(f"output            {out_path} ({written} frames, {plan.out_width}x{plan.out_height})")
    click.echo(f"invalid frames    {quality['invalid_frame_pct']:.2f}%")
    click.echo(f"longest hold      {quality['longest_invalid_run_s']:.2f}s")
    click.echo(f"mean players      {quality['mean_players_detected']:.2f}")
    rate = written / seconds if seconds > 0 else 0.0
    click.echo(f"render time       {seconds:.1f}s ({rate:.1f} fps)")
    if seconds > 0 and info.frames:
        projected = info.frames / rate / 60.0 if rate else float("inf")
        click.echo(f"projected full    {projected:.1f} min for {info.frames} frames")
    for warning in quality["warnings"]:
        click.secho(f"[warn] {warning}", fg="yellow")
    if not quality["warnings"]:
        click.secho("no quality warnings", fg="green")


@main.command()
@video_arg
@config_opt
@set_opt
@low_res_opt
@click.option("--preview-only", is_flag=True,
              help="Render a short 720p segment for fast constant tuning.")
@click.option("--start", "start_s", type=float, default=None, help="Start time, seconds.")
@click.option("--end", "end_s", type=float, default=None, help="End time, seconds.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None)
@click.option("--no-telemetry", is_flag=True, help="Skip the JSON sidecar.")
def render(video, config_path, overrides, allow_low_res, preview_only, start_s, end_s,
           out_path, no_telemetry):
    """Apply the control law to cached detections and encode the output."""
    cfg = _load_config(config_path, overrides)
    info = _probe(video, cfg, allow_low_res)
    pitch = _require_pitch(video, info)
    frames = ingest.frame_count(info)

    track, trace = _compute(video, cfg, info, pitch, frames)
    quality = centroid.quality_report(track, cfg)
    crops = render_mod.crop_rects(trace.px, cfg, info, pitch)

    watch = render_mod.Stopwatch()
    out, plan, written = render_mod.render(
        video, trace.px, cfg, info, pitch,
        out_path=out_path, start_s=start_s, end_s=end_s,
        preview=preview_only, progress=_progress("render"),
    )
    _end_progress()
    seconds = watch.elapsed()

    if not no_telemetry:
        doc = telemetry_mod.build(
            video, info, cfg, pitch, track, trace, crops, quality,
            extra={"render": {
                "output": str(out),
                "preview": bool(preview_only),
                "out_width": plan.out_width,
                "out_height": plan.out_height,
                "frames_written": written,
                "start_s": plan.start_s,
                "end_s": plan.end_s,
                "seconds": round(seconds, 2),
            }},
        )
        path = telemetry_mod.save(doc, video)
        click.echo(f"telemetry -> {path} ({len(doc['events'])} candidate events)")

    _print_report(video, info, cfg, quality, plan, out, written, seconds, pitch)


@main.command()
@video_arg
@config_opt
@set_opt
@low_res_opt
@click.option("--preview-only", is_flag=True)
@click.option("--start", "start_s", type=float, default=None)
@click.option("--end", "end_s", type=float, default=None)
@click.option("--full-frame", is_flag=True,
              help="Calibrate to the whole frame if no calibration exists.")
@click.pass_context
def run(ctx, video, config_path, overrides, allow_low_res, preview_only, start_s, end_s, full_frame):
    """Calibrate if needed, detect, then render."""
    cfg = _load_config(config_path, overrides)
    _probe(video, cfg, allow_low_res)      # fail on an unusable source before any work
    if cal.load_if_present(video) is None:
        ctx.invoke(calibrate, video=video, config_path=config_path, overrides=overrides,
                   allow_low_res=allow_low_res, corners=None, full_frame=full_frame,
                   frame_index=0, force=False)
    ctx.invoke(detect, video=video, config_path=config_path, overrides=overrides,
               allow_low_res=allow_low_res, force=False)
    ctx.invoke(render, video=video, config_path=config_path, overrides=overrides,
               allow_low_res=allow_low_res, preview_only=preview_only,
               start_s=start_s, end_s=end_s, out_path=None, no_telemetry=False)


@main.command()
@video_arg
@click.option("--count", default=100, show_default=True, help="Frames to sample.")
@click.option("--seed", default=1234, show_default=True, help="Sampling seed (logged).")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default=None)
def sample(video, count, seed, out_dir):
    """Export random frames of a RENDERED file for the manual ball-in-frame count."""
    manifest = render_mod.sample_frames(video, count=count, seed=seed, out_dir=out_dir)
    click.echo(f"{len(manifest['frames'])} frames -> {Path(out_dir) if out_dir else 'alongside the video'}")
    click.echo(f"seed {manifest['seed']}, indices logged in manifest.json")
    click.echo("Count the frames showing the ball inside the crop; >= 92% passes.")


@main.command("config")
@config_opt
@set_opt
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None,
              help="Write the resolved config here -- this is the Phase 2 input.")
def config_cmd(config_path, overrides, out_path):
    """Print the fully resolved config and its hash."""
    cfg = _load_config(config_path, overrides)
    if out_path:
        config_mod.dump(cfg, out_path)
        click.echo(f"resolved config -> {out_path}")
    else:
        click.echo(json.dumps(cfg.to_dict(), indent=2))
    click.echo(f"config_hash {cfg.hash()}", err=True)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
