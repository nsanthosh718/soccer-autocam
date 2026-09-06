# soccer-autocam

Watchable, auto-framed youth soccer footage from a fixed phone on a pole, plus a
per-frame telemetry sidecar for downstream clip tagging.

This repository is **Phase 1**: a post-process virtual camera. A 4K recording
goes in; a 1080p video whose framing follows play comes out, along with
`<video>.telemetry.json`. Phase 2 (a motorized mount) is not built, and stopping
here is a legitimate outcome.

## The core design decision

**The system does not track the ball. It tracks the robust centroid of detected
players.**

Ball detection at youth-match distances -- 30-50 yd, motion blur, variable light
-- is unreliable, and it is the primary failure mode of commercial gimbal
products. Person detection is a solved problem, and the player cluster
co-locates with the ball in the overwhelming majority of frames. Everything in
`centroid.py` follows from that: median rather than mean, MAD outlier rejection,
and a pitch polygon that throws away anyone whose feet are not on the field.

## Capture assumptions

These are requirements on the operator, not on the code. The code cannot
recover from a violation of any of them.

| | |
|---|---|
| Source | iPhone, 3840x2160, 30 or 60 fps, rear ultrawide or wide |
| Camera motion | **Static for the entire recording.** Any pan or bump invalidates the pitch calibration |
| Height | **>= 10 ft**, at or near the halfway line |
| Orientation | Landscape, locked exposure, locked focus |

Camera height dominates output quality more than any parameter in this
repository. At 5 ft the player cluster self-occludes and no algorithm recovers
it. If you change one thing, change the height.

## Install

Apple Silicon Mac, Python 3.11+, `ffmpeg` on `PATH`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[detect,gui,dev]'
```

* `detect` pulls in `ultralytics` + `torch` (the YOLO11 backend, MPS device).
* `gui` pulls in `matplotlib` (the four-corner calibration picker).
* Everything else -- the control law, the render path, telemetry -- runs without
  either, which is why the test suite does not need a GPU.

## Quickstart

```bash
autocam calibrate match.MOV        # click the four pitch corners, once per recording
autocam detect    match.MOV        # ~10 Hz person detection, cached to disk
autocam render    match.MOV --preview-only   # 60 s at 720p, for tuning
autocam render    match.MOV        # the full 1080p output + telemetry
```

or `autocam run match.MOV` to do all of it in one pass.

### Commands

| Command | What it does |
|---|---|
| `autocam calibrate <video>` | Four-corner pitch quad -> `<video>.pitch.json`. `--corners "x1,y1,...,x4,y4"` for non-interactive use; `--full-frame` disables the pitch filter (don't) |
| `autocam detect <video>` | Detection at `detect_hz` -> `<video>.detections.json`. Re-run is a no-op unless a detection constant changed |
| `autocam render <video>` | Control law + crop + encode + telemetry. `--preview-only`, `--start S`, `--end S` |
| `autocam run <video>` | calibrate (if needed) -> detect -> render |
| `autocam sample <rendered>` | Random frames + a seeded manifest, for the manual ball-in-frame count |
| `autocam config` | Print or write the fully resolved config. `--out tuned.json` is the artefact Phase 2 consumes |

Every command takes `--config PATH` (a JSON file layered over the defaults) and
repeatable `--set section.key=value` for one-off overrides.

## How it works

| Stage | Module | What happens |
|---|---|---|
| A | `ingest.py` | ffprobe/PyAV probe. Sub-4K sources rejected. **Rotation metadata honoured before any coordinate math** -- libav hands over unrotated pixels, and getting this wrong looks exactly like a tracking bug |
| B | `calibrate.py` | Four-corner pitch quad. Every detection whose bbox foot-point falls outside it is discarded |
| C | `detect.py` | YOLO11 person detection at 10 Hz on a 1280 px downscale. **Raw** boxes cached to disk |
| D | `centroid.py` | Confidence/area/pitch filtering, then median x with MAD rejection. IQR spread recorded for v1.1 zoom. Fewer than `min_players` survivors marks the timestep invalid and holds the previous target |
| E | `control.py` | Lead -> deadzone -> critically damped follower -> velocity clamp -> acceleration clamp -> edge clamp |
| F | `render.py` | Fixed 1920x1080 window, pan only. H.264 CRF 20, audio remuxed untouched |
| G | `telemetry.py` | `<video>.telemetry.json`: per-frame trace, candidate events, quality block |

### Why detections are cached

The pitch polygon and bbox-area filters are applied when the cache is *read*,
not when it is written. Re-tuning any control, filter or event constant is
therefore a seconds-long cycle rather than a re-run of inference. Only the
constants under `detect` invalidate the cache; `autocam detect` tells you when
that happens and why.

## Tuning

All constants live in `configs/default.json` and nowhere else. Units:
everything spatial under `control` is a **fraction of source frame width**, so
0.03 on a 3840 px source is 115 px; velocities are fw/s, accelerations fw/s^2;
`lead_gain` is in seconds; `omega_n` is in Hz.

Tune against `--preview-only` on a segment that actually contains the problem:

| Symptom | Try |
|---|---|
| Framing twitches during static play | raise `control.deadzone` |
| Framing feels nervous, overshoots | check `control.zeta` is 1.0; lower `control.omega_n` |
| Framing lags on counter-attacks | raise `control.max_pan_vel`, then `control.omega_n`; then `control.lead_gain` |
| Pan visibly whips / reads as robotic | lower `control.max_pan_accel` |
| Crop drifts toward the touchline | re-run `calibrate`; the quad is wrong or the camera moved |
| Crop chases a spectator walking past | lower `detect.bbox_area_max` |
| Long "held" stretches in the run report | camera is too low, or `detect.conf_min` is too high |

When you are done, record the tuned config -- it is the input to Phase 2:

```bash
autocam config --config my-tuning.json --out configs/tuned-2026-09.json
```

`configs/default.json` also carries the `phase2` block (FOV, pan limits,
watchdog, the 2x deadzone widening). One file, both phases, by design: any
constant that exists in two places will diverge.

## Measuring the Phase 1 exit criteria

Phase 1 is a hard gate. Evaluate on a full 60+ minute match, not a clip.

1. **Ball-in-frame rate >= 92%.** `autocam sample match.autocam.mp4 --count 100`
   writes 100 uniformly-random frames plus `manifest.json` recording the seed and
   the sampled indices, so the count is reproducible. Count by hand.
2. **No visible oscillation.** Inspect the `px` trace in the telemetry for
   periodic components during static play, and confirm visually.
3. **Transition tracking.** On a fast counter-attack, the crop centre reaches the
   new play location within 1.5 s, without overshoot. `control.settling_time()`
   measures this off the trace; the shipped constants settle a quarter-frame-width
   step in 1.4 s with zero overshoot (`tests/test_control.py`).
4. **Runtime.** 60-minute 4K source in under 20 minutes with detection cached.
   The run report prints achieved fps and a projected full-length figure.
5. **Subjective.** A full half is watchable end to end without irritation. This
   is the criterion that actually matters. If it fails, the others are decoration.

## Fail visible, not silent

The run report and the telemetry `quality` block carry `invalid_frame_pct`,
`longest_invalid_run_s` and `mean_players_detected`, and warn when a stretch was
*held* rather than tracked. A tracker that quietly framed the wrong half of the
pitch for ten minutes is worse than one that stopped.

The original recording is never modified. The auto-framed output is derived and
disposable.

## Deviations from the build spec

* **Lead is applied to the target, not to the output.** The spec lists
  anticipation lead as step 5, after the follower. Applied there it is a raw
  offset on the output position that the follower never converges on, and it
  snaps discontinuously whenever centroid velocity changes sign -- manufacturing
  exactly the jitter the deadzone exists to suppress. It is applied to the
  target instead, ahead of the deadzone, and the smoothing chain absorbs it.
  Constants and every other step are unchanged. See `control.py`.
* **`omega_n` is interpreted as Hz** (converted to `2*pi*f` rad/s), as the spec
  states. Read as rad/s it would settle a half-frame-width step in ~6 s and fail
  exit criterion 3.
* **The crop is vertically centred on the pitch quad**, not on the frame
  (`render.crop_y_mode`). A 10 ft camera puts the pitch well below the frame
  centre; centring on the frame spends a third of every output frame on sky.
* **A synthetic detection backend exists** alongside YOLO
  (`detect.backend: "synthetic"`), and the YOLO backend imports torch lazily.
  This is what lets the whole pipeline be tested without a GPU. It is not a
  substitute for YOLO on real footage.
* **Calibration has non-interactive paths** (`--corners`, `--full-frame`) so the
  pipeline is scriptable and testable. The interactive four-click picker is
  still the default.

## Deliberately not built

* **Variable zoom** (v1.1). `spread` is computed and carried in telemetry so the
  data is there when it is built; the render path would need crop-then-scale, and
  the zoom filter must run at roughly 1/3 the pan `omega_n` -- zoom that reacts as
  fast as pan is nauseating.
* Multi-camera stitch.
* Vertical (tilt) axis. Soccer at this level is a horizontal problem.
* All of Phase 2.

## Tests

```bash
pytest
```

138 tests, no GPU required. They cover the control law against each of its
documented properties (zero overshoot, both clamps, deadzone, edge clamp, lead
sign and cap), the robust centroid against the stranded-goalkeeper case, the
pitch and area filters, rotation handling, cache invalidation, the telemetry
schema and event debouncing, and a full synthetic match driven through the real
CLI -- including a ball-in-frame check against footage whose ground truth is
known.
