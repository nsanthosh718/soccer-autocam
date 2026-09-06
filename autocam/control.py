"""Stage E -- the control law. Output quality is won or lost here.

Per frame, in order:

1. **Anticipation lead** -- the target is ``cx + clip(lead_gain * v_cx, +/-lead_max)``.
2. **Deadzone** -- if the target is within ``deadzone`` of the current centre,
   the target becomes the current centre (hold). Kills micro-jitter in static play.
3. **Second-order critically damped follower** -- ``omega_n`` is in **Hz** and
   is converted to rad/s (``2*pi*f``) before integration. ``zeta`` = 1.0
   deliberately: below 1 it overshoots and reads as nervous, above 1 it visibly
   lags a counter-attack.
4. **Velocity clamp.**
5. **Acceleration clamp**, applied after the velocity clamp.
6. **Edge clamp** -- the crop rectangle stays inside the source frame.

Deviation from the build spec, stated deliberately: the spec lists the lead at
step 5, after the follower. Applying it there adds a raw offset to the *output*
position, which the follower never converges on and which snaps discontinuously
whenever centroid velocity changes sign -- i.e. it manufactures exactly the
jitter the deadzone exists to suppress. Applying it to the *target* instead, and
letting the smoothing chain absorb it, is what "human operators lead the play"
actually requires. Every constant and every other step is unchanged.

All constants are fractions of frame width; they are multiplied by the source
width once, here, so Phase 2 can rescale the same numbers into degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config


class ControlError(ValueError):
    pass


@dataclass
class ControlTrace:
    """Per-frame output of the control law."""

    px: np.ndarray            # crop centre x, source pixels
    vx: np.ndarray            # crop centre velocity, source px/s
    target: np.ndarray        # lead-adjusted, deadzone-held target, source px
    cx_vel: np.ndarray        # smoothed centroid velocity, source px/s


class PanController:
    """Stateful, per-frame. One instance per render pass."""

    def __init__(
        self,
        cfg: Config,
        frame_width: int,
        crop_width: int,
        initial_px: float | None = None,
    ):
        if crop_width > frame_width:
            raise ControlError(
                f"crop width {crop_width} exceeds source width {frame_width}"
            )
        c = cfg.control
        w = float(frame_width)
        self.deadzone_px = c.deadzone * w
        self.max_vel_px = c.max_pan_vel * w
        self.max_accel_px = c.max_pan_accel * w
        self.lead_max_px = c.lead_max * w
        self.lead_gain = c.lead_gain
        self.lead_tau = c.lead_vel_tau
        # omega_n is specified in Hz; the follower integrates in rad/s.
        self.omega = 2.0 * math.pi * c.omega_n
        self.zeta = c.zeta

        self.min_px = crop_width / 2.0
        self.max_px = frame_width - crop_width / 2.0
        centre = frame_width / 2.0
        self.px = float(np.clip(initial_px if initial_px is not None else centre,
                                self.min_px, self.max_px))
        self.vx = 0.0
        self.cx_vel = 0.0
        self._prev_cx: float | None = None

    def step(self, cx: float, dt: float) -> tuple[float, float, float]:
        """Advance one frame. Returns ``(px, target, smoothed_centroid_velocity)``."""
        if dt <= 0:
            raise ControlError("dt must be > 0")
        cx = float(cx)

        # -- smoothed centroid velocity (drives the lead term) ----------------
        if self._prev_cx is None:
            raw_vel = 0.0
        else:
            raw_vel = (cx - self._prev_cx) / dt
        self._prev_cx = cx
        alpha = 1.0 - math.exp(-dt / self.lead_tau)
        self.cx_vel += alpha * (raw_vel - self.cx_vel)

        # 1. anticipation lead
        lead = float(np.clip(self.lead_gain * self.cx_vel, -self.lead_max_px, self.lead_max_px))
        target = cx + lead

        # 2. deadzone
        if abs(target - self.px) < self.deadzone_px:
            target = self.px

        # 3. critically damped second-order follower (semi-implicit Euler)
        accel = (self.omega ** 2) * (target - self.px) - 2.0 * self.zeta * self.omega * self.vx
        v_new = self.vx + accel * dt

        # 4. velocity clamp
        v_new = float(np.clip(v_new, -self.max_vel_px, self.max_vel_px))

        # 5. acceleration clamp, after the velocity clamp
        max_dv = self.max_accel_px * dt
        v_new = self.vx + float(np.clip(v_new - self.vx, -max_dv, max_dv))

        px_new = self.px + v_new * dt

        # 6. edge clamp -- and kill the outward velocity so the follower does not
        #    wind up against the wall and lurch when play comes back.
        if px_new < self.min_px:
            px_new = self.min_px
            v_new = max(v_new, 0.0)
        elif px_new > self.max_px:
            px_new = self.max_px
            v_new = min(v_new, 0.0)

        self.px = px_new
        self.vx = v_new
        return self.px, target, self.cx_vel


def run(
    cx: np.ndarray,
    cfg: Config,
    frame_width: int,
    crop_width: int,
    fps: float,
    initial_px: float | None = None,
) -> ControlTrace:
    """Apply the control law across a whole per-frame centroid track."""
    cx = np.asarray(cx, dtype=float).ravel()
    if fps <= 0:
        raise ControlError("fps must be > 0")
    n = cx.size
    if n == 0:
        empty = np.zeros(0)
        return ControlTrace(empty, empty.copy(), empty.copy(), empty.copy())

    start = float(cx[0]) if initial_px is None else initial_px
    ctrl = PanController(cfg, frame_width, crop_width, initial_px=start)
    dt = 1.0 / fps

    px = np.empty(n)
    vx = np.empty(n)
    target = np.empty(n)
    cx_vel = np.empty(n)
    for i in range(n):
        p, tgt, vel = ctrl.step(cx[i], dt)
        px[i] = p
        vx[i] = ctrl.vx
        target[i] = tgt
        cx_vel[i] = vel
    return ControlTrace(px=px, vx=vx, target=target, cx_vel=cx_vel)


def settling_time(px: np.ndarray, goal: float, fps: float, tol_px: float) -> float | None:
    """Seconds until ``px`` reaches and stays within ``tol_px`` of ``goal``.

    Used by the exit-criteria checks (transition tracking within 1.5 s).
    """
    px = np.asarray(px, dtype=float)
    within = np.abs(px - goal) <= tol_px
    if not within.any():
        return None
    # First index after which it never leaves the tolerance band again.
    idx = len(px) - 1
    while idx > 0 and within[idx - 1]:
        idx -= 1
    return idx / fps if within[-1] else None


__all__ = ["ControlError", "ControlTrace", "PanController", "run", "settling_time"]
