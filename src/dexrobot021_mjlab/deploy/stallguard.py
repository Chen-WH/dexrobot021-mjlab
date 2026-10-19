"""Pull a blocked joint's target back to where the joint actually is.

Measured with a force sensor under the fingertip:

* A motor board latches PARAM_ERROR when its motor keeps drawing current while
  the joint does not move, after ~0.2-1 s, at any force from 2 to 7 N. It is a
  stall trip, not a force limit, and it hits the HOLDING joint as well as the
  pushing one.
* The force a finger has built is held by the self-locking worm at ~0 mA:
  pulling the target back to the measured angle keeps the force without a latch,
  and short pushes separated by pull-backs never latch (the stall timer resets).

It runs while the operator places or removes the object (`during_wait()`, entered by
`ControlLoop._hold_while`) and during the loop's own approach ramps (`_ramp`,
`_settle`), where a finger the object pushes aside would otherwise keep being driven
back and latch its board. During episodes it is a pass-through: the policy is not
guarded. Per joint, while active:

  FREE     -> send the policy's target. Count consecutive steps where the target
              is > `detect_deg` ahead and the joint made < `progress_deg` of net
              progress toward it over the last `window` reads. `trigger` such
              steps -> CLAMPED. The count is what lets a joint that is merely
              STARTING from rest get going: the first steps of any move look
              exactly like a block.
  CLAMPED  -> send `measured +/- margin_deg`. Every `probe_every` steps release
              the real target for `probe_steps` (shorter than the fastest latch);
              progress during the probe, or the target falling back inside
              `detect_deg`, -> FREE.

Net progress, not movement: a stationary joint's reading toggles between two
quantisation levels ~0.5 deg apart, which a plain range test calls motion.
Nothing here changes what the policy sees; its observation keeps its own
`last_action`.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class StallGuardConfig:
  window: int = 8            # reads over which progress is measured (128 ms at 62.5 Hz)
  detect_deg: float = 0.7    # target ahead of the joint by more than this ...
  progress_deg: float = 0.35 # ... with less net progress than this -> a blocked step
  trigger: int = 6           # consecutive blocked steps before clamping (96 ms; fastest latch 200 ms)
  margin_deg: float = 0.4    # clamped target = measured +/- this
  probe_every: int = 20      # while clamped, release the real target every this many steps ...
  probe_steps: int = 5       # ... for this many (80 ms)


#: What the guard uses inside `during_wait()`. A person pushing a joint aside does not
#: stop it dead -- it creeps, about 1 deg per window, which slips past `progress_deg`
#: 0.35 and latches the board anyway. No policy runs during a wait, so over-clamping
#: costs nothing. `detect_deg` stays at the default: the board latches from about
#: 0.6 deg of lead, so raising it would leave a band that latches and is never
#: clamped (`tests/test_stallguard.py`). Only the creep tolerance moves.
WAIT_CONFIG = StallGuardConfig(progress_deg=2.0, trigger=4)


class StallGuardHand:
  def __init__(self, hand, n_joints: int, config: StallGuardConfig | None = None):
    self._inner = hand
    self.cfg = config or StallGuardConfig()
    self.active = False
    self._in_wait = False
    self._hist = deque(maxlen=self.cfg.window)
    n = n_joints
    self._count = np.zeros(n, int)
    self._clamped = np.zeros(n, bool)
    self._since = np.zeros(n, int)       # steps since clamping / since the probe began
    self._probe_start = np.full(n, np.nan)
    self.last_requested = None
    self.last_sent = None
    self.last_blocked = np.zeros(n, bool)
    self.clamped_steps = np.zeros(n, int)
    self.clamp_events = np.zeros(n, int)
    self.sends = 0
    #: The same counts outside the operator waits (approach + episodes; the trace has per-step `clamped`).
    self.ep_clamped_steps = np.zeros(n, int)
    self.ep_sends = 0

  def _reset_state(self) -> None:
    self._count[:] = 0
    self._clamped[:] = False
    self._since[:] = 0
    self._probe_start[:] = np.nan
    self.last_blocked = np.zeros_like(self._clamped)

  @contextmanager
  def during_wait(self):
    """Guard for the duration of an operator wait or one of our own ramps.

    Swaps in `WAIT_CONFIG`, which clamps a joint that is merely creeping. Outside
    this scope the guard is a pass-through: guarding a running policy clamps the
    very pushes that hold the object.
    """
    self._reset_state()
    episode_cfg, self.cfg = self.cfg, WAIT_CONFIG
    self._hist = deque(self._hist, maxlen=WAIT_CONFIG.window)
    self.active = True
    self._in_wait = True
    try:
      yield
    finally:
      self._reset_state()
      self.cfg = episode_cfg
      self._hist = deque(self._hist, maxlen=episode_cfg.window)
      self.active = False
      self._in_wait = False

  def __getattr__(self, name):          # everything else is the driver's
    return getattr(self._inner, name)

  def read(self):
    pos, vel = self._inner.read()
    if np.all(np.isfinite(pos)):
      self._hist.append(np.array(pos, float))
    return pos, vel

  def guard(self, target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, float)
    c = self.cfg
    if not self.active:
      self.last_blocked = np.zeros(target.shape, bool)
      return target
    if len(self._hist) < c.window:
      self.last_blocked = np.zeros(target.shape, bool)
      return target
    hist = np.stack(self._hist)
    now = hist[-3:].mean(0)                         # beats the two-level toggle
    err = target - now
    progress = (now - hist[:3].mean(0)) * np.sign(err)
    ahead = np.abs(err) > np.radians(c.detect_deg)
    stuck = ahead & (progress < np.radians(c.progress_deg))

    # FREE joints: count, and clamp after `trigger` consecutive stuck steps.
    free = ~self._clamped
    self._count = np.where(free & stuck, self._count + 1, np.where(free, 0, self._count))
    new = free & (self._count >= c.trigger)
    self._clamped |= new
    self._since[new] = 0
    self.clamp_events += new

    # CLAMPED joints: release when the target comes back, or when a probe shows progress.
    self._since[self._clamped] += 1
    phase = self._since % c.probe_every
    probing = self._clamped & (phase >= c.probe_every - c.probe_steps)
    probe_first = self._clamped & (phase == c.probe_every - c.probe_steps)
    self._probe_start[probe_first] = now[probe_first]
    probe_progress = (now - self._probe_start) * np.sign(err)
    release = self._clamped & (~ahead | (probing & (probe_progress > np.radians(c.progress_deg))))
    self._clamped &= ~release
    self._count[release] = 0

    m = np.radians(c.margin_deg)
    hold = self._clamped & ~probing
    self.last_blocked = hold
    # Clamp relative to the reading LEAST advanced toward the target: a toggling
    # reading's mean can sit past the true angle, and margin is measured from there.
    base = np.where(err > 0, hist[-3:].min(0), hist[-3:].max(0))
    return np.where(hold, base + np.clip(target - base, -m, m), target)

  def send(self, joint_pos_rad):
    self.last_requested = np.asarray(joint_pos_rad, float)
    out = self.guard(self.last_requested)
    self.last_sent = out
    self.clamped_steps += self.last_blocked
    self.sends += 1
    if not self._in_wait:
      self.ep_clamped_steps += self.last_blocked
      self.ep_sends += 1
    self._inner.send(out)

  def summary(self, joint_names) -> dict:
    frac = self.clamped_steps / max(self.sends, 1)
    ep = self.ep_clamped_steps / max(self.ep_sends, 1)
    names = list(joint_names)
    return {"sends": int(self.sends), "config": asdict(self.cfg),
            "outside_wait_sends": int(self.ep_sends),
            "outside_wait_clamped_fraction": {
              n: round(float(f), 4) for n, f in zip(names, ep, strict=True) if f > 0},
            "clamped_fraction": {
              n: round(float(f), 4) for n, f in zip(names, frac, strict=True) if f > 0},
            "clamp_events": {
              n: int(e) for n, e in zip(names, self.clamp_events, strict=True) if e > 0}}
