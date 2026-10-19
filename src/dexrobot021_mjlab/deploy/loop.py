"""The control loop, and the conditions under which it must stop.

Timing. The loop runs at the contract's rate -- 62.5 Hz, 16 ms, which is
`sim.timestep * decimation` of the training config; the action EMA, the velocity
observation scale and the reward's spin history are all tuned against it. The
camera runs faster than the loop, so the loop can always find a frame newer than
one control period.

Vision latency is the measured risk: `LoopStats.vision_age_ms` p95 above ~48 ms
end to end (exposure through to the observation vector) is outside the regime the
policy was trained for (up to one control period of pose delay).
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

import numpy as np

from .contract import PolicyContract
from .observation import ObservationEncoder
from .policy import ActionPipeline, Policy
from .tracker import CubeTracker


@dataclass
class SafetyLimits:
  """Everything that stops the loop. All of these end an episode."""

  #: Consecutive control steps with no usable object pose.
  max_lost_steps: int = 12
  #: Object displacement from where the episode started, metres. Also catches a
  #: tracker that has locked onto something that is not the object.
  max_cube_drift_m: float = 0.12
  #: **Measured** joint rate, rad/s -- what the hand achieves, not what it is told
  #: (the policy commands a saturated position, so a limit on the command would
  #: fire every step).
  max_joint_rate: float = 6.0
  #: Consecutive steps above `max_joint_rate` before the episode stops. A finger
  #: slipping off the object snaps free for one or two steps; a divergence holds
  #: the rate high for many.
  max_joint_rate_steps: int = 3
  #: Instantaneous ceiling, rad/s, on the driver's UNCLIPPED rate estimate. Fingers
  #: knocked open by the object have been measured up to ~63 rad/s; this sits above
  #: anything a finger has produced and catches bad readings.
  max_joint_rate_hard: float = 80.0
  #: Count excursions past the soft limit but do not stop for them. The soft limit
  #: protects the experiment, not the hardware (board faults and the hard ceiling
  #: do that); the excursions it fires on are contact releases.
  observe_joint_rate_only: bool = False
  #: Wall-clock overrun of a single control step before the loop gives up, as a
  #: multiple of the control period.
  max_step_overrun: float = 3.0
  #: Episode length. 20 s matches the sim episode.
  max_duration_s: float = 20.0


@dataclass
class LoopStats:
  steps: int = 0
  #: Seconds spent ramping to the start pose before the episode clock started.
  approach_s: float = 0.0
  #: Highest measured joint rate seen (the driver clips it at 15.71 rad/s).
  joint_rate_max: float = 0.0
  #: Same, from the driver's unclipped estimate (`last_raw_rate`).
  joint_rate_raw_max: float = 0.0
  #: Control steps whose unclipped rate exceeded 20 rad/s: contact knocks.
  joint_rate_raw_events: int = 0
  #: `feedback - start pose` after the approach settled, degrees per joint.
  approach_residual_deg: dict = field(default_factory=dict)
  joint_rate_streak: int = 0
  #: Times the soft rate limit was passed for a full streak while observing only.
  joint_rate_excursions: int = 0
  vision_age_ms: list[float] = field(default_factory=list)
  step_ms: list[float] = field(default_factory=list)
  #: `seq` of the pose sample used at each step, so the fraction of steps that
  #: reused the previous estimate is measurable rather than assumed.
  _seen_seq: list[int] = field(default_factory=list)
  stop_reason: str = ""

  def summary(self) -> dict:
    def p95(xs):
      return round(float(np.percentile(xs, 95)), 2) if xs else None

    def mean(xs):
      return round(float(np.mean(xs)), 3) if xs else None

    return {
      "steps": self.steps,
      "duration_s": round(self.steps * 0.016, 2),
      "vision_age_ms_p95": p95(self.vision_age_ms),
      "vision_age_ms_mean": mean(self.vision_age_ms),
      "step_ms_p95": p95(self.step_ms),
      "vision_repeat_frac": (
        round(1.0 - len(set(self._seen_seq)) / max(self.steps, 1), 3)
        if self._seen_seq else None),
      "approach_s": self.approach_s,
      "joint_rate_max": round(self.joint_rate_max, 2),
      "joint_rate_raw_max": round(self.joint_rate_raw_max, 2),
      "joint_rate_raw_events": self.joint_rate_raw_events,
      "approach_residual_max_deg": (
        max(self.approach_residual_deg.values(), key=abs)
        if self.approach_residual_deg else None),
      "approach_residual_deg": self.approach_residual_deg,
      "joint_rate_streak": self.joint_rate_streak,
      "joint_rate_excursions": self.joint_rate_excursions,
      "stop_reason": self.stop_reason,
    }


class ControlLoop:
  """One episode of closed-loop in-hand manipulation."""

  def __init__(
    self,
    contract: PolicyContract,
    policy: Policy,
    hand,
    pose_source,
    tracker: CubeTracker,
    limits: SafetyLimits | None = None,
    clock=time.perf_counter,
    on_step=None,
    approach_rate: float | None = 1.0,
    place_hook=None,
    place_pose=None,
    settle_s: float = 0.3,
    start_tol_deg: float | None = None,
  ):
    self.c = contract
    self.policy = policy
    self.hand = hand
    #: rad/s for the pre-episode ramps; None to skip the approach entirely.
    self.approach_rate = approach_rate
    #: Called once, while the hand waits for the operator to place the object.
    #: None: ramp straight to the start pose and begin.
    self.place_hook = place_hook
    #: Where the hand waits while the operator places the object
    #: (`contract.place_joint_pos`). None: wait in the start pose itself.
    self.place_pose = None if place_pose is None else np.asarray(place_pose, float)
    #: Seconds to keep commanding the start pose after the ramp before step 0,
    #: so the fingers finish arriving instead of the policy starting mid-motion.
    self.settle_s = settle_s
    #: Refuse to start when any joint is further than this from the start pose
    #: after settling. None records the residual without refusing.
    self.start_tol_deg = start_tol_deg
    #: Publishes fused wrist-frame object poses, latest-wins (`deploy/pipeline.py`).
    self.poses = pose_source
    self.tracker = tracker
    self._last_seq = -1
    self.limits = limits or SafetyLimits()
    self.clock = clock
    self.on_step = on_step

    self.encoder = ObservationEncoder(contract)
    # If the hand runs its own command filter, this side must not apply the EMA
    # as well -- two first-order filters in series is not the trained plant.
    self.actions = ActionPipeline(
      contract, apply_ema=not getattr(hand, "applies_command_filter", False)
    )
    self.stats = LoopStats()

  # -- one step ---------------------------------------------------------------

  def step(self, now: float) -> dict:
    sample = self.poses.latest()
    if sample is None or sample.seq == self._last_seq:
      # No estimate, or the same one as last step. A repeat must NOT be re-fed to
      # the tracker: differentiating a repeated pose reports real motion as
      # having stopped.
      cube = self.tracker.update_pose(None, now)
      if sample is not None:
        self.stats.vision_age_ms.append((self.clock() - sample.host_time) * 1e3)
    else:
      self._last_seq = sample.seq
      # The tracker differentiates poses, so it gets the device clock; the age is
      # measured against the host clock.
      cube = self.tracker.update_pose(sample.T_hand_cube, sample.device_time,
                                      sample.info)
      self.stats.vision_age_ms.append((self.clock() - sample.host_time) * 1e3)

    self.stats._seen_seq.append(sample.seq if sample is not None else -1)
    joint_pos, joint_vel = self.hand.read()
    raw_rate = float(getattr(self.hand, "last_raw_rate",
                             np.abs(joint_vel).max()))
    obs = self.encoder.encode(
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      cube=cube,
      last_action=self.actions.last_action,
    )
    if not np.isfinite(obs).all():
      # One NaN anywhere in the observation comes out as NaN on all twelve
      # actions. Hold the start pose; `_violation` stops the episode.
      self._nonfinite_obs = int(np.count_nonzero(~np.isfinite(obs)))
      target = self.actions.step(np.zeros(self.c.n_joints), joint_pos)
    else:
      self._nonfinite_obs = 0
      target = self.actions.step(self.policy(obs), joint_pos)
    self.hand.send(target)
    self.stats.steps += 1
    return {"cube": cube, "target": target, "obs": obs,
            "joint_pos": joint_pos, "joint_vel": joint_vel,
            "joint_rate_raw": raw_rate}

  # -- safety -----------------------------------------------------------------

  def _violation(self, record: dict, elapsed: float) -> str | None:
    if getattr(self, "_nonfinite_obs", 0):
      return f"{self._nonfinite_obs} non-finite values in the observation"
    # A latched board fault silently freezes its joints for the rest of the run.
    errors = self.hand.board_errors() if hasattr(self.hand, "board_errors") else {}
    if errors:
      return "hand board fault: " + "; ".join(f"board {b}: {m}"
                                              for b, m in errors.items())
    cube = record["cube"]
    if not cube.valid and cube.age_steps > self.limits.max_lost_steps:
      return f"object lost for {cube.age_steps} steps"
    if self._origin is not None:
      drift = float(np.linalg.norm(cube.pos - self._origin))
      if cube.valid and drift > self.limits.max_cube_drift_m:
        return f"object drifted {drift * 1e3:.0f} mm from the start"
    rate = float(np.abs(record["joint_vel"]).max())
    raw = float(record.get("joint_rate_raw", rate))
    if raw > self.limits.max_joint_rate_hard:
      return (f"measured joint rate {raw:.1f} rad/s (unclipped), above the "
              f"instantaneous ceiling of {self.limits.max_joint_rate_hard}")
    if rate > self.limits.max_joint_rate:
      self._fast_steps += 1
      self._fast_peak = max(self._fast_peak, rate)
      if self._fast_steps >= self.limits.max_joint_rate_steps:
        if self.limits.observe_joint_rate_only:
          self.stats.joint_rate_excursions += 1
          self._fast_steps = 0
          self._fast_peak = 0.0
        else:
          return (f"measured joint rate above {self.limits.max_joint_rate} rad/s "
                  f"for {self._fast_steps} consecutive steps "
                  f"(peak {self._fast_peak:.1f})")
    else:
      self._fast_steps = 0
      self._fast_peak = 0.0
    if elapsed > self.limits.max_duration_s:
      return "episode duration reached"
    return None

  # -- episode ----------------------------------------------------------------

  def _ramp(self, target: np.ndarray, rate: float, t0: float) -> float:
    """Move the hand to `target` at `rate` rad/s. Returns the time it finished.

    Runs under the stall guard: the closing ramp is exactly where a finger meets
    an object placed a centimetre off and keeps pushing until its board latches.
    A clamped joint stops where it touches instead, and `_settle` reports the
    residual.
    """
    scope = getattr(self.hand, "during_wait", None)
    with scope() if scope is not None else contextlib.nullcontext():
      return self._ramp_unguarded(target, rate, t0)

  def _ramp_unguarded(self, target: np.ndarray, rate: float, t0: float) -> float:
    pos, _ = self.hand.read()
    pos = np.where(np.isfinite(pos), pos, target)
    dt = self.c.control_dt
    span = float(np.abs(target - pos).max())
    steps = max(1, int(np.ceil(span / max(rate * dt, 1e-9))))
    for k in range(1, steps + 1):
      self.hand.send(pos + (target - pos) * (k / steps))
      self.hand.read()
      slack = t0 + k * dt - self.clock()
      if slack > 0:
        time.sleep(slack)
    return t0 + steps * dt

  def approach_default(self, rate: float = 1.0) -> float:
    """Take the hand to the pose a simulated episode starts from.

    In simulation an episode begins with the hand already closed around the
    object (`contract.reset_joint_pos`, NOT `default_joint_pos`). On hardware:

    1. ramp to `place_pose` (the open placement pose), if the bundle has one;
    2. hold it while `place_hook` blocks on the operator placing the object;
    3. close to the start pose at `rate` and verify it arrived (`_close_to_start`).

    Ramping instead of stepping keeps the first command from being a large step
    that trips the rate limit and hits the object at speed. It runs before the
    episode clock starts.
    """
    target = self.c.episode_start_pose()
    moving = 0.0
    if self.place_hook is not None:
      t0 = self.clock()
      wait_pose = target if self.place_pose is None else self.place_pose
      self._ramp(wait_pose, rate, t0)
      moving += self.clock() - t0
      self._check_boards("opening for placement")
      # The hold commands the pose the hand is ALREADY IN; commanding the grasp
      # here would close the hand while the operator is still placing the object.
      self._hold_while(self.place_hook, wait_pose, "waiting for the object to be placed")
    return moving + self._close_to_start(target, rate)

  def _close_to_start(self, target: np.ndarray, rate: float) -> float:
    """Take the hand to the episode's start pose and check that it arrived.

    `_ramp` ends when the COMMAND arrives, not the fingers, so the residual is
    measured after settling. One slower retry before giving up: a finger stopped
    by the stall guard where it first touched the object usually completes the
    close on a second pass.
    """
    moving = 0.0
    for attempt in range(2):
      t0 = self.clock()
      self._ramp(target, rate if attempt == 0 else 0.5 * rate, t0)
      moving += self.clock() - t0
      self._check_boards("approaching the start pose")
      residual = self._settle(target)
      worst = int(np.nanargmax(np.abs(residual)))
      if self.start_tol_deg is None or abs(residual[worst]) <= self.start_tol_deg:
        return moving
      print(f"  start pose off by {residual[worst]:+.1f} deg on "
            f"{self.c.joint_names[worst]} after settling"
            + ("  -- closing again, slower" if attempt == 0 else ""))
    raise RuntimeError(
      f"start pose off by {residual[worst]:+.1f} deg on "
      f"{self.c.joint_names[worst]} after two closing legs "
      f"(tolerance {self.start_tol_deg:g} deg); the policy would start from a pose "
      f"it never trained on. Re-seat the object, or pass a looser --start-tol-deg.")

  def _hold_while(self, hook, target: np.ndarray, doing: str) -> None:
    """Run `hook` (it blocks on the operator) while commanding `target` every period.

    With nothing commanded, an object pushed into a finger makes the firmware
    drive current against a joint that cannot move, and the board latches.
    Commanding through the hand wrapper keeps the stall guard running, so a finger
    the object displaces is clamped where it is instead. `target` stays FIXED so
    the hand returns to a determinate pose afterwards.
    """
    import threading

    done = threading.Event()
    failure: list[BaseException] = []

    def body():
      try:
        hook()
      except BaseException as exc:  # noqa: BLE001 - re-raised on the control thread
        failure.append(exc)
      finally:
        done.set()

    threading.Thread(target=body, daemon=True).start()
    dt = self.c.control_dt
    k = 0
    scope = getattr(self.hand, "during_wait", None)
    with scope() if scope is not None else contextlib.nullcontext():
      while not done.is_set():
        self.hand.read()
        self.hand.send(target)
        k += 1
        if k % 30 == 0:
          self._check_boards(doing)
        time.sleep(dt)
    if failure:
      raise failure[0]

  def _settle(self, target: np.ndarray) -> np.ndarray:
    """Keep commanding the start pose for `settle_s`, then report how far off it is.

    Settling removes the part of the residual that is lag; what remains is contact
    or calibration, and it is recorded either way.
    """
    dt = self.c.control_dt
    t_end = self.clock() + max(self.settle_s, 0.0)
    pos, _ = self.hand.read()
    scope = getattr(self.hand, "during_wait", None)
    with scope() if scope is not None else contextlib.nullcontext():
      while self.clock() < t_end:
        self.hand.send(target)
        time.sleep(dt)
        pos, _ = self.hand.read()
    residual = np.degrees(np.where(np.isfinite(pos), pos - target, np.nan))
    self.stats.approach_residual_deg = {
      name.replace("l_f_joint", ""): round(float(r), 1)
      for name, r in zip(self.c.joint_names, residual, strict=True)}
    return residual

  def _check_boards(self, doing: str) -> None:
    errors = self.hand.board_errors() if hasattr(self.hand, "board_errors") else {}
    if errors:
      raise RuntimeError(f"hand board fault while {doing}: "
                         + "; ".join(f"board {b}: {m}" for b, m in errors.items()))

  def run(self, max_steps: int | None = None) -> LoopStats:
    self.tracker.reset()
    self.actions.reset()
    self.stats = LoopStats()
    self._origin = None
    self._nonfinite_obs = 0
    self._last_seq = -1
    self._fast_steps = 0
    self._fast_peak = 0.0

    # The hand reports only when commanded, so the first `read()` of an episode
    # would block for the driver's timeout (~111 ms against a 16 ms period) and
    # trip the overrun guard on step 1. This costs 2 ms and commands no motion.
    if hasattr(self.hand, "warm_feedback"):
      self.hand.warm_feedback()
    # The network's first forward pass allocates its workspace (~100 ms); pay it
    # before the clock starts.
    try:
      self.policy(np.zeros(self.c.observation_dim, dtype=np.float32))
    except Exception:  # noqa: BLE001 - a warm-up must never be what ends a run
      pass
    if self.approach_rate is not None:
      self.stats.approach_s = round(self.approach_default(self.approach_rate), 3)
    # `_settle` ends on a `read()`, so warm the feedback again before step 0.
    if hasattr(self.hand, "warm_feedback"):
      self.hand.warm_feedback()

    dt = self.c.control_dt
    start = self.clock()
    next_deadline = start
    limit = max_steps if max_steps is not None else int(
      self.limits.max_duration_s / dt
    )

    try:
      for _ in range(limit):
        now = self.clock()
        record = self.step(now)
        cube = record["cube"]
        if self._origin is None and cube.valid:
          self._origin = cube.pos.copy()

        self.stats.step_ms.append((self.clock() - now) * 1e3)
        rate_now = float(np.abs(record["joint_vel"]).max())
        self.stats.joint_rate_max = max(self.stats.joint_rate_max, rate_now)
        raw_now = record.get("joint_rate_raw", rate_now)
        self.stats.joint_rate_raw_max = max(self.stats.joint_rate_raw_max, raw_now)
        self.stats.joint_rate_raw_events += int(raw_now > 20.0)
        if self.on_step is not None:
          self.on_step(self.stats.steps, record)

        reason = self._violation(record, now - start)
        self.stats.joint_rate_streak = max(self.stats.joint_rate_streak,
                                           self._fast_steps)
        if reason:
          self.stats.stop_reason = reason
          break

        next_deadline += dt
        slack = next_deadline - self.clock()
        if slack > 0:
          time.sleep(slack)
        elif -slack > self.limits.max_step_overrun * dt:
          # Falling this far behind means the loop is no longer running the rate
          # the policy was trained at; stop and say so.
          self.stats.stop_reason = (
            f"overran the control period by {-slack * 1e3:.0f} ms")
          break
      else:
        self.stats.stop_reason = self.stats.stop_reason or "step limit reached"
    finally:
      self.stats.stop_reason = self.stats.stop_reason or "interrupted"
    return self.stats
