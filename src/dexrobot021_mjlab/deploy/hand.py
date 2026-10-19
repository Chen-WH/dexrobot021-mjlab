"""Hand drivers.

`JOINT_MAP` is the only place the simulator's joint names meet the SDK's
(`pyzlg_dexhand`, DexRobot's CAN driver). The finger-to-joint mapping follows the
MJCF body tree; the sign and zero of each axis follow how the encoders are wired,
so run `scripts/deploy/verify_hand.py` on a new hand before trusting
`sign`/`offset_deg`.

Angles cross this boundary in **degrees** -- `pyzlg_dexhand.move_joints` takes
degrees, everything upstream of here is radians.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

#: Smallest non-zero change any joint's feedback reports (0.09 deg). The feedback
#: is a staircase that refreshes on its own schedule, which is why velocity is
#: differenced across changes rather than across reads (`PyzlgHand.read`).
FEEDBACK_QUANTISATION_RAD = float(np.radians(0.09))


@dataclass(frozen=True)
class JointMapping:
  """One simulator joint and the SDK keyword that drives it."""

  sim_name: str
  sdk_name: str
  sign: float = 1.0
  offset_deg: float = 0.0


# Finger index in `l_f_joint<finger>_<segment>` follows the MJCF body tree:
# 1 = thumb, 2 = index, 3 = middle, 4 = ring, 5 = little.  Segment 1 is the
# base rotation (thumb opposition, four-finger spread), 2 is MCP flexion, 3 is
# the PIP.  The SDK calls that last channel `*_dip`, but the motor and its Hall
# sensor are on the PIP.  Segment 4, the DIP, has no motor: the hardware drives it
# by a rigid linkage and the model by <equality>, which is why 12 of the 19 joints
# appear here.
JOINT_MAP: tuple[JointMapping, ...] = (
  JointMapping("l_f_joint1_1", "th_rot"),
  JointMapping("l_f_joint1_2", "th_mcp"),
  JointMapping("l_f_joint1_3", "th_dip"),
  JointMapping("l_f_joint2_1", "ff_spr"),
  JointMapping("l_f_joint2_2", "ff_mcp"),
  JointMapping("l_f_joint2_3", "ff_dip"),
  JointMapping("l_f_joint3_2", "mf_mcp"),
  JointMapping("l_f_joint3_3", "mf_dip"),
  JointMapping("l_f_joint4_2", "rf_mcp"),
  JointMapping("l_f_joint4_3", "rf_dip"),
  JointMapping("l_f_joint5_2", "lf_mcp"),
  JointMapping("l_f_joint5_3", "lf_dip"),
)


class HandDriver(Protocol):
  """What the control loop needs from a hand."""

  #: True if the hand runs its own command smoothing, in which case the policy
  #: side must not also apply the EMA or the filter is applied twice.
  applies_command_filter: bool

  def connect(self) -> None: ...
  def close(self) -> None: ...
  def send(self, joint_pos_rad: np.ndarray) -> None: ...
  def read(self) -> tuple[np.ndarray, np.ndarray]:
    """`(joint_pos_rad, joint_vel_rad_s)` in the contract's joint order."""

  def warm_feedback(self) -> None:
    """Make the next `read()` cheap. Called once before an episode starts.

    A hand that only reports when commanded blocks in `read()` until its driver
    times out if nothing has commanded it since the last read -- which is
    exactly the state the loop's first step is in.  Measured on the DexHand:
    111 ms, against a 16 ms control period.
    """


class MockHand:
  """A rate-limited first-order hand model, for running the loop with no hardware.

  Not a simulator: it exists so the loop's timing, encoding and safety paths can
  be exercised on a laptop. It has the real command filter's 30 ms time constant
  and a velocity saturation -- the policy commands a saturated position, so a mock
  that tracked the command exactly would trip the loop's rate limit at once.
  """

  applies_command_filter = False

  def __init__(self, joint_names, initial_pos: np.ndarray, control_dt: float,
               tau: float = 0.030, max_rate: float = 3.3):
    self.joint_names = list(joint_names)
    self.control_dt = float(control_dt)
    self.max_rate = float(max_rate)
    self.alpha = 1.0 - float(np.exp(-control_dt / tau))
    self._pos = np.asarray(initial_pos, float).copy()
    self._vel = np.zeros_like(self._pos)
    self._initial = self._pos.copy()
    self.sent: list[np.ndarray] = []

  def connect(self) -> None:
    self._pos = self._initial.copy()
    self._vel[:] = 0.0

  def close(self) -> None:
    pass

  def send(self, joint_pos_rad: np.ndarray) -> None:
    target = np.asarray(joint_pos_rad, float)
    self.sent.append(target.copy())
    step = self.alpha * (target - self._pos)
    bound = self.max_rate * self.control_dt
    new = self._pos + np.clip(step, -bound, bound)
    self._vel = (new - self._pos) / self.control_dt
    self._pos = new

  def read(self) -> tuple[np.ndarray, np.ndarray]:
    self.last_raw_rate = float(np.abs(self._vel).max())
    return self._pos.copy(), self._vel.copy()

  def warm_feedback(self) -> None:
    pass


class PyzlgHand:
  """`pyzlg_dexhand` adapter."""

  applies_command_filter = True  # the SDK runs a 150 Hz alpha=0.2 filter

  def __init__(self, joint_names, mapping=JOINT_MAP, device_index: int = 0,
               control_mode: str = "IMPEDANCE_GRASP",
               velocity_from_feedback: bool = False,
               speed: int | None = None, current: int | None = None):
    self.joint_names = list(joint_names)
    #: Per-motor `speed` / `current` fields of the SDK command frame; None keeps
    #: the SDK defaults (15000 / 20), which is what the released policies ran
    #: with. `current` is packed into one byte, so the usable range is 10-255.
    if current is not None and not 10 <= current <= 255:
      raise ValueError(f"current {current} outside the packable 10-255")
    if speed is not None and not 0 <= speed <= 32767:
      raise ValueError(f"speed {speed} outside 0-32767")
    self.speed = speed
    self.current = current
    self.mapping = {m.sim_name: m for m in mapping}
    missing = [n for n in self.joint_names if n not in self.mapping]
    if missing:
      raise KeyError(f"no SDK mapping for {missing}")
    self.device_index = device_index
    self.control_mode_name = control_mode
    self.velocity_from_feedback = velocity_from_feedback
    self._hand = None
    #: Position and device time at each joint's last *changed* reading. The
    #: velocity is differenced across these, not across successive reads.
    self._change_pos: np.ndarray | None = None
    self._change_t: np.ndarray | None = None
    self._held_vel: np.ndarray | None = None
    self._last_good: np.ndarray | None = None
    #: Per-joint count of reads that came back NaN or missing (a single NaN in the
    #: observation turns all twelve actions into NaN).
    self.nan_reads = np.zeros(len(joint_names), int)
    #: Largest UNCLIPPED rate among the joints whose feedback changed on the last
    #: read. `read()` returns rates bounded at quantisation / 1e-4 s = 15.71 rad/s,
    #: so the control loop's hard ceiling reads this instead.
    self.last_raw_rate = 0.0

  def connect(self) -> None:
    from pyzlg_dexhand.dexhand_interface import ControlMode, LeftDexHand

    self._hand = LeftDexHand()
    if not self._hand.init(device_index=self.device_index):
      raise RuntimeError("pyzlg_dexhand init() failed -- check the CAN adapter")
    self._control_mode = getattr(ControlMode, self.control_mode_name)
    self._last_good = None
    self._change_pos = None
    self._change_t = None
    self._held_vel = None
    self.nan_reads[:] = 0

    # **Motor errors latch, and nothing else clears them.** A joint driven into a
    # stall puts its board into a PARAM_ERROR that survives `init()` and
    # `reset_joints()`: the board stops driving and reporting BOTH its motors, so
    # two joints freeze with no error in the obvious API, and every later session
    # inherits it. Broadcast is avoided: `use_broadcast=True` is broken for this
    # call in the SDK version used here.
    self._hand.clear_errors(use_broadcast=False)
    time.sleep(0.1)

    # Feedback is NaN until the hand has been commanded at least once;
    # `reset_joints()` plus a few commands populate it.
    self._hand.reset_joints()
    for _ in range(5):
      self._hand.move_joints(control_mode=self._control_mode, request_feedback=True)
      time.sleep(0.01)
    pos, _ = self.read()
    if not np.isfinite(pos).all():
      bad = [n for n, v in zip(self.joint_names, pos, strict=True)
             if not np.isfinite(v)]
      raise RuntimeError(
        f"hand feedback still not valid after reset_joints(): {bad}. "
        "Power-cycle the hand or check the CAN wiring for those boards."
      )

  def board_errors(self) -> dict[int, str]:
    """Boards currently reporting a motor fault, as `{board index: description}`.

    Read from `board_states` rather than `get_errors()`, which has been seen to
    return `None` for boards latched in PARAM_ERROR.
    """
    if self._hand is None:
      return {}
    out = {}
    for index, state in getattr(self._hand, "board_states", {}).items():
      info = getattr(state, "error_info", None)
      if info is not None:
        out[index] = f"{info.error_type.name}: {info.description}"
      elif not getattr(state, "is_normal", True):
        out[index] = "not normal, no error info"
    return out

  def close(self) -> None:
    if self._hand is not None:
      try:
        self._hand.reset_joints()
      finally:
        self._hand = None

  def send(self, joint_pos_rad: np.ndarray) -> None:
    if self._hand is None:
      raise RuntimeError("connect() first")
    kwargs = {}
    for value, name in zip(np.asarray(joint_pos_rad, float), self.joint_names,
                           strict=True):
      m = self.mapping[name]
      kwargs[m.sdk_name] = float(m.sign * np.degrees(value) + m.offset_deg)
    if self.speed is not None or self.current is not None:
      from pyzlg_dexhand.dexhand_interface import JointCommand
      kwargs = {k: JointCommand(position=v, current=self.current, velocity=self.speed)
                for k, v in kwargs.items()}
    # `request_feedback=True` is not optional: the boards report only in response
    # to a command, and without it `read()` occasionally blocks for the driver's
    # ~111 ms timeout (worst case 114 ms vs 2.4 ms with the flag).
    self._hand.move_joints(control_mode=self._control_mode,
                           request_feedback=True, **kwargs)

  def warm_feedback(self) -> None:
    """Re-issue the current targets and ask the boards to report.

    An angle-less `move_joints()` holds whatever was last commanded, so this
    moves nothing; it exists only so the first `read()` of an episode finds
    feedback waiting instead of paying the driver's ~111 ms timeout. Measured:
    a cold read costs 111 ms, the read after this costs 0.4 ms.
    """
    if self._hand is None:
      raise RuntimeError("connect() first")
    self._hand.move_joints(control_mode=self._control_mode,
                           request_feedback=True)
    time.sleep(0.002)

  def read_motor(self) -> dict[str, np.ndarray]:
    """Motor-side feedback from the LAST `read()`: current (mA), speed (rpm),
    impedance, error code, per joint in `joint_names` order (NaN where absent).

    Taken from the feedback `read()` already fetched, never a second
    `get_feedback()` -- that one blocks ~111 ms.
    """
    keys = ("current", "velocity", "impedance", "error_code", "encoder_position")
    out = {k: np.full(len(self.joint_names), np.nan) for k in keys}
    feedback = getattr(self, "_last_feedback", None)
    if feedback is None:
      return out
    for i, name in enumerate(self.joint_names):
      joint = feedback.joints.get(self.mapping[name].sdk_name)
      if joint is None:
        continue
      for k in keys:
        v = getattr(joint, k, None)
        if v is not None:
          out[k][i] = float(v)
    return out

  def read(self) -> tuple[np.ndarray, np.ndarray]:
    if self._hand is None:
      raise RuntimeError("connect() first")
    feedback = self._hand.get_feedback()
    self._last_feedback = feedback
    now = float(getattr(feedback, "query_timestamp", time.time()))
    # The feedback's own clock, so a caller never needs a second `get_feedback()`.
    self.last_read_time = now
    pos = np.full(len(self.joint_names), np.nan)
    for i, name in enumerate(self.joint_names):
      m = self.mapping[name]
      joint = feedback.joints.get(m.sdk_name)
      if joint is None:
        continue
      angle = getattr(joint, "angle", getattr(joint, "position", None))
      if angle is None:
        continue
      pos[i] = np.radians((float(angle) - m.offset_deg) / m.sign)

    # A joint whose feedback blinked keeps its last good value rather than
    # entering the observation as NaN. Zero would be worse than either: it is a
    # plausible number, so nothing downstream would notice, and it tells the
    # policy the finger is straight when it may be fully flexed.
    missing = ~np.isfinite(pos)
    if missing.any():
      self.nan_reads += missing
      if self._last_good is None:
        return pos, np.zeros_like(pos)   # caller must check; connect() does
      pos[missing] = self._last_good[missing]
    self._last_good = pos.copy()

    # The SDK reports position, not rate, so velocity is differenced -- but not
    # across successive reads. The feedback is a 0.09 deg staircase that refreshes
    # on its own schedule: most reads repeat the previous value, and the read that
    # finally moves carries the whole accumulated step. Dividing that by the time
    # since the last READ reports spikes of 10-30 rad/s from a still hand. So each
    # joint is differenced across the reads where it actually changed, over the
    # interval it changed across, and that rate is held in between.
    vel = np.zeros_like(pos)
    if self._change_pos is None:
      self._change_pos = pos.copy()
      self._change_t = np.full(len(pos), now)
      self._held_vel = np.zeros_like(pos)
      self.last_raw_rate = 0.0
    else:
      moved = np.abs(pos - self._change_pos) > 0.5 * FEEDBACK_QUANTISATION_RAD
      fresh = moved & ((now - self._change_t) > 1e-4)
      self.last_raw_rate = 0.0
      if fresh.any():
        self._held_vel[fresh] = ((pos[fresh] - self._change_pos[fresh])
                                 / (now - self._change_t[fresh]))
        self.last_raw_rate = float(np.abs(self._held_vel[fresh]).max())
        self._change_pos[fresh] = pos[fresh]
        self._change_t[fresh] = now
      # A joint that has not changed for `stale` seconds cannot be turning
      # faster than one quantisation step per `stale`, so bound the held rate by
      # that. The bound tightens on its own as the silence lengthens, which
      # brings a stopped joint to zero without inventing a decay constant, and
      # leaves a joint that just moved untouched.
      stale = np.maximum(now - self._change_t, 1e-4)
      bound = FEEDBACK_QUANTISATION_RAD / stale
      vel = np.clip(self._held_vel, -bound, bound)
    return pos, vel
