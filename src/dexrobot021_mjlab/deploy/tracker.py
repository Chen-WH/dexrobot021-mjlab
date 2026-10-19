"""Object state estimation: fused tag poses in, the pose and velocity terms the
policy expects out.

1.  **Velocity is differentiated, not measured.** Differentiating a pose stream
    amplifies detection noise by `1/dt`, so it is taken over a short fixed window
    (`velocity_window` control steps) rather than frame to frame.

2.  **Dropouts are bursty.** Occlusion lasts tens of milliseconds, not one frame.
    Holding the last pose is right for a short gap and wrong for a long one, so
    after `max_hold_steps` the object is declared lost rather than feeding the
    policy an increasingly stale pose.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .cube import Cube, TagObservation
from .geometry import angular_velocity_from_quats, se3_split


@dataclass
class CubeState:
  """Cube pose and velocity in the wrist (`left_hand_base`) frame."""

  pos: np.ndarray            # (3,) m
  quat: np.ndarray           # (4,) w x y z
  lin_vel: np.ndarray        # (3,) m/s
  ang_vel: np.ndarray        # (3,) rad/s
  age_steps: int             # control steps since the last real detection
  n_tags: int
  valid: bool


class CubeTracker:
  """Turns tag detections into the wrist-frame cube state the policy reads."""

  def __init__(
    self,
    cube: Cube,
    T_hand_cam: np.ndarray,
    control_dt: float,
    velocity_window: int = 3,
    max_hold_steps: int = 6,
    max_orientation_spread_deg: float = 15.0,
  ):
    self.cube = cube
    self.T_hand_cam = np.asarray(T_hand_cam, float)
    self.control_dt = float(control_dt)
    # A 3-step window at 62.5 Hz is 48 ms: long enough that per-frame PnP noise
    # averages down, short enough not to add much latency.
    self.velocity_window = int(velocity_window)
    self.max_hold_steps = int(max_hold_steps)
    self.max_orientation_spread_deg = float(max_orientation_spread_deg)

    self._history: deque[tuple[float, np.ndarray, np.ndarray]] = deque(
      maxlen=max(velocity_window + 1, 2)
    )
    self._last: CubeState | None = None
    self._age = 0
    self.stats = {"frames": 0, "detected": 0, "held": 0, "lost": 0,
                  "rejected_spread": 0}

  def reset(self) -> None:
    self._history.clear()
    self._last = None
    self._age = 0

  def update(self, detections: list[TagObservation], timestamp: float) -> CubeState:
    """Fuse detections and advance one control step. `timestamp` in seconds."""
    T_cam_cube, info = self.cube.pose_from_detections(detections)
    T_hand_cube = None if T_cam_cube is None else self.T_hand_cam @ T_cam_cube
    return self.update_pose(T_hand_cube, timestamp, info)

  def update_pose(self, T_hand_cube, timestamp: float,
                  info: dict | None = None) -> CubeState:
    """Advance one control step from an already-fused wrist-frame pose.

    Split out from `update` so the fusion can live on the capture thread
    (`deploy/pipeline.py`) while the hold/lost bookkeeping stays here, on the
    control loop's clock -- `max_hold_steps` counts control steps.
    """
    info = info or {}
    self.stats["frames"] += 1

    spread = info.get("orientation_spread_deg")
    if T_hand_cube is not None and spread is not None:
      if spread > self.max_orientation_spread_deg:
        # Single-tag votes that disagree by more than this are not noise. The
        # usual cause is a wrong face-to-id table, which shows up as tens of
        # degrees; rejecting is better than averaging two incompatible poses.
        self.stats["rejected_spread"] += 1
        T_hand_cube = None
    self.last_info = info

    if T_hand_cube is None:
      return self._hold()

    self.stats["detected"] += 1
    self._age = 0
    pos, quat = se3_split(T_hand_cube)
    if self._history and np.dot(quat, self._history[-1][2]) < 0.0:
      # Keep the quaternion on one branch across time. Differentiating across a
      # sign flip reports an angular velocity of roughly 2*pi/dt out of nothing.
      quat = -quat
    self._history.append((timestamp, pos, quat))

    lin_vel, ang_vel = self._velocity()
    state = CubeState(pos, quat, lin_vel, ang_vel, 0, info.get("n_used", 1), True)
    self._last = state
    return state

  def _hold(self) -> CubeState:
    """No usable detection this step."""
    # Age advances even before the first detection, so a camera that never sees
    # the object at all still trips the loop's lost-object guard.
    self._age += 1
    if self._last is None:
      self.stats["lost"] += 1
      return CubeState(np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3),
                       np.zeros(3), self._age, 0, False)
    if self._age > self.max_hold_steps:
      self.stats["lost"] += 1
      return CubeState(self._last.pos, self._last.quat, np.zeros(3), np.zeros(3),
                       self._age, 0, False)
    self.stats["held"] += 1
    # Velocity is zeroed while holding rather than extrapolated: a held pose with
    # a live velocity tells the policy the cube is moving when the only thing
    # actually moving is our ignorance.
    return CubeState(self._last.pos, self._last.quat, np.zeros(3), np.zeros(3),
                     self._age, 0, True)

  def _velocity(self) -> tuple[np.ndarray, np.ndarray]:
    if len(self._history) < 2:
      return np.zeros(3), np.zeros(3)
    t0, p0, q0 = self._history[0]
    t1, p1, q1 = self._history[-1]
    dt = t1 - t0
    if dt <= 1e-6:
      return np.zeros(3), np.zeros(3)
    return (p1 - p0) / dt, angular_velocity_from_quats(q0, q1, dt)
