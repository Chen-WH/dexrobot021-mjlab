"""Object pose estimation on its own thread.

The control loop has to run at exactly 62.5 Hz. Capture and AprilTag detection run
on the camera's schedule and cost whatever the scene costs, so running them inline
lets a slow frame eat the control period. Detection runs here, on a daemon thread,
and publishes **latest-wins**: the loop reads the most recent completed estimate
and never waits. A slow frame then costs freshness (measured as
`vision_age_ms`), not the control rate.

`seq` makes the hand-off honest: some control steps legitimately see the same
estimate twice, and feeding a repeated pose into a finite-difference velocity would
report real motion as having stopped. The loop advances its tracker only when
`seq` changes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .cube import MAX_OBLIQUITY_DEG, Cube


@dataclass
class PoseSample:
  """One completed estimate, as published to the control loop."""

  T_hand_cube: np.ndarray | None
  device_time: float           # camera clock, for differentiation
  host_time: float             # perf_counter at capture, for age
  n_tags: int
  seq: int                     # increments once per processed frame
  info: dict = field(default_factory=dict)


class ThreadedPoseSource:
  """Capture -> detect -> fuse on a daemon thread."""

  def __init__(self, frame_source, detector, cube: Cube, T_hand_cam: np.ndarray,
               max_obliquity_deg: float = MAX_OBLIQUITY_DEG):
    self.frames = frame_source
    self.detector = detector
    self.cube = cube
    self.T_hand_cam = np.asarray(T_hand_cam, float)
    self.max_obliquity_deg = float(max_obliquity_deg)
    #: Optional `(tag_id, T_hand_wristtag)`. When set, `T_hand_cam` is recomputed
    #: from the wrist tag on every frame instead of being the constant handed in.
    #:
    #: This removes the camera's extrinsic from the chain: the object pose and the
    #: wrist tag pose come out of the SAME image, so
    #: `T_hand_wristtag . inv(T_cam_wristtag) . T_cam_cube` never mentions where
    #: the camera is, and much of the intrinsic and distortion bias cancels between
    #: the two. It also follows the arm if it moves.
    self.wrist_ref: tuple[int, np.ndarray] | None = None

    self._lock = threading.Lock()
    self._latest: PoseSample | None = None
    self._thread: threading.Thread | None = None
    self._stop = threading.Event()
    self._seq = 0
    #: `wrist_missing` counts frames where the wrist tag was not decoded and the
    #: previous transform stood in: the object pose is still produced, against a
    #: slightly stale reference.
    self.stats = {"frames": 0, "fused": 0, "empty": 0, "skipped": 0,
                  "wrist_missing": 0}
    self.detect_ms: list[float] = []

  def _hand_from_cam(self, detections) -> np.ndarray:
    """`T_hand_cam` for this frame: constant, or solved from the wrist tag."""
    if self.wrist_ref is None:
      return self.T_hand_cam
    tag_id, T_hand_wristtag = self.wrist_ref
    for det in detections:
      if int(det.tag_id) != int(tag_id):
        continue
      T_cam_wristtag = np.eye(4)
      T_cam_wristtag[:3, :3] = det.R
      T_cam_wristtag[:3, 3] = det.t
      self.T_hand_cam = T_hand_wristtag @ np.linalg.inv(T_cam_wristtag)
      return self.T_hand_cam
    self.stats["wrist_missing"] += 1
    return self.T_hand_cam

  # -- lifecycle --------------------------------------------------------------

  def start(self) -> None:
    if self._thread is not None:
      return
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, name="pose", daemon=True)
    self._thread.start()

  def stop(self, timeout: float = 2.0) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=timeout)
      self._thread = None

  def __enter__(self) -> ThreadedPoseSource:
    self.start()
    return self

  def __exit__(self, *exc) -> None:
    self.stop()

  # -- producer ---------------------------------------------------------------

  def _run(self) -> None:
    while not self._stop.is_set():
      frame = self.frames.read(block=True)
      if frame is None:
        continue
      # Drain to the newest queued frame before spending time on detection;
      # otherwise the thread falls further behind every time a detection takes
      # longer than a frame period.
      for _ in range(8):
        newer = self.frames.read(block=False)
        if newer is None:
          break
        frame = newer
        self.stats["skipped"] += 1
      self.stats["frames"] += 1

      started = time.perf_counter()
      detections = self.detector.detect(frame)
      self.detect_ms.append((time.perf_counter() - started) * 1e3)

      T_cam_cube, info = self.cube.pose_from_detections(
        detections, max_obliquity_deg=self.max_obliquity_deg)
      T_hand_cam = self._hand_from_cam(detections)
      T_hand_cube = None if T_cam_cube is None else T_hand_cam @ T_cam_cube
      if T_hand_cube is None:
        self.stats["empty"] += 1
      else:
        self.stats["fused"] += 1

      self._seq += 1
      sample = PoseSample(
        T_hand_cube=T_hand_cube,
        device_time=frame.timestamp,
        host_time=frame.host_time,
        n_tags=info.get("n_used", 0),
        seq=self._seq,
        info=info,
      )
      with self._lock:
        self._latest = sample

  # -- consumer ---------------------------------------------------------------

  def latest(self) -> PoseSample | None:
    """Most recent completed estimate. Never blocks, never waits for a frame."""
    with self._lock:
      return self._latest

  def wait_for_first(self, timeout: float = 5.0) -> bool:
    """Block until the thread has published once, so the loop starts with a pose."""
    deadline = time.time() + timeout
    while time.time() < deadline:
      if self.latest() is not None:
        return True
      time.sleep(0.005)
    return False

  def summary(self) -> dict:
    d = np.asarray(self.detect_ms) if self.detect_ms else np.zeros(1)
    return {
      **self.stats,
      "detect_ms_mean": round(float(d.mean()), 2),
      "detect_ms_p95": round(float(np.percentile(d, 95)), 2),
    }
