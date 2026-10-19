"""Frame sources and AprilTag detection.

Two cameras are supported:

* `DahengSource` -- the external Daheng MER2-301-125U3C (IMX265, 2048x1536, global
  shutter) the object is tracked with. Its long focal length is what makes the
  2x2 tag tiles decodable at the working distance.
* `RealSenseSource` -- the palm-mounted D435i's left IR imager (848x480, 90 fps,
  projector off). Used as the calibration reference for the Daheng (`extcal.py`).
  IR rather than colour because the colour sensor has a rolling shutter. Its
  intrinsics are the depth stream's, so a pose solved in the IR frame is carried
  to `d435i_color_optical` (where the CAD extrinsic ends) with the device's own
  `T_color_ir`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .cube import TagObservation


@dataclass
class Intrinsics:
  fx: float
  fy: float
  cx: float
  cy: float
  width: int
  height: int
  #: OpenCV distortion coefficients `(k1, k2, p1, p2, k3, ...)`, or None for a
  #: pinhole camera. `pupil_apriltags` solves the tag pose with no distortion
  #: model, so when this is set `TagDetector` undistorts the corners and solves
  #: the pose itself.
  dist: np.ndarray | None = None

  def as_tuple(self) -> tuple[float, float, float, float]:
    return (self.fx, self.fy, self.cx, self.cy)

  def matrix(self) -> np.ndarray:
    return np.array([[self.fx, 0.0, self.cx],
                     [0.0, self.fy, self.cy],
                     [0.0, 0.0, 1.0]])


@dataclass
class Frame:
  """One image, with two clocks that must not be confused.

  `timestamp` is the device clock: monotonic and finely resolved, which is what
  differentiating a pose stream needs, but its epoch is the camera's.
  `host_time` is `time.perf_counter()` when the frame reached this process, so it
  is comparable with the loop's clock. Age measured against it excludes
  sensor-to-USB transport and is a lower bound on the true latency.
  """

  image: np.ndarray        # (H, W) uint8, single channel
  timestamp: float         # seconds, device clock -- for differentiation only
  host_time: float = 0.0   # seconds, perf_counter -- for age
  frame_number: int = 0


class FrameSource(Protocol):
  intrinsics: Intrinsics

  def start(self) -> None: ...
  def stop(self) -> None: ...
  def read(self) -> Frame | None: ...


class RealSenseSource:
  """Left IR stream off a D435i."""

  def __init__(self, serial: str | None = None, width: int = 848, height: int = 480,
               fps: int = 90, emitter: bool = False,
               exposure_us: float | None = 6000.0, gain: float = 64.0,
               settle_s: float = 0.5):
    self.serial = serial
    self.width, self.height, self.fps = width, height, fps
    self.emitter = emitter
    # Exposure is fixed, not auto: an exposure longer than the frame period
    # silently caps the frame rate, and auto-exposure retunes mid-episode.
    self.exposure_us = exposure_us
    self.gain = gain
    self.settle_s = settle_s
    self._pipeline = None
    self.intrinsics: Intrinsics | None = None
    #: `T_color_ir`: pose of the left IR optical frame in the colour optical
    #: frame, read from the device. The contract's extrinsic ends at colour.
    self.T_color_ir: np.ndarray | None = None
    #: Rotation carrying an accelerometer reading into the colour optical frame.
    self.T_accel_to_color: np.ndarray | None = None

  def start(self) -> None:
    import pyrealsense2 as rs

    config = rs.config()
    if self.serial:
      config.enable_device(self.serial)
    # Stream index 1 is the left imager; 2 is the right.
    config.enable_stream(rs.stream.infrared, 1, self.width, self.height,
                         rs.format.y8, self.fps)
    self._pipeline = rs.pipeline()
    profile = self._pipeline.start(config)

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    # The projector's dot pattern lands on the tags and destroys the decision margin.
    if depth_sensor.supports(rs.option.emitter_enabled):
      depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if self.emitter else 0.0)
    if self.exposure_us is not None:
      if depth_sensor.supports(rs.option.enable_auto_exposure):
        depth_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
      if depth_sensor.supports(rs.option.exposure):
        depth_sensor.set_option(rs.option.exposure, float(self.exposure_us))
      if depth_sensor.supports(rs.option.gain):
        depth_sensor.set_option(rs.option.gain, float(self.gain))
    self.exposure_budget_us = 1e6 / self.fps

    ir_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    i = ir_profile.get_intrinsics()
    self.intrinsics = Intrinsics(i.fx, i.fy, i.ppx, i.ppy, i.width, i.height)

    # Extrinsics are static device calibration, readable off a stream profile that
    # is not being streamed (only infrared is enabled).
    self.T_color_ir = self._static_extrinsic(rs, device, ir_profile, rs.stream.color)
    self.T_accel_to_color = self._static_extrinsic(
      rs, device, ir_profile, rs.stream.accel, via_color=True
    )

    # Let the sensor come up before anyone asks it for a frame; the first reads
    # otherwise return nothing and look like a dead camera.
    deadline = time.time() + self.settle_s
    while time.time() < deadline:
      self._pipeline.try_wait_for_frames(timeout_ms=100)

  @staticmethod
  def _static_extrinsic(rs, device, from_profile, to_stream, via_color=False):
    """`T` mapping points in `to_stream`'s frame into `from_profile`'s frame."""
    try:
      target = None
      for sensor in device.query_sensors():
        for p in sensor.get_stream_profiles():
          if p.stream_type() == to_stream:
            target = p
            break
        if target is not None:
          break
      if target is None:
        return None
      e = from_profile.get_extrinsics_to(target)
      T = np.eye(4)
      # librealsense stores rotation column-major.
      T[:3, :3] = np.asarray(e.rotation, float).reshape(3, 3).T
      T[:3, 3] = np.asarray(e.translation, float)
      return np.linalg.inv(T) if via_color else T
    except Exception:  # noqa: BLE001 - a missing extrinsic is not fatal
      return None

  def stop(self) -> None:
    if self._pipeline is not None:
      self._pipeline.stop()
      self._pipeline = None

  def measure_rate(self, seconds: float = 2.0) -> float:
    """Frames per second actually delivered."""
    if self._pipeline is None:
      raise RuntimeError("start() first")
    n, deadline = 0, time.time() + seconds
    while time.time() < deadline:
      if self.read() is not None:
        n += 1
    return n / seconds

  def read(self, block: bool = True) -> Frame | None:
    """Newest queued frame, or None.

    `block=False` returns only what is already queued and never waits; the pose
    thread drains with it (a drain built on blocking reads never empties).
    """
    if self._pipeline is None:
      raise RuntimeError("start() first")
    if block:
      ok, frames = self._pipeline.try_wait_for_frames(timeout_ms=50)
      if not ok:
        return None
    else:
      frames = self._pipeline.poll_for_frames()
      if not frames:
        return None
    ir = frames.get_infrared_frame(1)
    if not ir:
      return None
    return Frame(
      image=np.asanyarray(ir.get_data()),
      timestamp=ir.get_timestamp() / 1000.0,
      host_time=time.perf_counter(),
      frame_number=ir.get_frame_number(),
    )


class DahengSource:
  """Daheng MER2-301-125U3C over the vendor's `gxipy`.

  IMX265 sensor, 2048x1536 on a 3.45 um global-shutter pixel, so an 8 mm C-mount
  lens gives fx ~2320 px -- enough to decode 28 mm tiles past 70 deg of obliquity.
  At full resolution it delivers ~95 fps (capped by `DeviceLinkThroughputLimit`).
  `get_timestamp()` is in nanoseconds on the device clock.

  The camera reports no intrinsics (there is no lens until one is screwed on), so
  they must come from a checkerboard calibration (`dahengcal.py`);
  `intrinsics_source` records whether that happened.
  """

  #: IMX265 pixel pitch. `fx_px = focal_length_mm / (PIXEL_SIZE_M * 1e3)`.
  PIXEL_SIZE_M = 3.45e-6

  def __init__(self, serial: str | None = None,
               roi: tuple[int, int, int, int] | None = None,
               fps: float | None = None, exposure_us: float | None = 10000.0,
               gain: float = 20.0, intrinsics: Intrinsics | None = None,
               nominal_focal_mm: float | None = None,
               throughput_bps: int | None = None, settle_s: float = 0.3):
    self.serial = serial
    #: `(offset_x, offset_y, width, height)` in sensor pixels, or None for full frame.
    self.roi = roi
    self.fps = fps
    # Fixed exposure and gain, tuned on tile decode count for the rig's lighting;
    # 10000 us stays inside the frame period at ~95 fps. Gain did not change pose
    # jitter measurably. Re-tune if the lighting changes.
    self.exposure_us = exposure_us
    self.gain = gain
    self.throughput_bps = throughput_bps
    self.settle_s = settle_s
    self.nominal_focal_mm = nominal_focal_mm
    self.intrinsics = intrinsics
    #: "calibrated" when the caller supplied intrinsics, "nominal" when they were
    #: derived from an assumed focal length, "none" when neither.
    self.intrinsics_source = "calibrated" if intrinsics is not None else "none"
    #: The optical frame is the frame the detector solves in; nothing to compose.
    self.T_color_ir: np.ndarray | None = None
    self.T_accel_to_color: np.ndarray | None = None
    self._cam = None
    self._gx = None
    #: **The manager must outlive the device**: `gxipy.DeviceManager.__del__`
    #: un-initialises the library underneath an open device.
    self._manager = None

  def start(self) -> None:
    import gxipy as gx

    self._gx = gx
    self._manager = gx.DeviceManager()
    count, _ = self._manager.update_device_list()
    if not count:
      raise RuntimeError("no Daheng camera enumerated -- check the USB3 link")
    self._cam = (self._manager.open_device_by_sn(self.serial) if self.serial
                 else self._manager.open_device_by_index(1))
    cam = self._cam

    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
    if self.throughput_bps is not None:
      cam.DeviceLinkThroughputLimit.set(int(self.throughput_bps))
    # The ROI persists in the camera across sessions, so `roi=None` has to
    # restore the full sensor rather than leave it alone.
    ox, oy, w, h = self.roi if self.roi is not None else (
      0, 0, int(cam.WidthMax.get()), int(cam.HeightMax.get()))
    # Offsets must go to zero before a larger width is accepted.
    cam.OffsetX.set(0)
    cam.OffsetY.set(0)
    cam.Width.set(int(w))
    cam.Height.set(int(h))
    cam.OffsetX.set(int(ox))
    cam.OffsetY.set(int(oy))
    if self.exposure_us is not None:
      try:
        cam.ExposureAuto.set(gx.GxAutoEntry.OFF)
      except Exception:  # noqa: BLE001 - not every firmware exposes it
        pass
      cam.ExposureTime.set(float(self.exposure_us))
    try:
      cam.GainAuto.set(gx.GxAutoEntry.OFF)
      cam.Gain.set(float(self.gain))
    except Exception:  # noqa: BLE001
      pass
    if self.fps is not None:
      cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
      cam.AcquisitionFrameRate.set(float(self.fps))

    width, height = int(cam.Width.get()), int(cam.Height.get())
    self._resolve_intrinsics(width, height)

    cam.stream_on()
    deadline = time.time() + self.settle_s
    while time.time() < deadline:
      self._cam.data_stream[0].get_image(timeout=100)

  def _resolve_intrinsics(self, width: int, height: int) -> None:
    """Fill `intrinsics` for the streamed rectangle, or leave it None.

    An ROI does not change `fx`/`fy` but moves the principal point to
    `cx - offset_x` in ROI coordinates.
    """
    if self.intrinsics is not None:
      supplied = self.intrinsics
      ox, oy = (self.roi[0], self.roi[1]) if self.roi else (0, 0)
      # Only shift when the caller handed us full-sensor intrinsics.
      if supplied.width != width or supplied.height != height:
        self.intrinsics = Intrinsics(supplied.fx, supplied.fy,
                                     supplied.cx - ox, supplied.cy - oy,
                                     width, height, dist=supplied.dist)
      return
    if self.nominal_focal_mm is None:
      return
    f = self.nominal_focal_mm / (self.PIXEL_SIZE_M * 1e3)
    self.intrinsics = Intrinsics(f, f, width / 2.0, height / 2.0, width, height)
    self.intrinsics_source = "nominal"

  def stop(self) -> None:
    if self._cam is not None:
      try:
        self._cam.stream_off()
      finally:
        self._cam.close_device()
        self._cam = None
        # Released only after the device, for the reason in `_manager`.
        self._manager = None

  def measure_rate(self, seconds: float = 2.0) -> float:
    """Frames per second actually delivered."""
    if self._cam is None:
      raise RuntimeError("start() first")
    n, deadline = 0, time.time() + seconds
    while time.time() < deadline:
      if self.read() is not None:
        n += 1
    return n / seconds

  def read(self, block: bool = True) -> Frame | None:
    """Next frame, demosaiced to grey, or None.

    `gxipy` has no true non-blocking poll, so `block=False` waits 1 ms. The sensor
    is a Bayer part; raw Bayer lays a checkerboard over every edge the corner
    refinement uses, so it is demosaiced here (~0.5 ms at full resolution).
    """
    if self._cam is None:
      raise RuntimeError("start() first")
    import cv2

    raw = self._cam.data_stream[0].get_image(timeout=50 if block else 1)
    if raw is None:
      return None
    plane = raw.get_numpy_array()
    if plane is None:
      return None
    return Frame(
      image=cv2.cvtColor(plane, cv2.COLOR_BayerRG2GRAY),
      timestamp=raw.get_timestamp() / 1e9,
      host_time=time.perf_counter(),
      frame_number=raw.get_frame_id(),
    )


class TagDetector:
  """`pupil_apriltags` wrapper producing `TagObservation`s.

  `pupil_apriltags` reports the tag pose with `+z` pointing into the tag, which is
  this package's convention (`z_points_outward=False`).
  """

  def __init__(self, intrinsics: Intrinsics, tag_size_m: float | dict[int, float],
               family: str = "tag36h11", nthreads: int = 4,
               quad_decimate: float = 1.5):
    from pupil_apriltags import Detector

    self.intrinsics = intrinsics
    #: When `tag_size_m` is a mapping, it is `{tag id: printed black-square side}`
    #: and an id absent from it is DROPPED: a tag solved with the wrong size yields
    #: a translation wrong by exactly the ratio that looks entirely healthy.
    self._size_map = ({int(k): float(v) for k, v in tag_size_m.items()}
                      if isinstance(tag_size_m, dict) else None)
    #: The scalar the detector's own solve uses. With a mapping, the most common
    #: size stands in for reporting purposes.
    self.tag_size_m = (float(tag_size_m) if self._size_map is None
                       else float(max(set(self._size_map.values()),
                                      key=list(self._size_map.values()).count)))
    # `quad_decimate` trades corner localisation for speed. Measured against
    # rendered ground truth, pose accuracy is flat up to 2.0 and nothing solves at
    # 2.5; the Daheng path runs at 3.0 on its much larger image.
    self._detector = Detector(families=family, nthreads=nthreads,
                              quad_decimate=quad_decimate, refine_edges=True)

  #: Tag corner positions in the tag frame, in the order `cv2.SOLVEPNP_IPPE_SQUARE`
  #: requires and `pupil_apriltags` reports: `(-h, +h), (+h, +h), (+h, -h), (-h, -h)`
  #: for half-side `h`. Checked against the detector's own pose solve.
  _CORNER_SIGNS = ((-1, +1), (+1, +1), (+1, -1), (-1, -1))

  def _object_points(self, size_m: float) -> np.ndarray:
    h = size_m / 2.0
    return np.array([[sx * h, sy * h, 0.0] for sx, sy in self._CORNER_SIGNS], float)

  def size_for(self, tag_id: int) -> float | None:
    """Printed black-square side for this id, or None to ignore the detection.

    One image carries two tag sizes (the object's tags and the 80 mm wrist tag),
    and `pupil_apriltags` takes a single `tag_size` per call, so with a size map
    each tag is solved here with its own size.
    """
    if self._size_map is None:
      return self.tag_size_m
    return self._size_map.get(int(tag_id))

  def detect(self, frame: Frame) -> list[TagObservation]:
    image = frame.image
    if image.ndim == 3:
      image = image[..., 0]
    # The built-in solve carries one tag size and no distortion, so it is used only
    # when neither applies. Quad detection runs once either way; the per-tag PnP
    # that replaces the built-in solve is microseconds.
    own_solve = self.intrinsics.dist is not None or self._size_map is not None
    results = self._detector.detect(
      image,
      estimate_tag_pose=not own_solve,
      camera_params=self.intrinsics.as_tuple(),
      tag_size=self.tag_size_m,
    )
    if not own_solve:
      return [
        TagObservation(
          tag_id=int(r.tag_id),
          R=np.asarray(r.pose_R, float),
          t=np.asarray(r.pose_t, float).ravel(),
          decision_margin=float(r.decision_margin),
          z_points_outward=False,
        )
        for r in results
      ]

    import cv2

    K = self.intrinsics.matrix()
    dist = (np.asarray(self.intrinsics.dist, float).ravel()
            if self.intrinsics.dist is not None else None)
    out = []
    for r in results:
      size = self.size_for(int(r.tag_id))
      if size is None:
        continue
      objp = self._object_points(size)
      corners = np.asarray(r.corners, float).reshape(-1, 1, 2)
      # Undistort the four corners (not the whole image); `P=K` maps back to
      # pixels so the solve runs on ideal pinhole coordinates.
      ideal = (cv2.undistortPoints(corners, K, dist, P=K).reshape(-1, 2)
               if dist is not None else corners.reshape(-1, 2))
      ok, rvec, tvec = cv2.solvePnP(objp, ideal, K, None,
                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
      if not ok:
        continue
      R, _ = cv2.Rodrigues(rvec)
      out.append(TagObservation(
        tag_id=int(r.tag_id),
        R=R,
        t=np.asarray(tvec, float).ravel(),
        decision_margin=float(r.decision_margin),
        z_points_outward=False,
      ))
    return out
