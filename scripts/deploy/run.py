"""Run a deployment bundle on the hand.

    # dry run: no hand, no camera (loop timing, encoding, safety paths)
    python scripts/deploy/run.py checkpoints/spin_cube --mock

    # camera only, the hand is not moved
    python scripts/deploy/run.py checkpoints/spin_cube --mock-hand

    # the real thing
    python scripts/deploy/run.py checkpoints/spin_cube --episodes 3 --trace

The object is tracked by the external Daheng camera, referenced to the wrist tag
(id 10) in the same image, so no camera extrinsic enters the chain; the wrist tag's
pose in the hand frame comes from `extcal.py` (`deploy/calib/extcal.json`).

Each episode:
  1. the hand opens to the bundle's placement pose (`contract.place_joint_pos`),
     empty;
  2. the operator lays the object onto the open fingers and lets go; once it has
     been seen still for `--auto-start-s` and passes the placement gates (height,
     position, and yaw for the cube / axis for the cylinder), the hand closes to
     the episode-start grasp under the stall guard;
  3. the policy runs until `--duration` or a safety stop;
  4. the operator takes the object out; the next episode starts from an empty hand.

Every run writes a JSON record (and with `--trace` a per-step `.npz`) next to the
bundle.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from dexrobot021_mjlab.deploy.camera import Intrinsics, TagDetector  # noqa: E402
from dexrobot021_mjlab.deploy.contract import PolicyContract  # noqa: E402
from dexrobot021_mjlab.deploy.cube import object_from_contract  # noqa: E402
from dexrobot021_mjlab.deploy.hand import MockHand, PyzlgHand  # noqa: E402
from dexrobot021_mjlab.deploy.loop import ControlLoop, SafetyLimits  # noqa: E402
from dexrobot021_mjlab.deploy.pipeline import ThreadedPoseSource  # noqa: E402
from dexrobot021_mjlab.deploy.policy import Policy  # noqa: E402
from dexrobot021_mjlab.deploy.tracker import CubeTracker  # noqa: E402
from dexrobot021_mjlab.paths import calib_dir  # noqa: E402

#: How far the placed cube may be turned about the palm normal from the grasp's yaw.
YAW_TOL_DEG = 10.0
#: How far the placed object may sit from the grasp position, mm. The reference is
#: where the object sits in the CLOSED grasp while it is measured on the OPEN hand,
#: so this is loose; the stall guard on the closing ramp is what protects the boards.
POS_TOL_MM = 35.0
#: How far the placed cylinder's axis may point from the grasp's, degrees (its roll
#: and which end faces the thumb are randomised in training, so only the direction
#: is gated, up to a flip).
AXIS_TOL_DEG = 15.0


class MockCamera:
  """A camera that sees nothing. The loop then runs on held/lost poses only."""

  def __init__(self, intrinsics):
    self.intrinsics = intrinsics

  def start(self):
    pass

  def stop(self):
    pass

  def read(self, block: bool = True):
    if block:
      time.sleep(0.005)
    return None


class _NullDetector:
  def detect(self, frame):
    return []


def _rejection_counts(info: dict | None) -> dict:
  """Object tags the fuser saw and discarded, split by why."""
  reasons = (info or {}).get("tags_rejected", {}) or {}
  values = [str(v) for v in reasons.values()]
  return {
    "cube_rejected_oblique": sum(v.startswith("obliquity") for v in values),
    "cube_rejected_margin": sum(v.startswith("decision margin") for v in values),
  }


def _daheng_intrinsics(args) -> Intrinsics:
  """Calibrated intrinsics for the Daheng (`dahengcal.py`); refuses nominal ones."""
  path = pathlib.Path(args.daheng_intrinsics)
  if not path.is_file():
    raise SystemExit(f"no Daheng intrinsics at {path} -- run "
                     f"scripts/deploy/dahengcal.py --square-mm <measured> first")
  d = json.loads(path.read_text())
  print(f"Daheng intrinsics from {path}: fx {d['fx']:.1f}, "
        f"held-out {d.get('rms_heldout_px')} px, model {d.get('dist_model')}")
  return Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"],
                    dist=np.asarray(d["dist"], float))


def _wrist_in_hand(args) -> np.ndarray:
  """`T_hand_wristtag`: where the wrist tag sits in the hand frame, from `extcal.json`.

  `extcal.py` stores the external camera's pose in the hand frame and the wrist
  tag's pose in that camera, solved with the object's tag size; the translation is
  rescaled here for the wrist tag's own size (the rotation is unaffected).
  """
  path = pathlib.Path(args.wrist_ref) if args.wrist_ref else calib_dir() / "extcal.json"
  if not path.is_file():
    raise SystemExit(f"no wrist-tag calibration at {path} -- run "
                     f"scripts/deploy/extcal.py checkpoints/spin_cube, or pass --wrist-ref")
  cal = json.loads(path.read_text())
  T_hand_ext = np.asarray(cal["T_hand_ext"], float)
  T_ext_wrist = np.asarray(cal["T_ext_wristtag"][str(args.wrist_tag_id)], float).copy()
  scale = args.wrist_tag_size / cal["solved_tag_size_m"]
  T_ext_wrist[:3, 3] *= scale
  T = T_hand_ext @ T_ext_wrist
  print(f"wrist tag {args.wrist_tag_id} from {path}: "
        f"{(T[:3, 3] * 1e3).round(1).tolist()} mm in the hand frame")
  return T


def build(args, contract: PolicyContract):
  cube = object_from_contract(contract)
  problems = cube.verify_layout()
  if problems:
    raise SystemExit(f"{contract.object_kind} layout self-check failed:\n  "
                     + "\n  ".join(problems))
  print(f"{contract.object_kind}: {len(cube.T_cube_tag)} tags, "
        f"{cube.tag_size_m * 1e3:g} mm black square, ids "
        f"{min(cube.T_cube_tag)}-{max(cube.T_cube_tag)}")

  if args.mock or args.mock_camera:
    source = MockCamera(Intrinsics(2300.0, 2300.0, 1024.0, 768.0, 2048, 1536))
    detector = None
  else:
    from dexrobot021_mjlab.deploy.camera import DahengSource

    source = DahengSource(serial=args.daheng_serial, intrinsics=_daheng_intrinsics(args))
    source.start()
    # Two tag sizes in one image (the object's tags and the wrist tag), so the
    # detector takes a per-id map.
    sizes = {t: cube.tag_size_m for t in cube.T_cube_tag}
    sizes[args.wrist_tag_id] = args.wrist_tag_size
    detector = TagDetector(source.intrinsics, sizes, quad_decimate=args.daheng_decimate)

  if args.mock or args.mock_hand:
    hand = MockHand(contract.joint_names, np.asarray(contract.reset_joint_pos),
                    contract.control_dt)
  else:
    hand = PyzlgHand(contract.joint_names, device_index=args.can_device)
  hand.connect()
  if not args.no_stall_guard:
    # Guards the placement/removal holds and the approach ramps; a pass-through
    # while the policy runs.
    from dexrobot021_mjlab.deploy.stallguard import StallGuardHand
    hand = StallGuardHand(hand, contract.n_joints)
  else:
    print("!! stall guard OFF -- a finger the object displaces during placement "
          "can latch its board")

  policy = Policy(pathlib.Path(args.bundle) / "policy.pt", contract, device=args.device)
  limits = SafetyLimits(max_duration_s=args.duration,
                        max_joint_rate=args.max_joint_rate,
                        observe_joint_rate_only=not args.stop_on_joint_rate)

  # The pose source already returns wrist-frame poses (via the wrist tag), so the
  # tracker applies no extrinsic of its own.
  tracker = CubeTracker(cube, np.eye(4), contract.control_dt,
                        velocity_window=args.velocity_window)
  poses = ThreadedPoseSource(source, _NullDetector() if detector is None else detector,
                             cube, np.eye(4))
  if detector is not None:
    poses.wrist_ref = (args.wrist_tag_id, _wrist_in_hand(args))
  poses.start()
  if detector is not None and not poses.wait_for_first(timeout=5.0):
    raise SystemExit("the pose thread published nothing in 5 s -- camera stalled?")

  place_hook = None if detector is None else _settled_hook(poses, args, contract)
  loop = ControlLoop(contract, policy, hand, poses, tracker, limits,
                     place_hook=place_hook, place_pose=contract.place_joint_pos,
                     settle_s=args.settle_s, start_tol_deg=args.start_tol_deg)
  return loop, hand, source, tracker, poses


def _placed_axis_error_deg(T_hand_object: np.ndarray, axis_wrist) -> float:
  """Angle between the placed object's own axis (local z) and the grasp's, folded
  over the end-for-end flip."""
  axis = np.asarray(T_hand_object)[:3, :3] @ np.array([0.0, 0.0, 1.0])
  cos = abs(float(axis @ np.asarray(axis_wrist, float)))
  return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _placed_yaw_error_deg(T_hand_cube: np.ndarray) -> float:
  """The placed cube's yaw about the palm normal (hand +x), folded over the cube's
  90 deg symmetry to (-45, 45]."""
  R = np.asarray(T_hand_cube)[:3, :3]
  k = int(np.argmin(np.abs(R[0, :])))           # the cube axis lying most in the palm plane
  yaw = np.degrees(np.arctan2(R[2, k], R[1, k]))
  return float((yaw + 45.0) % 90.0 - 45.0)


def _settled_hook(poses, args, contract: PolicyContract, timeout_s: float = 120.0,
                  still_mm: float = 3.0):
  """Block until the object has been placed, released and passes the gates.

  The object must be seen, fresh and not moving, for `args.auto_start_s`: the
  release is what `still_mm` waits for, so an object still in the operator's
  fingers does not start the clock. Then:

  * height: resting more than `--place-z-tol-mm` above the grasp height means it
    lies on the fingertips, and the closing ramp would stop the thumb short;
  * position: within `POS_TOL_MM` of the grasp position;
  * orientation: the cylinder's axis within `AXIS_TOL_DEG`, the cube's yaw within
    `YAW_TOL_DEG`.

  A failed gate prints what to fix and keeps waiting.
  """
  pos_wrist = np.asarray(contract.object_pos_wrist, float)
  axis_wrist = contract.object_axis_wrist
  stable_s = args.auto_start_s

  def hook() -> None:
    print("\n  hand is waiting OPEN, EMPTY. Lay the object onto the open fingers and "
          f"let go; the episode starts after it has been still for {stable_s:g} s.")
    t0 = time.perf_counter()
    since, anchor, last_seen = None, None, None
    while True:
      now = time.perf_counter()
      if now - t0 > timeout_s:
        raise RuntimeError(f"no settled object within {timeout_s:g} s")
      sample = poses.latest()
      fresh = (sample is not None and sample.T_hand_cube is not None
               and now - sample.host_time < 0.15)
      if fresh:
        pos = np.asarray(sample.T_hand_cube)[:3, 3]
        if since is None or np.linalg.norm(pos - anchor) * 1e3 > still_mm:
          since, anchor = now, pos
        last_seen = now
        if now - since >= stable_s:
          d_mm = float(np.linalg.norm(pos - pos_wrist) * 1e3)
          z_mm = float((pos[2] - pos_wrist[2]) * 1e3)
          if z_mm > args.place_z_tol_mm:
            print(f"  !! placed {z_mm:.0f} mm ABOVE the grasp height "
                  f"(> {args.place_z_tol_mm:g}) -- NOT starting; it is resting on the "
                  f"fingertips. Lay it deeper into the palm, toward z = "
                  f"{pos_wrist[2] * 1e3:.0f} mm")
            since = now
            continue
          if d_mm > POS_TOL_MM:
            print(f"  !! placed {d_mm:.0f} mm from where the grasp expects it "
                  f"(> {POS_TOL_MM:g}) -- NOT starting. Move it toward "
                  f"{np.round(pos_wrist * 1e3, 0)} mm (hand frame)")
            since = now
            continue
          if axis_wrist is not None:
            off = _placed_axis_error_deg(sample.T_hand_cube, axis_wrist)
            what, tol = f"axis is {off:.0f} deg", AXIS_TOL_DEG
          else:
            off = _placed_yaw_error_deg(sample.T_hand_cube)
            what, tol = f"yaw is {off:+.0f} deg", YAW_TOL_DEG
          if abs(off) > tol:
            print(f"  !! placed {what} from the grasp's (> {tol:g}) "
                  "-- NOT starting; lift it and place it again")
            since = now
            continue
          print(f"  object settled at {np.round(anchor * 1e3, 1)} mm (hand frame), "
                f"{what} from the grasp's, {d_mm:.0f} mm from its position -- closing")
          args.placed = {
            "placed_pos_mm": [round(float(v), 1) for v in pos * 1e3],
            "placed_above_grasp_mm": round(z_mm, 1),
            "placed_off_deg": round(float(off), 1),
            "placed_off_grasp_mm": round(d_mm, 1),
          }
          return
      elif last_seen is not None and now - last_seen > 0.3:
        since, anchor = None, None
      time.sleep(0.02)

  return hook


def _wait_until_empty(poses, clear_s: float = 1.5, timeout_s: float = 180.0) -> None:
  """Block until no object has been seen for `clear_s`: every episode starts empty."""
  print("\n  take the object OUT of the hand; the next episode starts from an empty hand.")
  t0 = time.perf_counter()
  last_seen = time.perf_counter()
  while True:
    now = time.perf_counter()
    if now - t0 > timeout_s:
      raise RuntimeError(f"object still in view after {timeout_s:g} s")
    sample = poses.latest()
    if sample is not None and sample.T_hand_cube is not None and now - sample.host_time < 0.15:
      last_seen = now
    if now - last_seen >= clear_s:
      print("  hand is empty")
      return
    time.sleep(0.02)


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("bundle", type=pathlib.Path)
  ap.add_argument("--episodes", type=int, default=1)
  ap.add_argument("--duration", type=float, default=20.0, help="episode length, s")
  ap.add_argument("--auto-start-s", type=float, default=2.0,
                  help="start once the placed object has been seen still this long")
  ap.add_argument("--place-z-tol-mm", type=float, default=10.0,
                  help="refuse to start while the placed object rests more than this far "
                       "ABOVE the grasp height (hand-frame z)")
  ap.add_argument("--start-tol-deg", type=float, default=10.0,
                  help="close again, then refuse to start, if any joint is further than "
                       "this from the episode-start grasp after settling")
  ap.add_argument("--settle-s", type=float, default=0.3,
                  help="keep commanding the start pose this long before step 0")
  ap.add_argument("--daheng-serial", default=None, help="default: the first camera found")
  ap.add_argument("--daheng-intrinsics", default=str(calib_dir() / "daheng_intrinsics.json"))
  ap.add_argument("--daheng-decimate", type=float, default=3.0,
                  help="AprilTag quad_decimate on the full-resolution Daheng image")
  ap.add_argument("--wrist-ref", default=None,
                  help="extcal.json with T_hand_ext and T_ext_wristtag; "
                       "default deploy/calib/extcal.json")
  ap.add_argument("--wrist-tag-id", type=int, default=10)
  ap.add_argument("--wrist-tag-size", type=float, default=0.080,
                  help="MEASURED black-square side of the wrist tag, m")
  ap.add_argument("--can-device", type=int, default=0)
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--velocity-window", type=int, default=3)
  ap.add_argument("--max-joint-rate", type=float, default=9.0,
                  help="soft limit on the MEASURED joint rate, rad/s")
  ap.add_argument("--stop-on-joint-rate", action="store_true",
                  help="end the episode on the soft joint-rate limit instead of counting "
                       "excursions (the hard ceiling and board faults always stop it)")
  ap.add_argument("--no-stall-guard", action="store_true",
                  help="do not guard the placement/removal holds and approach ramps")
  ap.add_argument("--mock", action="store_true", help="no hand, no camera")
  ap.add_argument("--mock-hand", action="store_true")
  ap.add_argument("--mock-camera", action="store_true")
  ap.add_argument("--out", type=pathlib.Path, default=None)
  ap.add_argument("--trace", action="store_true",
                  help="save per-step joint positions, velocities, targets, motor current "
                       "and object pose next to the run record")
  args = ap.parse_args()
  args.placed = {}

  contract = PolicyContract.load(args.bundle / "contract.json")
  problems = contract.check()
  if problems:
    print("contract failed its checks:", *problems, sep="\n  ")
    return 1
  if contract.place_joint_pos is None or contract.object_pos_wrist is None:
    print("the bundle has no placement pose; re-export it with export_bundle.py")
    return 1
  if not contract.cube_tag_size_measured:
    print(f"WARNING: the bundle carries a NOMINAL tag size "
          f"({contract.cube_tag_size_m * 1e3:.1f} mm); a wrong size scales every "
          "recovered distance. Measure the printed black square and re-export with "
          "--tag-size-m.\n")
  if contract.object_kind == "cylinder":
    print("lay the cylinder ACROSS the open fingers, roughly along the thumb-to-little "
          "direction; its roll and which end faces the thumb do not matter, the axis "
          f"direction does (gate {AXIS_TOL_DEG:g} deg)\n")
  else:
    print("lay the cube on the open fingers with a face square to the palm "
          f"(yaw gate {YAW_TOL_DEG:g} deg); the hand then closes onto it\n")

  loop, hand, source, tracker, poses = build(args, contract)

  records = []
  aborted: dict | None = None
  trace: list[dict] = []
  if args.trace:
    def on_step(step, record):
      trace.append({
        "episode": len(records),
        "step": step,
        "joint_pos": np.asarray(record["joint_pos"], float).copy(),
        "joint_vel": np.asarray(record["joint_vel"], float).copy(),
        "target": np.asarray(record["target"], float).copy(),
        # Motor-side current (mA) and speed (rpm) from the feedback `read()`
        # already fetched.
        **{f"motor_{k}": np.asarray(v, float).copy()
           for k, v in (getattr(hand, "read_motor", dict)() or {}).items()
           if k in ("current", "velocity")},
        "cube_valid": bool(record["cube"].valid),
        "cube_pos": np.asarray(record["cube"].pos, float).copy(),
        "cube_quat": np.asarray(record["cube"].quat, float).copy(),
        "cube_age": int(record["cube"].age_steps),
        "cube_n_tags": int(record["cube"].n_tags),
        **_rejection_counts(getattr(tracker, "last_info", None)),
      })
    loop.on_step = on_step

  mock_camera = args.mock or args.mock_camera
  try:
    for episode in range(args.episodes):
      print(f"--- episode {episode + 1}/{args.episodes} "
            f"({contract.control_hz:.4g} Hz, {args.duration:g} s max) ---")
      if not mock_camera:
        # Keep commanding (through the stall guard) while the operator takes the
        # object out, so a finger it drags on the way out is clamped, not latched.
        pos, _ = hand.read()
        hold = np.where(np.isfinite(pos), pos, np.asarray(contract.reset_joint_pos, float))
        loop._hold_while(lambda: _wait_until_empty(poses), hold,
                         "waiting for the hand to be emptied")
      args.placed = {}
      stats = loop.run()
      summary = stats.summary()
      summary["episode"] = episode
      if args.placed:
        summary["placed"] = dict(args.placed)
      summary["tracker"] = dict(tracker.stats)
      summary["vision"] = poses.summary()
      if hasattr(hand, "clamped_steps"):
        summary["stall_guard"] = hand.summary(contract.joint_names)
      records.append(summary)
      for key, value in summary.items():
        print(f"  {key:22s} {value}")
      if summary["vision_age_ms_p95"] and summary["vision_age_ms_p95"] > 48.0:
        print("  !! vision age p95 above 48 ms: outside the regime the policy was "
              "trained for.")
      print()
  except KeyboardInterrupt:
    print("\ninterrupted")
    aborted = {"kind": "KeyboardInterrupt", "after_episodes": len(records)}
  except Exception as exc:  # noqa: BLE001 - the record outlives the traceback
    # Hardware episodes are not repeatable: write what was recorded before the
    # fault (e.g. a board latch) rather than losing it with the exception.
    traceback.print_exc()
    aborted = {"kind": type(exc).__name__, "detail": str(exc),
               "after_episodes": len(records)}
  finally:
    poses.stop()
    hand.close()
    source.stop()

  out = args.out or (args.bundle / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json")
  out.write_text(json.dumps({
    "task": contract.task_id,
    "checkpoint": contract.checkpoint,
    "mock": {"hand": bool(args.mock or args.mock_hand), "camera": bool(mock_camera)},
    "limits": {"max_obliquity_deg": poses.max_obliquity_deg,
               "max_joint_rate": loop.limits.max_joint_rate,
               "observe_joint_rate_only": loop.limits.observe_joint_rate_only,
               "max_lost_steps": loop.limits.max_lost_steps,
               "max_duration_s": loop.limits.max_duration_s,
               "stall_guard": not args.no_stall_guard},
    #: Non-null when the session ended on a fault rather than on its episode
    #: count. The episodes before it are still valid data.
    "aborted": aborted,
    "episodes": records,
  }, indent=2) + "\n")
  print(f"wrote {out}")

  if args.trace and trace:
    npz = out.with_suffix(".npz")
    np.savez_compressed(
      npz,
      episode=np.array([t["episode"] for t in trace]),
      step=np.array([t["step"] for t in trace]),
      joint_pos=np.stack([t["joint_pos"] for t in trace]),
      joint_vel=np.stack([t["joint_vel"] for t in trace]),
      target=np.stack([t["target"] for t in trace]),
      **{k: np.stack([t[k] for t in trace])
         for k in ("motor_current", "motor_velocity") if k in trace[0]},
      cube_valid=np.array([t["cube_valid"] for t in trace]),
      cube_pos=np.stack([t["cube_pos"] for t in trace]),
      cube_quat=np.stack([t["cube_quat"] for t in trace]),
      cube_age=np.array([t["cube_age"] for t in trace]),
      cube_n_tags=np.array([t["cube_n_tags"] for t in trace]),
      cube_rejected_oblique=np.array([t["cube_rejected_oblique"] for t in trace]),
      cube_rejected_margin=np.array([t["cube_rejected_margin"] for t in trace]),
      joint_names=np.array(contract.joint_names),
      reset_joint_pos=np.asarray(contract.reset_joint_pos, float),
    )
    print(f"wrote {npz}  ({len(trace)} steps)")
  if aborted is not None and aborted["kind"] != "KeyboardInterrupt":
    print(f"\n!! session ended on {aborted['kind']} after "
          f"{aborted['after_episodes']} episode(s) -- the record above is complete "
          f"up to that point")
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
