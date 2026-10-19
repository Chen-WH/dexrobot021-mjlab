"""Everything that can be checked before a policy runs on the hand.

    python scripts/deploy/preflight.py checkpoints/spin_cube
    python scripts/deploy/preflight.py checkpoints/spin_cube --camera --hand

Run it first in a hardware session. Checks that need no hardware always run; the
camera and hand checks run only when asked for.

**`--hand` moves the hand**: `PyzlgHand.connect()` runs `reset_joints()` (the
feedback is NaN until the hand has been commanded once), which straightens every
finger. Take the object out first. Beyond that it only reads feedback; per-joint
motion checks are `verify_hand.py`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from dexrobot021_mjlab.paths import calib_dir  # noqa: E402

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


class Report:
  def __init__(self) -> None:
    self.rows: list[tuple[str, str, str]] = []

  def add(self, status: str, name: str, detail: str = "") -> None:
    self.rows.append((status, name, detail))

  def show(self) -> int:
    icon = {PASS: "  ok  ", FAIL: " FAIL ", SKIP: " skip ", WARN: " warn "}
    for status, name, detail in self.rows:
      print(f"[{icon[status]}] {name}")
      if detail:
        for line in detail.splitlines():
          print(f"           {line}")
    n_fail = sum(1 for s, _, _ in self.rows if s == FAIL)
    n_skip = sum(1 for s, _, _ in self.rows if s == SKIP)
    n_warn = sum(1 for s, _, _ in self.rows if s == WARN)
    print()
    if n_fail:
      print(f"NO-GO: {n_fail} check(s) failed.")
    elif n_skip:
      print(f"PARTIAL: everything checked passed, {n_skip} skipped.")
    else:
      print("GO: all checks passed." + (f"  ({n_warn} warning(s))" if n_warn else ""))
    return 1 if n_fail else 0


def check_deps(r: Report) -> None:
  for mod, why in (("numpy", "everything"), ("torch", "running the policy"),
                   ("cv2", "tag pose solve and Bayer demosaic"),
                   ("pupil_apriltags", "tag detection"),
                   ("gxipy", "the Daheng camera (vendor SDK)"),
                   ("pyzlg_dexhand", "the DexHand (vendor SDK)")):
    try:
      m = importlib.import_module(mod)
      r.add(PASS, f"import {mod}", getattr(m, "__version__", ""))
    except Exception as exc:  # noqa: BLE001
      r.add(FAIL, f"import {mod}", f"needed for {why}: {exc}")


def check_bundle(r: Report, bundle: pathlib.Path):
  from dexrobot021_mjlab.deploy.contract import PolicyContract

  try:
    contract = PolicyContract.load(bundle / "contract.json")
  except Exception as exc:  # noqa: BLE001
    r.add(FAIL, "bundle contract.json", str(exc))
    return None
  problems = contract.check()
  if problems:
    r.add(FAIL, "bundle contract self-check", "\n".join(problems))
    return None
  r.add(PASS, "bundle contract self-check",
        f"{contract.task_id}\n{contract.observation_dim} obs, {contract.n_joints} "
        f"joints, {contract.control_hz:.4g} Hz, object: {contract.object_kind}")

  try:
    from dexrobot021_mjlab.deploy.policy import Policy
    policy = Policy(bundle / "policy.pt", contract, device="cpu")
    out = policy(np.zeros(contract.observation_dim, np.float32))
    r.add(PASS, "policy.pt loads and runs", f"output {out.shape}, finite="
          f"{bool(np.isfinite(out).all())}")
  except Exception as exc:  # noqa: BLE001
    r.add(FAIL, "policy.pt loads and runs", str(exc))

  if not contract.cube_tag_size_measured:
    r.add(WARN, "object tag size",
          f"NOMINAL {contract.cube_tag_size_m * 1e3:.1f} mm. Measure the printed black "
          "square\nand re-export with --tag-size-m; a wrong size scales every distance.")
  else:
    r.add(PASS, "object tag size", f"measured {contract.cube_tag_size_m * 1e3:.2f} mm")
  return contract


def check_geometry(r: Report, contract) -> None:
  from dexrobot021_mjlab.deploy.cube import object_from_contract

  cube = object_from_contract(contract)
  problems = cube.verify_layout()
  label = f"{contract.object_kind} tag layout"
  if problems:
    r.add(FAIL, label, "\n".join(problems))
  else:
    r.add(PASS, label, f"{len(cube.T_cube_tag)} tags, ids "
          f"{min(cube.T_cube_tag)}-{max(cube.T_cube_tag)}")


def check_calibration(r: Report) -> None:
  intr = calib_dir() / "daheng_intrinsics.json"
  if intr.exists():
    d = json.loads(intr.read_text())
    r.add(PASS, "Daheng intrinsics", f"{intr}\nfx {d['fx']:.1f}, held-out "
          f"{d.get('rms_heldout_px')} px")
  else:
    r.add(FAIL, "Daheng intrinsics", f"{intr} missing -- run dahengcal.py")
  ext = calib_dir() / "extcal.json"
  if ext.exists():
    d = json.loads(ext.read_text())
    r.add(PASS, "wrist-tag calibration", f"{ext}\nwritten {d.get('written', '?')}, "
          f"{d.get('n_samples', '?')} samples, leave-one-out "
          f"{d.get('loo_deg_max', float('nan')):.2f} deg / "
          f"{d.get('loo_mm_max', float('nan')):.1f} mm")
  else:
    r.add(FAIL, "wrist-tag calibration", f"{ext} missing -- run extcal.py")


def check_joint_verification(r: Report, bundle: pathlib.Path) -> None:
  path = calib_dir() / "joint_verification.json"
  if not path.exists():
    r.add(WARN, "per-joint sign verification",
          "not run. The SIGN and ZERO of each axis come from the hand's wiring;\n"
          "a flipped sign does not error -- that finger just drives backwards.\n"
          "  -> python scripts/deploy/verify_hand.py " + str(bundle))
    return
  raw = json.loads(path.read_text())
  data = {k: v for k, v in raw.items() if not k.startswith("_")}
  bad = [k for k, v in data.items() if v != "ok"]
  if not raw.get("_complete", True):
    bad = bad or ["(record is from a partial run)"]
  if bad:
    r.add(FAIL, "per-joint sign verification", f"not ok: {bad}")
  else:
    r.add(PASS, "per-joint sign verification", f"{len(data)} joints confirmed")


def check_camera(r: Report, contract, args) -> None:
  if not args.camera:
    r.add(SKIP, "Daheng camera", "pass --camera to open it")
    return
  from dexrobot021_mjlab.deploy.camera import DahengSource, Intrinsics, TagDetector
  from dexrobot021_mjlab.deploy.cube import MIN_DECISION_MARGIN, object_from_contract

  intr_path = calib_dir() / "daheng_intrinsics.json"
  if not intr_path.exists():
    r.add(SKIP, "Daheng camera", "no intrinsics yet")
    return
  d = json.loads(intr_path.read_text())
  intr = Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"],
                    dist=np.asarray(d["dist"], float))
  source = DahengSource(serial=args.daheng_serial, intrinsics=intr)
  try:
    source.start()
  except Exception as exc:  # noqa: BLE001
    r.add(FAIL, "Daheng camera", str(exc))
    return
  try:
    rate = source.measure_rate(2.0)
    status = PASS if rate > 62.5 else FAIL
    r.add(status, "Daheng frame rate", f"{rate:.1f} fps (the loop runs at 62.5 Hz)")
    frame = None
    for _ in range(50):
      frame = source.read()
      if frame is not None:
        break
    if frame is None:
      r.add(FAIL, "Daheng frames", "no frame returned")
      return
    img = frame.image
    r.add(PASS, "Daheng frames", f"{img.shape} mean {img.mean():.0f}")
    cube = object_from_contract(contract)
    sizes = {t: cube.tag_size_m for t in cube.T_cube_tag}
    sizes[args.wrist_tag_id] = args.wrist_tag_size
    seen = TagDetector(source.intrinsics, sizes, quad_decimate=3.0).detect(frame)
    ids = {d.tag_id: d.decision_margin for d in seen}
    if args.wrist_tag_id in ids:
      r.add(PASS, "wrist tag visible",
            f"id {args.wrist_tag_id}, margin {ids[args.wrist_tag_id]:.0f}")
    else:
      r.add(FAIL, "wrist tag visible",
            f"id {args.wrist_tag_id} not decoded; the object pose is referenced to it")
    usable = [i for i, m in ids.items() if i in cube.T_cube_tag and m >= MIN_DECISION_MARGIN]
    if usable:
      r.add(PASS, "object tags visible", f"ids {sorted(usable)}")
    else:
      r.add(WARN, "object tags visible",
            "none usable. Expected if the object is not in view -- hold it where the "
            "hand\nwill hold it and re-run.")
  finally:
    source.stop()


def check_hand(r: Report, contract, can_device: int, allow: bool) -> None:
  if not allow:
    r.add(SKIP, "DexHand CAN", "pass --hand to open the CAN device (connect() "
                               "runs reset_joints(): fingers straighten)")
    return
  try:
    from dexrobot021_mjlab.deploy.hand import JOINT_MAP, PyzlgHand
  except Exception as exc:  # noqa: BLE001
    r.add(FAIL, "pyzlg_dexhand import", str(exc))
    return
  mapped = {m.sim_name for m in JOINT_MAP}
  missing = [n for n in contract.joint_names if n not in mapped]
  if missing:
    r.add(FAIL, "every contract joint has an SDK mapping", str(missing))
    return
  r.add(PASS, "every contract joint has an SDK mapping",
        f"{len(contract.joint_names)} joints")

  hand = PyzlgHand(contract.joint_names, device_index=can_device)
  try:
    hand.connect()
  except Exception as exc:  # noqa: BLE001
    r.add(FAIL, "DexHand CAN connect", str(exc))
    return
  try:
    faults = hand.board_errors()
    if faults:
      r.add(FAIL, "hand board faults",
            "\n".join(f"board {b}: {m}" for b, m in faults.items()) +
            "\nconnect() clears latched faults, so a fault still present is live.")
    else:
      r.add(PASS, "hand board faults", "none latched")
    pos, _ = hand.read()
    if not np.isfinite(pos).all():
      r.add(FAIL, "DexHand feedback", "still NaN after connect()")
      return
    lo = np.asarray(contract.joint_limits, float)[:, 0]
    hi = np.asarray(contract.joint_limits, float)[:, 1]
    outside = [contract.joint_names[i] for i in range(len(pos))
               if not (lo[i] - 0.15 <= pos[i] <= hi[i] + 0.15)]
    detail = "  ".join(f"{n.split('_', 2)[-1]}={np.degrees(v):+.0f}"
                       for n, v in zip(contract.joint_names, pos, strict=True))
    if outside:
      r.add(WARN, "DexHand feedback", detail +
            f"\noutside the model's limits: {outside}. Either the hand is in a pose "
            "the\nmodel cannot reach, or the sign/zero mapping is wrong -- run "
            "verify_hand.py.")
    else:
      r.add(PASS, "DexHand feedback", detail + "\n(inside the model's limits; this "
            "does NOT verify per-joint signs)")
  finally:
    hand.close()


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("bundle", type=pathlib.Path)
  ap.add_argument("--camera", action="store_true", help="open the Daheng and look for tags")
  ap.add_argument("--daheng-serial", default=None)
  ap.add_argument("--wrist-tag-id", type=int, default=10)
  ap.add_argument("--wrist-tag-size", type=float, default=0.080)
  ap.add_argument("--hand", action="store_true",
                  help="open the CAN device and read feedback; connect() runs "
                       "reset_joints(), so the fingers straighten -- object out first")
  ap.add_argument("--can-device", type=int, default=0)
  args = ap.parse_args()

  print(f"preflight: {args.bundle}\n")
  r = Report()
  check_deps(r)
  contract = check_bundle(r, args.bundle)
  if contract is None:
    return r.show()
  check_geometry(r, contract)
  check_calibration(r)
  check_joint_verification(r, args.bundle)
  check_camera(r, contract, args)
  check_hand(r, contract, args.can_device, args.hand)
  return r.show()


if __name__ == "__main__":
  raise SystemExit(main())
