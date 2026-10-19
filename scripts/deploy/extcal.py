"""Calibrate the external Daheng camera and the wrist tag against the hand frame.

    python scripts/deploy/extcal.py checkpoints/spin_cube

Uses the tiled 70 mm cube as the only thing both the palm D435i and the Daheng can
see. Run it with the CUBE bundle even if you will run the cylinder: the product,
the wrist tag's pose in the hand frame, is the same for both objects.

The arm does not move, so `T_hand_ext` is a constant and this is a plain
rigid-transform calibration. A cube held where both cameras see it gives one
equation per pose:

    T_hand_ext = (T_hand_cam @ T_palm_cube) @ inv(T_ext_cube)

where `T_hand_cam` is the palm camera's CAD extrinsic (`extrinsic.py`). What
`run.py` uses at runtime is the wrist tag (id 10) in the hand frame,
`T_hand_ext @ T_ext_wristtag`, so the wrist tag has to be in the Daheng's view for
every sample; it is solved at its own printed size.

Two failure modes this guards against, because both look like success:

*   **A moving cube.** The cameras are not synchronised, so a sample taken while
    the cube moves pairs two different poses. Samples are accepted only when both
    cameras agree the cube is still.
*   **All samples at one cube pose.** The rotation of `T_hand_ext` is only
    observable across DIFFERENT cube orientations. Samples must be separated by
    `--min-separation-deg`.

A single small planar tag can take the flipped PnP solution and still look still,
so each camera must fuse at least `--min-tags` tiles, and a pairwise consensus
rejects outlier samples before averaging.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from dexrobot021_mjlab.deploy.camera import (  # noqa: E402
  DahengSource,
  Intrinsics,
  RealSenseSource,
  TagDetector,
)
from dexrobot021_mjlab.deploy.contract import PolicyContract  # noqa: E402
from dexrobot021_mjlab.deploy.cube import object_from_contract  # noqa: E402
from dexrobot021_mjlab.deploy.geometry import (  # noqa: E402
  mat_to_quat,
  mean_pose,
  pose_delta,
  quat_angle,
  quat_conjugate,
  quat_mul,
  se3,
  se3_inv,
)
from dexrobot021_mjlab.paths import calib_dir  # noqa: E402

#: Tiles decode reliably far past a single full-face tag.
OBLIQUITY_DEG = 75.0


def read_pose(source, detector, cube):
  """One (T_cam_cube, tags used, other detections) from one frame, or (None, ...)."""
  frame = source.read()
  if frame is None:
    return None, [], []
  dets = detector.detect(frame)
  known = [d for d in dets if d.tag_id in cube.T_cube_tag]
  other = [d for d in dets if d.tag_id not in cube.T_cube_tag]
  T, info = cube.pose_from_detections(known, max_obliquity_deg=OBLIQUITY_DEG)
  return T, info.get("tags_used", []), other


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("bundle", type=pathlib.Path, help="a CUBE bundle")
  ap.add_argument("--palm-serial", default=None, help="the palm D435i's serial")
  ap.add_argument("--daheng-serial", default=None)
  ap.add_argument("--daheng-intrinsics", default=str(calib_dir() / "daheng_intrinsics.json"))
  ap.add_argument("--samples", type=int, default=12, help="distinct cube poses to collect")
  ap.add_argument("--min-separation-deg", type=float, default=15.0,
                  help="a new sample must differ from every accepted one by this much")
  ap.add_argument("--still-deg", type=float, default=0.8)
  ap.add_argument("--still-mm", type=float, default=2.0)
  ap.add_argument("--fps", type=int, default=30, help="palm camera rate")
  ap.add_argument("--timeout", type=float, default=240.0)
  ap.add_argument("--wrist-tag-id", type=int, default=10)
  ap.add_argument("--wrist-tag-size", type=float, default=0.080,
                  help="MEASURED black-square side of the wrist tag, m")
  ap.add_argument("--min-tags", type=int, default=3,
                  help="tiles each camera must fuse for a sample to count")
  ap.add_argument("--consensus-deg", type=float, default=2.0)
  ap.add_argument("--consensus-mm", type=float, default=15.0)
  ap.add_argument("--min-inliers", type=int, default=8)
  ap.add_argument("--out", type=pathlib.Path, default=None)
  args = ap.parse_args()

  contract = PolicyContract.load(args.bundle / "contract.json")
  if contract.object_kind != "cube":
    raise SystemExit("calibrate with the cube bundle: the palm camera sees the "
                     "cylinder's end tags edge-on")
  cube = object_from_contract(contract)
  problems = cube.verify_layout()
  if problems:
    raise SystemExit("cube layout self-check failed:\n  " + "\n  ".join(problems))

  d = json.loads(pathlib.Path(args.daheng_intrinsics).read_text())
  palm = RealSenseSource(serial=args.palm_serial, fps=args.fps)
  ext = DahengSource(serial=args.daheng_serial,
                     intrinsics=Intrinsics(d["fx"], d["fy"], d["cx"], d["cy"],
                                           d["width"], d["height"],
                                           dist=np.asarray(d["dist"], float)))
  palm.start()
  ext.start()
  try:
    T_hand_cam = contract.T_hand_cam_array()
    if palm.T_color_ir is not None:
      # The contract's extrinsic ends at the colour frame; the detector solves in IR.
      T_hand_cam = T_hand_cam @ palm.T_color_ir
    sizes = {t: cube.tag_size_m for t in cube.T_cube_tag}
    palm_det = TagDetector(palm.intrinsics, sizes)
    ext_det = TagDetector(ext.intrinsics, {**sizes, args.wrist_tag_id: args.wrist_tag_size},
                          quad_decimate=2.0)
    print(f"palm {palm.intrinsics.width}x{palm.intrinsics.height}  "
          f"Daheng {ext.intrinsics.width}x{ext.intrinsics.height}  "
          f"cube {cube.edge_m * 1e3:g} mm, tile {cube.tag_size_m * 1e3:g} mm")
    print(f"hold the cube where BOTH cameras see it, still for a moment, then move "
          f"it to a clearly different orientation ({args.min_separation_deg:.0f} deg+). "
          f"Need {args.samples}.")

    samples: list[dict] = []
    prev: tuple[np.ndarray, np.ndarray] | None = None
    last_report, deadline = 0.0, time.time() + args.timeout
    while len(samples) < args.samples and time.time() < deadline:
      T_palm_cube, palm_tags, _ = read_pose(palm, palm_det, cube)
      T_ext_cube, ext_tags, ext_other = read_pose(ext, ext_det, cube)
      if T_palm_cube is not None and len(palm_tags) < args.min_tags:
        T_palm_cube = None
      if T_ext_cube is not None and len(ext_tags) < args.min_tags:
        T_ext_cube = None
      if T_palm_cube is None or T_ext_cube is None:
        prev = None
        if time.time() - last_report > 2.0:
          last_report = time.time()
          which = ("palm" if T_palm_cube is None else "") + \
                  (" daheng" if T_ext_cube is None else "")
          print(f"  waiting: no cube in {which.strip()}  "
                f"(palm {palm_tags}, daheng {ext_tags})", flush=True)
        continue

      if prev is not None:
        d_palm = pose_delta(T_palm_cube, prev[0])
        d_ext = pose_delta(T_ext_cube, prev[1])
        still = (max(d_palm[0], d_ext[0]) < args.still_deg and
                 max(d_palm[1], d_ext[1]) < args.still_mm)
      else:
        still = False
      prev = (T_palm_cube, T_ext_cube)
      if not still:
        continue

      # Distinctness is measured on the cube's orientation in the HAND frame: that
      # is what the rotation of T_hand_ext is solved against.
      q_new = mat_to_quat((T_hand_cam @ T_palm_cube)[:3, :3])
      if any(np.degrees(quat_angle(quat_mul(q_new, quat_conjugate(s["q"])))) <
             args.min_separation_deg for s in samples):
        continue

      T_hand_ext = (T_hand_cam @ T_palm_cube) @ se3_inv(T_ext_cube)
      wrist = {int(det.tag_id): se3(np.asarray(det.t, float),
                                    mat_to_quat(np.asarray(det.R, float)))
               for det in ext_other}
      samples.append({"T": T_hand_ext, "q": q_new, "palm_tags": palm_tags,
                      "ext_tags": ext_tags, "wrist": wrist})
      print(f"  [{len(samples):2d}/{args.samples}] palm tags {palm_tags}  "
            f"daheng tags {ext_tags}  wrist tags {sorted(wrist)}", flush=True)

    if len(samples) < 4:
      raise SystemExit(f"only {len(samples)} samples -- not enough. Is the cube "
                       "visible to both cameras at once?")

    # Consensus before averaging: keep the largest set of pairwise-agreeing samples.
    n_collected = len(samples)
    agree = []
    for i in range(n_collected):
      row = []
      for j in range(n_collected):
        d_deg, d_mm = pose_delta(samples[i]["T"], samples[j]["T"])
        if d_deg < args.consensus_deg and d_mm < args.consensus_mm:
          row.append(j)
      agree.append(row)
    inliers = agree[max(range(n_collected), key=lambda i: len(agree[i]))]
    T_consensus = mean_pose([samples[i]["T"] for i in inliers])
    detail = []
    print(f"\nper sample, against the consensus of {len(inliers)}/{n_collected} "
          f"({args.consensus_deg:g} deg / {args.consensus_mm:g} mm):")
    for i, smp in enumerate(samples):
      d_deg, d_mm = pose_delta(smp["T"], T_consensus)
      ok = i in inliers
      detail.append({"sample": i + 1, "palm_tags": [int(t) for t in smp["palm_tags"]],
                     "ext_tags": [int(t) for t in smp["ext_tags"]],
                     "deg": float(d_deg), "mm": float(d_mm), "inlier": ok})
      print(f"  [{i + 1:2d}] {d_deg:6.2f} deg {d_mm:7.1f} mm  "
            f"palm {len(smp['palm_tags'])} daheng {len(smp['ext_tags'])}  "
            f"{'' if ok else 'REJECTED'}")
    if len(inliers) < args.min_inliers:
      raise SystemExit(f"only {len(inliers)} samples agree (need {args.min_inliers}) "
                       "-- nothing written. Collect more, with more tiles in view.")
    samples = [samples[i] for i in inliers]

    T_mean = mean_pose([s["T"] for s in samples])
    spreads = [pose_delta(s["T"], T_mean) for s in samples]
    deg = np.array([x for x, _ in spreads])
    mm = np.array([y for _, y in spreads])

    # Leave-one-out: fit on the others, predict this one.
    loo_deg, loo_mm = [], []
    for i in range(len(samples)):
      others = mean_pose([s["T"] for j, s in enumerate(samples) if j != i])
      x, y = pose_delta(samples[i]["T"], others)
      loo_deg.append(x)
      loo_mm.append(y)

    print()
    print(f"samples                {len(samples)}")
    print(f"spread about the mean  {deg.mean():.2f} deg (max {deg.max():.2f})   "
          f"{mm.mean():.1f} mm (max {mm.max():.1f})")
    print(f"leave-one-out          {np.mean(loo_deg):.2f} deg (max {max(loo_deg):.2f})   "
          f"{np.mean(loo_mm):.1f} mm (max {max(loo_mm):.1f})")
    print(f"Daheng in hand frame   {np.round(T_mean[:3, 3] * 1e3, 1)} mm")

    # The wrist tag is constant while the arm is static; its spread here says
    # nothing moved during the collection.
    ids = sorted({i for s in samples for i in s["wrist"]})
    wrist_ref = {}
    for tag_id in ids:
      seen = [s["wrist"][tag_id] for s in samples if tag_id in s["wrist"]]
      T_tag = mean_pose(seen)
      dd = [pose_delta(T, T_tag) for T in seen]
      print(f"wrist tag {tag_id}          seen in {len(seen)}/{len(samples)} samples, "
            f"drift within this run {max(x for x, _ in dd):.2f} deg / "
            f"{max(y for _, y in dd):.1f} mm")
      wrist_ref[str(tag_id)] = T_tag.tolist()
    if str(args.wrist_tag_id) not in wrist_ref:
      raise SystemExit(f"wrist tag {args.wrist_tag_id} was never seen by the Daheng -- "
                       "run.py needs it. Nothing written.")
    wrist_in_hand = {}
    for tag_id, T_tag in wrist_ref.items():
      T = T_mean @ np.asarray(T_tag, float)
      wrist_in_hand[tag_id] = T.tolist()
      print(f"wrist tag {tag_id} in hand      {np.round(T[:3, 3] * 1e3, 1)} mm")

    out = args.out or (calib_dir() / "extcal.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
      "T_hand_ext": T_mean.tolist(),
      "cube_edge_m": cube.edge_m,
      "cube_tag_size_m": cube.tag_size_m,
      # Tags in `T_ext_wristtag` were solved at this size; `run.py` rescales.
      "solved_tag_size_m": args.wrist_tag_size,
      "T_hand_wristtag": wrist_in_hand,
      "n_samples": len(samples),
      "n_collected": n_collected,
      "min_tags": args.min_tags,
      "consensus_deg_mm": [args.consensus_deg, args.consensus_mm],
      "samples_detail": detail,
      "spread_deg_mean": float(deg.mean()), "spread_deg_max": float(deg.max()),
      "spread_mm_mean": float(mm.mean()), "spread_mm_max": float(mm.max()),
      "loo_deg_mean": float(np.mean(loo_deg)), "loo_deg_max": float(max(loo_deg)),
      "loo_mm_mean": float(np.mean(loo_mm)), "loo_mm_max": float(max(loo_mm)),
      "T_ext_wristtag": wrist_ref,
      "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    print(f"\nwrote {out}")
  finally:
    palm.stop()
    ext.stop()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
