"""The CAD hand-eye extrinsic of the palm camera: `T_left_hand_base__d435i_color_optical`.

The reference rig hangs the DexHand and a RealSense D435i off one bracket at the
UR10 flange, so the transform closes inside the bracket and the arm's pose never
enters it. The frames below are the bracket's CAD, with the hand installed yawed
180 deg about its own z relative to the grasping build (which swings the camera
round to face the palm).

This transform is exported into every bundle (`contract.T_hand_cam`). It is used
directly when the palm camera observes the object, and as the reference that
`extcal.py` calibrates an external camera (the Daheng) against. A different rig
needs its own numbers here, or its own calibration.

    python scripts/deploy/extrinsic.py            # print the transform and viewing geometry
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

# mount -> body, (pos, quat wxyz), from the rig's CAD scene.
MJCF_FRAMES = {
  # mount -> d435i body origin
  "mount_d435i": ((0.0, 0.16654, -0.0130), (0.0, 0.0, -0.9659, 0.2588)),
  # mount -> left_hand_base, as installed for the in-hand tasks
  "mount_hand": ((0.0, 0.0, 0.0), (0.0, 0.7071, 0.7071, 0.0)),
}

# d435i body -> its RGB camera. MuJoCo cameras look down -z with +y up; the
# OpenCV / librealsense optical frame is +z forward, +y down.
D435I_TO_RGB_CAM = ((0.03, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0))
MUJOCO_TO_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0])

HAND_INSTALL_YAW_DEG = 180.0

# Wrist-frame point the object settles into; only used to report viewing geometry.
CUBE_POS_WRIST = np.array([0.068, -0.001, 0.170])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
  """MuJoCo quaternion order, `w x y z`."""
  w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
  """Inverse of `quat_to_mat`, returned as `w x y z` with a non-negative w."""
  t = np.trace(R)
  if t > 0.0:
    s = np.sqrt(t + 1.0) * 2.0
    q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                  (R[1, 0] - R[0, 1]) / s])
  else:
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
  q /= np.linalg.norm(q)
  return q if q[0] >= 0.0 else -q


def se3(pos, quat) -> np.ndarray:
  T = np.eye(4)
  T[:3, :3] = quat_to_mat(np.asarray(quat, float))
  T[:3, 3] = np.asarray(pos, float)
  return T


def se3_error(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
  """Rotation (deg) and translation (mm) of `B` relative to `A`."""
  D = np.linalg.inv(A) @ B
  cos = np.clip((np.trace(D[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
  return float(np.degrees(np.arccos(cos))), float(np.linalg.norm(D[:3, 3]) * 1e3)


def cad_chain() -> np.ndarray:
  """`T_left_hand_base__d435i_color_optical` from the bracket CAD."""
  T_mount_cam = (se3(*MJCF_FRAMES["mount_d435i"]) @ se3(*D435I_TO_RGB_CAM)
                 @ MUJOCO_TO_OPTICAL)
  return np.linalg.inv(se3(*MJCF_FRAMES["mount_hand"])) @ T_mount_cam


def derive(hand_yaw_deg: float = HAND_INSTALL_YAW_DEG) -> dict:
  """The deployed `T_hand_cam` and where it came from."""
  return {"T_hand_cam": cad_chain(), "source": "cad", "hand_yaw_deg": hand_yaw_deg}


def viewing_geometry(T_hand_cam: np.ndarray) -> dict:
  """Where the object sits in the camera's field of view, at the nominal grasp."""
  cam_pos = T_hand_cam[:3, 3]
  optical_axis = T_hand_cam[:3, 2]  # optical +z is column 2
  to_cube = CUBE_POS_WRIST - cam_pos
  distance = float(np.linalg.norm(to_cube))
  cos = float(np.dot(to_cube / distance, optical_axis))
  return {
    "camera_pos_wrist_mm": (cam_pos * 1e3).round(1).tolist(),
    "cube_distance_mm": round(distance * 1e3, 1),
    "off_axis_deg": round(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))), 1),
    # +x is the palm side.
    "camera_on_palm_side": bool(cam_pos[0] > 0.0),
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--write", type=pathlib.Path, default=None,
                  help="write the extrinsic to this JSON path")
  args = ap.parse_args()

  T = cad_chain()
  np.set_printoptions(precision=4, suppress=True)
  print("T_left_hand_base__d435i_color_optical  (CAD, as installed)")
  print(T)
  print("\nviewing geometry at the nominal grasp")
  for key, value in viewing_geometry(T).items():
    print(f"  {key:24s} {value}")
  if args.write is not None:
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(json.dumps({
      "name": "T_left_hand_base__d435i_color_optical",
      "parent_frame": "left_hand_base",
      "child_frame": "d435i_color_optical",
      "camera_axes": "OpenCV/librealsense optical: +x right, +y down, +z forward",
      "source": "cad",
      "hand_install_yaw_deg": HAND_INSTALL_YAW_DEG,
      "matrix_row_major": T.tolist(),
      "translation_m": T[:3, 3].tolist(),
      "quaternion_wxyz": mat_to_quat(T[:3, :3]).tolist(),
    }, indent=2) + "\n")
    print(f"\nwrote {args.write}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
