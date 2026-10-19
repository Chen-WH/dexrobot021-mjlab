"""SE(3) and quaternion helpers, in the conventions this project already uses.

Quaternions are MuJoCo order, `w x y z`.  Camera frames are the OpenCV /
librealsense optical convention, `+x right, +y down, +z forward`.  Everything
here is plain numpy so the deployment runtime does not need torch, warp or a GPU.
"""

from __future__ import annotations

import numpy as np


def quat_to_mat(q) -> np.ndarray:
  w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
  R = np.asarray(R, float)
  t = np.trace(R)
  if t > 0.0:
    s = np.sqrt(t + 1.0) * 2.0
    q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                  (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
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


def quat_mul(a, b) -> np.ndarray:
  aw, ax, ay, az = a
  bw, bx, by, bz = b
  return np.array([
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ])


def quat_conjugate(q) -> np.ndarray:
  w, x, y, z = q
  return np.array([w, -x, -y, -z])


def quat_angle(q) -> float:
  """Rotation magnitude in radians, folded into [0, pi]."""
  return float(2.0 * np.arccos(np.clip(abs(np.asarray(q, float)[0]), 0.0, 1.0)))


def se3(pos, quat) -> np.ndarray:
  T = np.eye(4)
  T[:3, :3] = quat_to_mat(quat)
  T[:3, 3] = np.asarray(pos, float)
  return T


def se3_inv(T: np.ndarray) -> np.ndarray:
  R, t = T[:3, :3], T[:3, 3]
  out = np.eye(4)
  out[:3, :3] = R.T
  out[:3, 3] = -R.T @ t
  return out


def se3_split(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """`T` -> (translation, quaternion `w x y z`)."""
  return T[:3, 3].copy(), mat_to_quat(T[:3, :3])


def orthonormalise(R: np.ndarray) -> np.ndarray:
  """Nearest rotation matrix, by SVD. PnP output drifts off SO(3)."""
  u, _, vt = np.linalg.svd(np.asarray(R, float))
  R = u @ vt
  if np.linalg.det(R) < 0.0:
    u[:, -1] *= -1.0
    R = u @ vt
  return R


def quat_mean(quats: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
  """Weighted average of unit quaternions (Markley's eigenvector method).

  Sign-invariant, which matters here: two faces of the same cube can produce
  numerically opposite quaternions for the same physical orientation, and a
  naive component-wise mean of those cancels to zero.
  """
  q = np.atleast_2d(np.asarray(quats, float))
  w = np.ones(len(q)) if weights is None else np.asarray(weights, float)
  M = (q * w[:, None]).T @ q
  vals, vecs = np.linalg.eigh(M)
  out = vecs[:, int(np.argmax(vals))]
  out = out / np.linalg.norm(out)
  return out if out[0] >= 0.0 else -out


def angular_velocity_from_quats(q_prev, q_now, dt: float) -> np.ndarray:
  """Body-frame angular velocity from two orientations, rad/s.

  Returned in the frame `q_prev`/`q_now` are expressed in, matching
  `mdp.object_ang_vel_wrist`, which is the world angular velocity rotated into
  the wrist frame rather than a body-frame rate.
  """
  dq = quat_mul(np.asarray(q_now, float), quat_conjugate(np.asarray(q_prev, float)))
  if dq[0] < 0.0:
    dq = -dq
  vec = dq[1:]
  norm = float(np.linalg.norm(vec))
  if norm < 1e-9:
    return np.zeros(3)
  angle = 2.0 * np.arctan2(norm, dq[0])
  return (vec / norm) * (angle / dt)


def pose_delta(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
  """(degrees, mm) between two SE(3) poses.

  Reported in the units a person judges a rig by, because that is what these
  numbers get compared against: a bumped tripod, a drifting extrinsic, the
  scatter of a calibration.
  """
  dq = quat_mul(mat_to_quat(A[:3, :3]), quat_conjugate(mat_to_quat(B[:3, :3])))
  return (float(np.degrees(quat_angle(dq))),
          float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1e3))


def mean_pose(poses) -> np.ndarray:
  """Average of SE(3) poses: `quat_mean` for rotation, arithmetic for position.

  `quat_mean` is sign-invariant, which matters here for the same reason it does
  in `cube.pose_from_detections`: one physical orientation reaches this function
  as either sign of its quaternion.
  """
  q = quat_mean(np.array([mat_to_quat(np.asarray(T, float)[:3, :3]) for T in poses]))
  t = np.mean([np.asarray(T, float)[:3, 3] for T in poses], axis=0)
  return se3(t, q)
