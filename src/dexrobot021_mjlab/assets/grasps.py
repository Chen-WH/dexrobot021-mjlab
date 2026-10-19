"""The bench objects and the episode-start grasp solved for each of them.

An episode starts with the hand already closed around the object, so the
starting joint pose and the object's resting position are ONE answer: the joints
were solved (cube) or screened (cylinder) against this exact object, with
sub-millimetre finger gaps. Change the object and the grasp has to be re-solved;
it is not a default that carries over.
"""

from __future__ import annotations

import numpy as np

from dexrobot021_mjlab.assets.dexrobot021 import DEXROBOT021_LEFT_INIT_STATE

#: The bench cube: 70 mm edge, 142.7 g (mean of two weighings), AprilTag tiles on
#: every face (`hardware/cube_70mm/`).
CUBE_EDGE_M = 0.070
CUBE_MASS_KG = 0.1427

#: The bench cylinder: 60 mm across, 100 mm long, 118.65 g, one AprilTag on each
#: end face (`hardware/cylinder/`). Its MJCF carries the same numbers and
#: `objects.py` checks the two against each other.
CYLINDER_DIAMETER_M = 0.060
CYLINDER_LENGTH_M = 0.100
CYLINDER_MASS_KG = 0.11865


class Grasp:
  """One episode-start pose: the hand's twelve motor joints and where the object sits."""

  def __init__(
    self,
    obj: str,
    mass_kg: float,
    joint_ranges: dict[str, tuple[float, float]],
    object_pos_wrist: tuple[float, float, float],
    place_joint_pos: dict[str, float],
    quat_wrist: tuple[float, float, float, float] | None = None,
    note: str = "",
  ):
    #: "cube" or "cylinder".
    self.obj = obj
    self.mass_kg = mass_kg
    #: Reset range per motor joint (a +-5 mrad band around the solved value).
    self.joint_ranges = joint_ranges
    #: Object position in the wrist (hand base) frame.
    self.object_pos_wrist = object_pos_wrist
    #: Object orientation in the wrist frame (wxyz); None = unrotated in the world.
    self.quat_wrist = quat_wrist
    #: The open pose the hand holds WHILE the operator puts the object in on the
    #: rig; the deployment then closes to the grasp. Screened in simulation over
    #: placement errors of +-10 mm across the palm and +-5 mm in height.
    self.place_joint_pos = place_joint_pos
    self.note = note

  @property
  def spin_axis_wrist(self) -> tuple[float, float, float]:
    """The axis the spin task pays for, wrist frame.

    Cube: the wrist y axis (across the finger row, in the palm plane). Cylinder:
    the cylinder's own axis (its local z) as it lies in the grasp.
    """
    if self.quat_wrist is not None:
      return tuple(float(v) for v in _quat_to_mat(self.quat_wrist)[:, 2])
    return (0.0, 1.0, 0.0)

  @property
  def object_quat_world(self) -> tuple[float, float, float, float]:
    """Spawn orientation (wxyz), carried into the world through the welded hand."""
    if self.quat_wrist is not None:
      R = _quat_to_mat(DEXROBOT021_LEFT_INIT_STATE.rot) @ _quat_to_mat(self.quat_wrist)
      return _mat_to_quat(R)
    return (1.0, 0.0, 0.0, 0.0)

  @property
  def object_pos_world(self) -> tuple[float, float, float]:
    """Spawn position, derived from `object_pos_wrist` through the welded hand."""
    rot = _quat_to_mat(DEXROBOT021_LEFT_INIT_STATE.rot)
    world = np.asarray(DEXROBOT021_LEFT_INIT_STATE.pos, float) + rot @ np.asarray(
      self.object_pos_wrist, float)
    return tuple(round(float(v), 5) for v in world)


def _quat_to_mat(q) -> np.ndarray:
  w, x, y, z = (float(v) for v in q)
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ])


def _mat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
  w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
  x = np.copysign(np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
  y = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
  z = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
  q = np.array([w, x, y, z])
  q /= np.linalg.norm(q)
  return tuple(round(float(v), 6) for v in q)


def _band(pose: dict[str, float]) -> dict[str, tuple[float, float]]:
  return {n: (round(v - 0.005, 3), round(v + 0.005, 3)) for n, v in pose.items()}


GRASPS: dict[str, Grasp] = {
  # Solved with a CEM + Nelder-Mead search over the twelve joints and the cube
  # position, under one hard constraint: nothing touches the cube's top band, so the
  # operator can lower the cube in from above. All five fingers touch with positive
  # gaps (0.5-2 mm; a preloaded reset cannot be commanded on hardware), the palm is
  # under the cube, no hand geom penetrates more than 0.2 mm. Thumb on the
  # thumb-side face, index/middle/ring tips over the far face, little finger on the
  # opposite side face. Held with zero action for 1.2 s: 0.7 mm / 1.1 deg.
  "cube": Grasp(
    obj="cube",
    mass_kg=CUBE_MASS_KG,
    joint_ranges={
      "l_f_joint1_1": (0.587, 0.597),
      "l_f_joint1_2": (0.471, 0.481),
      "l_f_joint1_3": (0.756, 0.766),
      "l_f_joint2_1": (0.121, 0.131),
      "l_f_joint2_2": (0.598, 0.608),
      "l_f_joint2_3": (0.787, 0.797),
      "l_f_joint3_2": (0.718, 0.728),
      "l_f_joint3_3": (0.774, 0.784),
      "l_f_joint4_2": (0.495, 0.505),
      "l_f_joint4_3": (0.887, 0.897),
      "l_f_joint5_2": (0.889, 0.899),
      "l_f_joint5_3": (0.844, 0.854),
    },
    object_pos_wrist=(0.0680, -0.0012, 0.1703),
    # Every flexion opened BY 0.45 rad from the grasp (floored at 0); thumb rotation
    # and index abduction stay put. Opening by a constant relieves every finger
    # equally; 14 of 15 screened placements closed back into the grasp.
    place_joint_pos={
      "l_f_joint1_1": 0.592, "l_f_joint1_2": 0.026, "l_f_joint1_3": 0.311,
      "l_f_joint2_1": 0.126, "l_f_joint2_2": 0.153, "l_f_joint2_3": 0.342,
      "l_f_joint3_2": 0.273, "l_f_joint3_3": 0.329, "l_f_joint4_2": 0.050,
      "l_f_joint4_3": 0.442, "l_f_joint5_2": 0.444, "l_f_joint5_3": 0.399,
    },
    note="70 mm cube, five-finger grasp from the side and far faces",
  ),
  # Screened by dropping the cylinder across the open hand (axis along wrist y) and
  # closing every flexion joint to 1.3 rad at 1 rad/s: all 36 drops (height, lateral
  # offset and thumb rotation varied) converged to this five-finger wrap on the palm.
  # With zero action it holds for 2 s: 3.6 mm / 1.7 deg. Every finger touches the
  # curved side except the index distal link, which rests on the thumb-side end face.
  "cylinder": Grasp(
    obj="cylinder",
    mass_kg=CYLINDER_MASS_KG,
    joint_ranges=_band({
      "l_f_joint1_1": 1.391, "l_f_joint1_2": 0.342, "l_f_joint1_3": 0.946,
      "l_f_joint2_1": 0.139, "l_f_joint2_2": 0.914, "l_f_joint2_3": 0.750,
      "l_f_joint3_2": 1.147, "l_f_joint3_3": 0.793, "l_f_joint4_2": 1.138,
      "l_f_joint4_3": 0.779, "l_f_joint5_2": 1.052, "l_f_joint5_3": 0.775,
    }),
    object_pos_wrist=(0.0554, -0.0039, 0.1546),
    quat_wrist=(0.6011, -0.7696, 0.1655, 0.1379),
    # Thumb already opposed, index abduction as in the grasp, every flexion at
    # 0.15 rad. Openings from 0.30 down to 0.10 rad all close into the same grasp
    # (36/36); 0.05 does not. 0.15 keeps two tested steps of margin.
    place_joint_pos={
      "l_f_joint1_1": 1.391, "l_f_joint1_2": 0.150, "l_f_joint1_3": 0.150,
      "l_f_joint2_1": 0.139, "l_f_joint2_2": 0.150, "l_f_joint2_3": 0.150,
      "l_f_joint3_2": 0.150, "l_f_joint3_3": 0.150, "l_f_joint4_2": 0.150,
      "l_f_joint4_3": 0.150, "l_f_joint5_2": 0.150, "l_f_joint5_3": 0.150,
    },
    note="cylinder lying across the palm, five-finger wrap, thumb rotation 1.39 rad",
  ),
}
