"""Tagged objects: which AprilTag is where, and how the tags fuse into one pose.

The cube carries a 2x2 grid of tag36h11 tiles on every face (ids 20-43, printed
from `hardware/cube_70mm/`); the cylinder carries one tag on each end face (ids 6
and 7, `hardware/cylinder/`). Tag 10 is the wrist tag and belongs to neither.

Tag frame convention is AprilTag's own: **`+x` right across the printed image,
`+y` down it, `+z` INTO the tag** -- so a face's outward normal is `-z`. The
handedness is not a choice: a mirrored tag36h11 carries no valid code, so a
mirrored layout simply never decodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import mat_to_quat, orthonormalise, quat_mean, se3, se3_inv

# Cube frame: +X right, +Y front, +Z top.
_AXES = {
  "+X": np.array([1.0, 0.0, 0.0]), "-X": np.array([-1.0, 0.0, 0.0]),
  "+Y": np.array([0.0, 1.0, 0.0]), "-Y": np.array([0.0, -1.0, 0.0]),
  "+Z": np.array([0.0, 0.0, 1.0]), "-Z": np.array([0.0, 0.0, -1.0]),
}

# face -> (full-face tag id, outward normal, direction the printed image's TOP edge
# points). The tiles inherit each face's normal and up direction from here.
# **`right` is the cube's own right (+X), which is on the LEFT of anyone facing
# the front face.**
PHYSICAL_LAYOUT: dict[str, tuple[int, str, str]] = {
  "top":    (0, "+Z", "-Y"),
  "front":  (1, "+Y", "+Z"),
  "right":  (4, "+X", "+Z"),  # +X is the viewer's LEFT when facing the front
  "back":   (3, "-Y", "+Z"),
  "left":   (2, "-X", "+Z"),  # -X is the viewer's RIGHT
  "bottom": (5, "-Z", "+Y"),
}

#: In-face displacement of a 2x2 tile, as `(along image-right, along image-up)` in
#: units of a quarter edge. A face's image-right in the cube frame is
#: `up x normal` and its image-up is `up`. Derived here rather than transcribed;
#: `tests/test_tile_layout.py` compares it with the print generator's mapping
#: (`hardware/cube_70mm/cube_tile_mapping_2x2_70mm.yaml`).
_TILE_CORNERS = {"TL": (-1, +1), "TR": (+1, +1), "BL": (-1, -1), "BR": (+1, -1)}

#: Face order the tile ids were handed out in, and the first id.
_TILE_FACE_ORDER = ("top", "front", "right", "back", "left", "bottom")
FIRST_TILE_ID = 20

#: A tag36h11 label is 10 modules across and its black square is 8 of them.
TILE_MODULES, TILE_DETECTION_MODULES = 10, 8


def tile_tag_size_m(edge_m: float, grid: int = 2) -> float:
  """Side of one printed tile's black square, derived from the cube edge.

  A wrong tag size does not fail: it scales every recovered TRANSLATION and leaves
  the rotation alone, so nothing looks broken. Hence derived, never typed.
  """
  return edge_m / grid * TILE_DETECTION_MODULES / TILE_MODULES


def tile_layout(base: dict[str, tuple[int, str, str]] | None = None
                ) -> dict[str, tuple[int, str, str, np.ndarray]]:
  """`{f"{face}:{corner}": (tag_id, normal, up, offset)}` for the 2x2 scheme.

  `offset` is the tile centre's displacement from the face centre, in the cube
  frame, for a unit cube edge -- `Cube` scales it by its own edge length.
  """
  faces = base or PHYSICAL_LAYOUT
  out, next_id = {}, FIRST_TILE_ID
  for face in _TILE_FACE_ORDER:
    _, normal, up = faces[face]
    n, u = _AXES[normal], _AXES[up]
    right = np.cross(u, n)
    for corner, (sr, su) in _TILE_CORNERS.items():
      out[f"{face}:{corner}"] = (next_id, normal, up,
                                 0.25 * (sr * right + su * u))
      next_id += 1
  return out


#: Below this, a detection's payload bits were not read cleanly enough to trust its
#: corners, which is what the pose is made of.
MIN_DECISION_MARGIN = 20.0

#: A tag is dropped past this obliquity, because corner localisation degrades long
#: before the payload stops decoding. With 2x2 tiles the fusion weights each vote by
#: cos(obliquity) and grazing tiles still agree with the rest, so the gate is loose.
MAX_OBLIQUITY_DEG = 75.0


@dataclass(frozen=True)
class TagObservation:
  """One detected tag, as the detector reports it.

  `R`/`t` place the tag in the camera's optical frame: a point `p_tag` maps to
  `R @ p_tag + t`. `z_points_outward` says whether the detector's tag `+z` leaves
  the printed surface toward the camera or enters it, as this file's frames do.
  Get it wrong and every orientation is mirrored about the face.
  """

  tag_id: int
  R: np.ndarray
  t: np.ndarray
  decision_margin: float = 0.0
  z_points_outward: bool = False


def _tag_axes(normal: str, up: str) -> np.ndarray:
  """Columns are the tag frame's x, y, z expressed in the object frame.

  `+y` runs down the printed image and `+z` points INTO the tag, so the outward
  face normal is `-z`, and `x = y cross z` points image-right.
  """
  n, u = _AXES[normal], _AXES[up]
  z = -n           # +z enters the face; the outward normal is -z
  y = -u           # +y runs down the printed image
  x = np.cross(y, z)
  return np.column_stack([x, y, z])


class Cube:
  """Geometry of one tagged object: `tag id -> T_cube_tag`, and the fusion.

  `edge_m` is the distance between opposite tagged faces (the cube's edge, or the
  cylinder's length).
  """

  def __init__(self, edge_m: float, tag_size_m: float,
               layout: dict | None = None):
    self.edge_m = float(edge_m)
    self.tag_size_m = float(tag_size_m)
    self.layout = dict(layout or tile_layout())

    self.T_cube_tag: dict[int, np.ndarray] = {}
    self.face_of_tag: dict[int, str] = {}
    #: Layout key per tag id: the face name, or `"face:corner"` for a tile.
    self.key_of_tag: dict[int, str] = {}
    #: Tag ids grouped by face. Four coplanar tiles agree with each other by
    #: construction, so a statistic meant to catch a wrong layout has to be
    #: computed across faces rather than across tags.
    self.tags_of_face: dict[str, list[int]] = {}
    for key, entry in self.layout.items():
      # A three-entry row is a tag centred on its face; a four-entry row carries a
      # tile's in-face offset, in units of the edge.
      tag_id, normal, up = entry[0], entry[1], entry[2]
      offset = np.zeros(3) if len(entry) < 4 else np.asarray(entry[3], float)
      face = key.split(":", 1)[0]
      T = np.eye(4)
      T[:3, :3] = _tag_axes(normal, up)
      # The sticker lies on the face, half an edge out along its OUTWARD normal
      # (`-z` of the tag frame), then displaced within the face to its tile.
      T[:3, 3] = (_AXES[normal] * 0.5 + offset) * self.edge_m
      self.T_cube_tag[tag_id] = T
      self.face_of_tag[tag_id] = face
      self.key_of_tag[tag_id] = key
      self.tags_of_face.setdefault(face, []).append(tag_id)

  # -- layout self-check ------------------------------------------------------

  def verify_layout(self) -> list[str]:
    """Every tag's outward normal and printed up direction match its face, every
    frame is right-handed, and every tile sits in its face plane."""
    problems = []
    for key, entry in self.layout.items():
      tag_id, normal, up = entry[0], entry[1], entry[2]
      offset = np.zeros(3) if len(entry) < 4 else np.asarray(entry[3], float)
      face = key
      R = self.T_cube_tag[tag_id][:3, :3]
      if not np.allclose(-R[:, 2], _AXES[normal], atol=1e-9):
        problems.append(
          f"tag {tag_id} ({face}): outward normal is {(-R[:, 2]).round(3).tolist()}, "
          f"expected {_AXES[normal].tolist()}")
      if not np.allclose(-R[:, 1], _AXES[up], atol=1e-9):
        problems.append(
          f"tag {tag_id} ({face}): printed up is {(-R[:, 1]).round(3).tolist()}, "
          f"expected {_AXES[up].tolist()}")
      if not np.isclose(np.linalg.det(R), 1.0, atol=1e-9):
        problems.append(f"tag {tag_id} ({face}): frame is not right-handed")
      origin = self.T_cube_tag[tag_id][:3, 3]
      expected = (_AXES[normal] * 0.5 + offset) * self.edge_m
      if not np.allclose(origin, expected, atol=1e-9):
        problems.append(f"tag {tag_id} ({face}): origin is not at its tile position")
      if abs(float(np.dot(offset, _AXES[normal]))) > 1e-12:
        problems.append(f"tag {tag_id} ({face}): offset leaves the face plane")
      if len(entry) >= 4 and float(np.linalg.norm(offset)) > 0.5 * np.sqrt(2) + 1e-12:
        problems.append(f"tag {tag_id} ({face}): offset reaches past the face edge")
    return problems

  # -- fusion -----------------------------------------------------------------

  def pose_from_detections(
    self,
    detections: list[TagObservation],
    max_obliquity_deg: float = MAX_OBLIQUITY_DEG,
    min_decision_margin: float = MIN_DECISION_MARGIN,
  ) -> tuple[np.ndarray, dict] | tuple[None, dict]:
    """Fuse tag detections into one `T_cam_cube`.

    Each accepted tag votes a full object pose. Translations are averaged with
    weights, orientations with `quat_mean`, which is sign-invariant (opposite faces
    routinely produce negated quaternions for one physical orientation).

    Weighting is by `cos(obliquity)`: a tag seen edge-on still solves, but its
    out-of-plane rotation is the poorly conditioned direction of a planar PnP.
    """
    votes_t, votes_q, weights, used, rejected = [], [], [], [], {}
    for det in detections:
      T_cube_tag = self.T_cube_tag.get(det.tag_id)
      if T_cube_tag is None:
        rejected[det.tag_id] = "unknown tag id"
        continue
      if det.decision_margin < min_decision_margin:
        rejected[det.tag_id] = f"decision margin {det.decision_margin:.0f}"
        continue

      R = orthonormalise(det.R)
      if det.z_points_outward:
        # Bring it into this file's frame, where +z enters the face.
        R = R @ np.diag([1.0, -1.0, -1.0])
      # Obliquity: angle between the outward face normal (-z) and the ray back to
      # the camera.
      normal_cam = -R[:, 2]
      to_camera = -np.asarray(det.t, float)
      to_camera = to_camera / max(np.linalg.norm(to_camera), 1e-9)
      cos = float(np.dot(normal_cam, to_camera))
      if cos <= 0.0 or np.degrees(np.arccos(np.clip(cos, -1, 1))) > max_obliquity_deg:
        oblique = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        rejected[det.tag_id] = f"obliquity {oblique:.0f} deg"
        continue

      T_cam_tag = np.eye(4)
      T_cam_tag[:3, :3] = R
      T_cam_tag[:3, 3] = np.asarray(det.t, float)
      T_cam_cube = T_cam_tag @ se3_inv(T_cube_tag)
      votes_t.append(T_cam_cube[:3, 3])
      votes_q.append(mat_to_quat(T_cam_cube[:3, :3]))
      weights.append(cos)
      used.append(det.tag_id)

    info = {"tags_used": used, "tags_rejected": rejected, "n_used": len(used)}
    if not used:
      return None, info

    w = np.asarray(weights)
    q = quat_mean(np.asarray(votes_q), w)
    t = (np.asarray(votes_t) * w[:, None]).sum(0) / w.sum()
    if len(used) > 1:
      # Spread between single-tag votes is the runtime consistency check: a
      # mis-assigned face shows up as tens of degrees, not tenths.
      from .geometry import quat_angle, quat_conjugate, quat_mul
      info["orientation_spread_deg"] = round(float(np.degrees(max(
        quat_angle(quat_mul(qi, quat_conjugate(q))) for qi in votes_q))), 2)
      info["position_spread_mm"] = round(
        float(np.linalg.norm(np.asarray(votes_t) - t, axis=1).max() * 1e3), 2)
    return se3(t, q), info


# -- the cylinder -------------------------------------------------------------
#
# One tag on each end face: id 6 on +Z, id 7 on -Z, 36 mm labels with a 28.8 mm
# black square. That is `Cube` geometry with a two-row layout and `edge_m` standing
# for the LENGTH (the diameter never enters the tag geometry). The in-plane angle
# of each sticker and which end carries id 6 do not matter: training randomises
# the cylinder's roll about its own axis and flips it end for end.
CYLINDER_TAG_SIZE_M = 0.0288

#: end -> (tag id, outward normal, direction the tag image's TOP edge points).
CYLINDER_LAYOUT: dict[str, tuple[int, str, str]] = {
  "top":    (6, "+Z", "+X"),
  "bottom": (7, "-Z", "+X"),
}


def cylinder(length_m: float, tag_size_m: float | None = None) -> Cube:
  """The tagged bench cylinder: `Cube` geometry with the end-face layout."""
  return Cube(length_m, tag_size_m or CYLINDER_TAG_SIZE_M, layout=CYLINDER_LAYOUT)


def object_from_contract(contract) -> Cube:
  """The tag geometry a bundle's object needs, from the contract alone. One
  constructor for every script, so the object cannot be wired differently in two
  of them."""
  if getattr(contract, "object_kind", "cube") == "cylinder":
    return cylinder(contract.object_length_m,
                    contract.cube_tag_size_m if contract.cube_tag_size_measured else None)
  return Cube(contract.cube_edge_m, tile_tag_size_m(contract.cube_edge_m),
              layout=tile_layout())
