"""Object MJCF bindings: the 70 mm cube and the 60 x 100 mm cylinder."""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg

from dexrobot021_mjlab.assets.grasps import (
  CUBE_EDGE_M,
  CUBE_MASS_KG,
  CYLINDER_DIAMETER_M,
  CYLINDER_LENGTH_M,
  CYLINDER_MASS_KG,
  GRASPS,
)
from dexrobot021_mjlab.paths import cube_mjcf, cylinder_mjcf

_CUBE_HALF_EXTENTS = (CUBE_EDGE_M / 2.0,) * 3


def _get_cube_spec() -> mujoco.MjSpec:
  path = cube_mjcf()
  if not path.exists():
    raise FileNotFoundError(f"Cube MJCF not found: {path}")
  spec = mujoco.MjSpec.from_file(str(path))
  asset_root = path.parent
  for texture in spec.textures:
    if texture.file:
      texture.file = str((asset_root / texture.file).resolve())
    for i in range(len(texture.cubefiles)):
      texture.cubefiles[i] = str((asset_root / texture.cubefiles[i]).resolve())
  _resize_cube(spec)
  return spec


def _resize_cube(spec: mujoco.MjSpec) -> None:
  """Size the collision box to the bench cube and give it the weighed mass."""
  found = False
  for geom in spec.geoms:
    if geom.name == "cube":
      geom.size[:3] = _CUBE_HALF_EXTENTS
      geom.mass = CUBE_MASS_KG
      found = True
      break
  if not found:
    raise ValueError("Cube MJCF has no geom named 'cube'")
  # The axis_* geoms are visualisation only (contype=0, density=0); keep them
  # flush with the faces they mark.
  for geom in spec.geoms:
    axis = {"axis_x": 0, "axis_y": 1, "axis_z": 2}.get(geom.name)
    if axis is None:
      continue
    geom.size[axis] = _CUBE_HALF_EXTENTS[axis]
    geom.pos[axis] = _CUBE_HALF_EXTENTS[axis]


def _get_cylinder_spec() -> mujoco.MjSpec:
  """The bench cylinder, checked against `grasps` so the two cannot drift apart.

  The asset's frame sensors and light are dropped: they name the body and the
  scene has its own.
  """
  path = cylinder_mjcf()
  if not path.exists():
    raise FileNotFoundError(f"Cylinder MJCF not found: {path}")
  spec = mujoco.MjSpec.from_file(str(path))
  for sensor in list(spec.sensors):
    spec.delete(sensor)
  for light in list(spec.lights):
    spec.delete(light)
  geom = next((g for g in spec.geoms if g.name == "cylinder"), None)
  if geom is None:
    raise ValueError(f"{path} has no geom named 'cylinder'")
  radius, half = float(geom.size[0]), float(geom.size[1])
  want = (CYLINDER_DIAMETER_M / 2, CYLINDER_LENGTH_M / 2, CYLINDER_MASS_KG)
  have = (radius, half, float(geom.mass))
  if any(abs(a - b) > 1e-6 for a, b in zip(have, want, strict=True)):
    raise ValueError(
      f"{path}: radius / half-length / mass {have} != {want} (assets/grasps.py)")
  return spec


def _object_cfg(spec_fn, obj: str) -> EntityCfg:
  grasp = GRASPS[obj]
  return EntityCfg(
    spec_fn=spec_fn,
    init_state=EntityCfg.InitialStateCfg(
      pos=grasp.object_pos_world,
      rot=grasp.object_quat_world,
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
      joint_pos={},
      joint_vel={},
    ),
  )


def get_cube_cfg() -> EntityCfg:
  return _object_cfg(_get_cube_spec, "cube")


def get_cylinder_cfg() -> EntityCfg:
  return _object_cfg(_get_cylinder_spec, "cylinder")
