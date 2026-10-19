"""Reset and startup events for coupled-hand MJCF assets."""

from __future__ import annotations

import math
from typing import Literal

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg, SceneEntityCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import sample_uniform


class reset_joints_with_mimic_followers(ManagerTermBase):
  """Reset active joints and initialize equality-coupled followers consistently."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)

    required = {"position_range", "velocity_range", "mimic_relations"}
    missing = sorted(required.difference(cfg.params))
    if missing:
      raise ValueError(
        "reset_joints_with_mimic_followers requires params "
        f"{sorted(required)}. Missing: {missing}."
      )

    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
    self._asset = env.scene[asset_cfg.name]
    use_default_offset = cfg.params.get("use_default_offset", False)
    operation = cfg.params.get("operation", "abs")
    if operation not in {"abs", "scale"}:
      raise ValueError(f"Unsupported operation for joint reset: {operation!r}")

    default_joint_pos = self._asset.data.default_joint_pos[0]
    default_joint_vel = self._asset.data.default_joint_vel[0]

    self._pos_ranges = self._asset.data.soft_joint_pos_limits[0].clone()
    pos_joint_ids: list[int] = []
    for joint_name, joint_range in cfg.params["position_range"].items():
      joint_ids = self._asset.find_joints(joint_name)[0]
      pos_joint_ids.extend(joint_ids)
      if operation == "abs":
        if joint_range[0] is not None:
          self._pos_ranges[joint_ids, 0] = joint_range[0]
        if joint_range[1] is not None:
          self._pos_ranges[joint_ids, 1] = joint_range[1]
      else:
        if joint_range[0] is not None:
          self._pos_ranges[joint_ids, 0] *= joint_range[0]
        if joint_range[1] is not None:
          self._pos_ranges[joint_ids, 1] *= joint_range[1]
      if use_default_offset:
        self._pos_ranges[joint_ids] += default_joint_pos[joint_ids].unsqueeze(1)

    self._pos_joint_ids = torch.tensor(
      list(dict.fromkeys(pos_joint_ids)),
      device=env.device,
      dtype=torch.long,
    )
    self._pos_ranges = self._pos_ranges[self._pos_joint_ids]

    self._vel_ranges = torch.zeros(
      self._asset.num_joints,
      2,
      device=env.device,
      dtype=default_joint_vel.dtype,
    )
    vel_joint_ids: list[int] = []
    for joint_name, joint_range in cfg.params["velocity_range"].items():
      joint_ids = self._asset.find_joints(joint_name)[0]
      vel_joint_ids.extend(joint_ids)
      if joint_range[0] is not None:
        self._vel_ranges[joint_ids, 0] = joint_range[0]
      if joint_range[1] is not None:
        self._vel_ranges[joint_ids, 1] = joint_range[1]
      if use_default_offset:
        self._vel_ranges[joint_ids] += default_joint_vel[joint_ids].unsqueeze(1)

    self._vel_joint_ids = torch.tensor(
      list(dict.fromkeys(vel_joint_ids)),
      device=env.device,
      dtype=torch.long,
    )
    self._vel_ranges = self._vel_ranges[self._vel_joint_ids]

    joint_name_to_id = {
      joint_name: joint_id for joint_id, joint_name in enumerate(self._asset.joint_names)
    }
    # Each entry is (target, reference, polycoef ascending). A follower is placed
    # by evaluating the MJCF's own polynomial, so the written pose sits on the
    # equality constraint manifold instead of a linear approximation of it.
    self._mimic_relations: list[tuple[int, int, tuple[float, ...]]] = []
    for target_name, relation in cfg.params["mimic_relations"].items():
      reference_name = relation[0]
      target_id = joint_name_to_id.get(target_name)
      reference_id = joint_name_to_id.get(reference_name)
      if target_id is None or reference_id is None:
        continue
      polycoef = tuple(float(c) for c in relation[1])
      self._mimic_relations.append((target_id, reference_id, polycoef))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    position_range: dict[str, tuple[float | None, float | None]],
    velocity_range: dict[str, tuple[float | None, float | None]],
    mimic_relations: dict[str, tuple[str, tuple[float, ...]]],
    use_default_offset: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    operation: Literal["abs", "scale"] = "abs",
  ) -> None:
    del position_range, velocity_range, mimic_relations, use_default_offset
    del asset_cfg, operation

    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    joint_pos = self._asset.data.default_joint_pos[env_ids].clone()
    joint_vel = self._asset.data.default_joint_vel[env_ids].clone()

    if len(self._pos_joint_ids) > 0:
      joint_pos[:, self._pos_joint_ids] = sample_uniform(
        self._pos_ranges[:, 0],
        self._pos_ranges[:, 1],
        (len(env_ids), len(self._pos_joint_ids)),
        device=joint_pos.device,
      )
      limits = self._asset.data.soft_joint_pos_limits[0, self._pos_joint_ids]
      joint_pos[:, self._pos_joint_ids] = joint_pos[:, self._pos_joint_ids].clamp(
        limits[:, 0],
        limits[:, 1],
      )

    if len(self._vel_joint_ids) > 0:
      joint_vel[:, self._vel_joint_ids] = sample_uniform(
        self._vel_ranges[:, 0],
        self._vel_ranges[:, 1],
        (len(env_ids), len(self._vel_joint_ids)),
        device=joint_vel.device,
      )

    for target_id, reference_id, polycoef in self._mimic_relations:
      reference_pos = joint_pos[:, reference_id]
      value = torch.zeros_like(reference_pos)
      derivative = torch.zeros_like(reference_pos)
      for power, coefficient in enumerate(polycoef):
        if coefficient == 0.0:
          continue
        value = value + coefficient * reference_pos.pow(power)
        if power >= 1:
          derivative = derivative + power * coefficient * reference_pos.pow(power - 1)
      joint_pos[:, target_id] = value
      # Chain rule, so follower velocities are consistent with the constraint too.
      joint_vel[:, target_id] = joint_vel[:, reference_id] * derivative

    limits = self._asset.data.soft_joint_pos_limits[0]
    joint_pos = joint_pos.clamp(limits[:, 0], limits[:, 1])
    self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def sample_extrinsic_bias(
  env,
  env_ids,
  max_angle_rad: float = 0.08,
  max_offset_m: float = 0.010,
) -> None:
  """Draw a per-environment constant error in the hand-eye transform.

  On hardware the object pose comes from tags seen through a camera whose pose
  relative to the wrist is known only as well as it was calibrated. Per-step
  Gaussian noise does not stand in for that: a policy averages zero-mean noise away
  over a few steps and cannot average away a constant.

  Modelled as a rigid error on the measurement, `T_measured = T_error . T_true`, so
  one rotation and one offset move the measured position and orientation together
  the way a real extrinsic error does. Redrawn every episode.
  """
  import torch
  from mjlab.utils.lab_api import math as math_utils

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  if not hasattr(env, "extrinsic_bias_quat"):
    env.extrinsic_bias_quat = torch.zeros(env.num_envs, 4, device=env.device)
    env.extrinsic_bias_quat[:, 0] = 1.0
    env.extrinsic_bias_pos = torch.zeros(env.num_envs, 3, device=env.device)

  axis = torch.randn(n, 3, device=env.device)
  axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
  angle = torch.rand(n, device=env.device) * max_angle_rad
  env.extrinsic_bias_quat[env_ids] = math_utils.quat_from_angle_axis(angle, axis)

  direction = torch.randn(n, 3, device=env.device)
  direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
  env.extrinsic_bias_pos[env_ids] = (
    direction * (torch.rand(n, 1, device=env.device) * max_offset_m)
  )


def reset_object_pose_and_orientation(
  env,
  env_ids,
  asset_cfg=None,
  pose_range: dict | None = None,
  symmetries: bool | str = True,
  max_tilt_rad: float = 0.70,
) -> None:
  """Place the object where a real grasp leaves it: position, symmetry branch, tilt.

  * Position: the grasp's spawn plus a uniform offset (`pose_range`, world axes).
  * Symmetry branch: a cube settles in one of its 24 self-maps depending on how it
    was dropped in, and a cylinder at any roll about its axis, either end first.
    The hand cannot tell them apart but the policy reads the raw quaternion, so
    every branch is drawn. This is a RIGHT multiplication -- a relabelling of the
    object's own frame.
  * Tilt: a small random rotation in the world frame (a LEFT multiplication),
    because that is a genuinely different pose for the physics to hold or reject.

  One event composes all three from `default_root_state` and writes once. A
  separate follow-up event would read `root_link_pose_w`, which is not refreshed
  by a write earlier in the same reset callback.
  """
  import itertools

  import torch
  from mjlab.utils.lab_api import math as math_utils

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  if n == 0:
    return
  name = "object" if asset_cfg is None else asset_cfg.name
  asset = env.scene[name]

  root = asset.data.default_root_state[env_ids].clone()
  pos, quat = root[:, 0:3].clone(), root[:, 3:7].clone()
  for axis, key in enumerate(("x", "y", "z")):
    lo, hi = (pose_range or {}).get(key, (0.0, 0.0))
    pos[:, axis] += torch.empty(n, device=env.device).uniform_(float(lo), float(hi))
  pos += env.scene.env_origins[env_ids]

  if symmetries == "cylinder":
    # A cylinder goes into the hand at any roll about its own axis (local z) and
    # either end first: a uniform roll, then a 180 deg flip about local x half the
    # time. Right multiplications, like the cube's, so they relabel the object's
    # own frame and leave the resting geometry where the grasp put it.
    roll = torch.rand(n, device=env.device) * 2.0 * math.pi
    z = torch.zeros(n, 3, device=env.device)
    z[:, 2] = 1.0
    quat = math_utils.quat_mul(quat, math_utils.quat_from_angle_axis(roll, z))
    flip = torch.rand(n, device=env.device) < 0.5
    if flip.any():
      x = torch.zeros(int(flip.sum()), 3, device=env.device)
      x[:, 0] = 1.0
      half_turn = torch.full((int(flip.sum()),), math.pi, device=env.device)
      quat[flip] = math_utils.quat_mul(quat[flip], math_utils.quat_from_angle_axis(half_turn, x))
  elif symmetries:
    cache = getattr(env, "_cube_symmetries", None)
    if cache is None:
      mats = []
      for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
          m = torch.zeros(3, 3)
          for i, j in enumerate(perm):
            m[i, j] = signs[i]
          if abs(float(torch.det(m)) - 1.0) < 1e-6:
            mats.append(m)
      cache = math_utils.quat_from_matrix(torch.stack(mats).to(env.device))
      env._cube_symmetries = cache
    pick = torch.randint(0, cache.shape[0], (n,), device=env.device)
    quat = math_utils.quat_mul(quat, cache[pick])

  if max_tilt_rad > 0.0:
    axis_v = torch.randn(n, 3, device=env.device)
    axis_v = axis_v / axis_v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    angle = torch.rand(n, device=env.device) * max_tilt_rad
    quat = math_utils.quat_mul(
      math_utils.quat_from_angle_axis(angle, axis_v), quat)

  asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(
    torch.zeros(n, 6, device=env.device), env_ids=env_ids)


@requires_model_fields("geom_solref")
def randomize_contact_solref(
  env,
  env_ids,
  timeconst_range: tuple[float, float],
  dampratio_range: tuple[float, float] | None = None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  distribution: Literal["uniform", "log_uniform"] = "log_uniform",
) -> None:
  """Randomise the hand's contact time constant (`geom_solref[0]`), one draw per env.

  `mjlab.envs.mdp.dr` has no term for `geom_solref`. Softer contacts were the only
  contact parameter measured to move the sim-to-real gap in how much rotation the
  hand gives back after turning an object; tangential friction was saturated over
  0.5-1.4 and `solimp` inert. The range covers that direction rather than sitting
  on an identified value: log-uniform, stopping at 0.10 s because softer contacts
  let the fingers sink into the object.

  One draw per environment, applied to every selected geom: stiffness is a
  property of the surface pair, and per-geom draws would give each phalanx of one
  finger a different stiffness. Reaches the contact unchanged because the hand
  geoms carry `priority="1"`.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device, dtype=torch.long)
  if env_ids.numel() == 0:
    return

  asset = env.scene[asset_cfg.name]
  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids].long()
  env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")

  def draw(bounds: tuple[float, float]) -> torch.Tensor:
    lo, hi = float(bounds[0]), float(bounds[1])
    u = torch.empty(env_ids.numel(), 1, device=env.device)
    if distribution == "log_uniform":
      return u.uniform_(math.log(lo), math.log(hi)).exp()
    return u.uniform_(lo, hi)

  env.sim.model.geom_solref[env_grid, geom_grid, 0] = draw(timeconst_range)
  if dampratio_range is not None:
    env.sim.model.geom_solref[env_grid, geom_grid, 1] = draw(dampratio_range)
