"""Episode-reset pose anchors for wrist-frame in-hand terms."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api import math as math_utils

from .frames import object_pose_in_wrist_frame


def spin_axis_unit(spin_axis) -> tuple[float, float, float]:
  """A spin axis as a wrist-frame unit vector.

  `spin_axis` is either a wrist-axis index (0 = palm normal x, 1 = across the
  finger row y, 2 = along the fingers z) or any wrist-frame 3-vector (the
  cylinder's axis as it lies in the grasp, `Grasp.spin_axis_wrist`).
  """
  if isinstance(spin_axis, (int, float)) and not isinstance(spin_axis, bool):
    unit = [0.0, 0.0, 0.0]
    unit[int(spin_axis)] = 1.0
    return tuple(unit)
  v = [float(c) for c in spin_axis]
  if len(v) != 3:
    raise ValueError(f"spin_axis must be an index or a 3-vector, got {spin_axis!r}")
  n = sum(c * c for c in v) ** 0.5
  if n < 1e-6:
    raise ValueError(f"spin_axis {spin_axis!r} has no direction")
  return tuple(c / n for c in v)


class ObjectWristAnchor:
  """Object wrist-frame pose captured at the start of each episode.

  Drift gates measure against this instead of a hand-authored target, so they
  stay reachable wherever the object actually comes to rest.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    wrist_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    spin_axis=0,
  ):
    self._object_cfg = object_cfg
    self._wrist_cfg = wrist_cfg
    self.pos = torch.zeros(env.num_envs, 3, device=env.device)
    self.quat = torch.zeros(env.num_envs, 4, device=env.device)
    self.quat[:, 0] = 1.0
    self._axis_unit = torch.tensor(
      spin_axis_unit(spin_axis), device=env.device).repeat(env.num_envs, 1)
    self.reset(env, None)

  def reset(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    if env_ids is None:
      env_ids = slice(None)
    pos, quat = object_pose_in_wrist_frame(env, self._object_cfg, self._wrist_cfg)
    self.pos[env_ids] = pos[env_ids]
    self.quat[env_ids] = quat[env_ids]

  def refresh(self, env: ManagerBasedRlEnv) -> None:
    """Re-anchor environments whose episode has just restarted."""
    restarted = env.episode_length_buf <= 1
    if torch.any(restarted):
      self.reset(env, restarted.nonzero(as_tuple=False).squeeze(-1))

  def errors(
    self,
    pos_wrist: torch.Tensor,
    quat_wrist: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Position drift and off-axis tilt relative to the anchored pose.

    Tilt is the swing angle of the spin axis: the angle between the object's body
    `spin_axis` now and at reset. Unlike the other two components of an axis-angle
    vector it is exactly invariant to the twist the spin reward is paying for, so an
    object that has turned half a revolution does not read as tilted. `spin_axis`
    must therefore match the axis the spin reward pays for.
    """
    pos_error = torch.norm(pos_wrist - self.pos, dim=-1)
    rel_quat = math_utils.quat_mul(quat_wrist, math_utils.quat_conjugate(self.quat))
    swung = math_utils.quat_apply(rel_quat, self._axis_unit)
    tilt_error = torch.acos(
      torch.clamp((swung * self._axis_unit).sum(dim=-1), min=-1.0, max=1.0)
    )
    return pos_error, tilt_error


def drift_factor(
  pos_error: torch.Tensor,
  tilt_error: torch.Tensor,
  *,
  position_threshold: float,
  tilt_threshold: float,
  mode: str = "exp",
  inside_factor: float = 1.0,
  outside_factor: float = 0.1,
  decay_k_pos: float = 30.0,
  decay_k_tilt: float = 3.0,
) -> torch.Tensor:
  """Multiplicative gate that suppresses reward once the object has drifted."""
  if mode == "step":
    in_bounds = (pos_error <= position_threshold) & (tilt_error <= tilt_threshold)
    return torch.where(
      in_bounds,
      torch.full_like(pos_error, inside_factor),
      torch.full_like(pos_error, outside_factor),
    )
  if mode == "exp":
    pos_excess = torch.clamp(pos_error - position_threshold, min=0.0)
    tilt_excess = torch.clamp(tilt_error - tilt_threshold, min=0.0)
    factor = torch.exp(-decay_k_pos * pos_excess - decay_k_tilt * tilt_excess)
    return torch.clamp(factor, min=outside_factor, max=inside_factor)
  raise ValueError(f"Unknown drift mode {mode!r}. Expected 'step' or 'exp'.")
