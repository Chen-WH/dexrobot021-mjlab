"""Action terms for reduced-coordinate dexterous-hand control."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import unscale_transform
from mjlab.utils.lab_api.string import resolve_matching_names_values


@dataclass(kw_only=True)
class ReducedJointPositionActionCfg(ActionTermCfg):
  """Position action over the actively controlled joint subset.

  The policy emits one action per active joint. Mimic/follower joints are not
  action dimensions; XML equality constraints and actuators own runtime coupling.
  """

  joint_names: tuple[str, ...] | list[str]
  scale: float | dict[str, float] = 1.0
  offset: float | dict[str, float] = 0.0
  rescale_to_limits: bool = False
  relative: bool = False
  use_default_offset: bool = False
  preserve_order: bool = True
  ema_alpha: float | None = None
  """If set, low-pass the absolute target: `t = a*(offset + action*scale) + (1-a)*t`.

  Only meaningful with `relative=False`, where the target is re-derived from the
  action every step. An absolute target stays anchored to the reset pose, so
  exploration noise perturbs the pose instead of integrating into a random walk
  that throws the object.
  """

  def build(self, env: ManagerBasedRlEnv) -> ReducedJointPositionAction:
    return ReducedJointPositionAction(self, env)


class ReducedJointPositionAction(ActionTerm):
  cfg: ReducedJointPositionActionCfg

  def __init__(self, cfg: ReducedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    self._joint_names = tuple(cfg.joint_names)
    joint_ids, resolved_names = self._entity.find_joints(
      self._joint_names,
      preserve_order=cfg.preserve_order,
    )
    if cfg.preserve_order and tuple(resolved_names) != self._joint_names:
      raise ValueError(
        "Resolved reduced joint order does not match configured action order. "
        f"Expected {self._joint_names}, got {tuple(resolved_names)}."
      )
    self._joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
    self._resolved_joint_names = tuple(resolved_names)
    self._action_dim = len(joint_ids)

    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)
    self._scale = self._parse_scale(cfg.scale)
    self._offset = self._parse_offset(cfg.offset)
    if cfg.use_default_offset:
      self._offset = self._entity.data.default_joint_pos[:, self._joint_ids].clone()
    self._clip = self._parse_clip(cfg.clip)
    self._lower_limits = self._entity.data.soft_joint_pos_limits[
      :, self._joint_ids, 0
    ]
    self._upper_limits = self._entity.data.soft_joint_pos_limits[
      :, self._joint_ids, 1
    ]
    if cfg.relative:
      self._processed_actions[:] = self._entity.data.joint_pos[:, self._joint_ids]
    if cfg.ema_alpha is not None:
      if cfg.relative:
        raise ValueError("ema_alpha is only supported with relative=False.")
      if not 0.0 < cfg.ema_alpha <= 1.0:
        raise ValueError(f"ema_alpha must be in (0, 1], got {cfg.ema_alpha}.")
      self._ema_target = torch.zeros_like(self._raw_actions) + self._offset
      self._processed_actions[:] = self._ema_target

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def processed_actions(self) -> torch.Tensor:
    return self._processed_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    if self.cfg.relative:
      self._processed_actions = self._processed_actions + self._raw_actions * self._scale
      self._processed_actions = torch.clamp(
        self._processed_actions,
        min=self._lower_limits,
        max=self._upper_limits,
      )
    elif self.cfg.rescale_to_limits:
      scaled_actions = (self._raw_actions * self._scale).clamp(-1.0, 1.0)
      self._processed_actions = unscale_transform(
        scaled_actions,
        self._lower_limits,
        self._upper_limits,
      )
    elif self.cfg.ema_alpha is not None:
      target = self._raw_actions * self._scale + self._offset
      alpha = self.cfg.ema_alpha
      self._ema_target = alpha * target + (1.0 - alpha) * self._ema_target
      self._processed_actions = torch.clamp(
        self._ema_target,
        min=self._lower_limits,
        max=self._upper_limits,
      )
    else:
      self._processed_actions = self._raw_actions * self._scale + self._offset

    if self._clip is not None:
      self._processed_actions = torch.clamp(
        self._processed_actions,
        min=self._clip[:, :, 0],
        max=self._clip[:, :, 1],
      )

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._joint_ids]
    self._entity.set_joint_position_target(
      self._processed_actions - encoder_bias,
      joint_ids=self._joint_ids,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    if self.cfg.relative:
      self._processed_actions[env_ids] = self._entity.data.joint_pos[env_ids][
        :, self._joint_ids
      ]
    elif self.cfg.ema_alpha is not None:
      if self.cfg.use_default_offset:
        # Re-anchor on the pose the reset event actually wrote (the solved grasp),
        # not on the asset's `init_state.joint_pos`: otherwise action=0 is a step
        # input that snaps the hand open on the first step. The deployment uses
        # the same origin (`contract.reset_joint_pos`).
        self._offset[env_ids] = self._entity.data.joint_pos[env_ids][
          :, self._joint_ids
        ]
      offset = self._offset
      self._ema_target[env_ids] = (
        offset[env_ids] if isinstance(offset, torch.Tensor) else offset
      )
      self._processed_actions[env_ids] = self._ema_target[env_ids]
    else:
      self._processed_actions[env_ids] = 0.0

  def _parse_scale(self, scale: float | dict[str, float]) -> torch.Tensor | float:
    if isinstance(scale, (float, int)):
      return float(scale)
    if isinstance(scale, dict):
      tensor = torch.ones(self.num_envs, self.action_dim, device=self.device)
      indices, _, values = resolve_matching_names_values(
        scale,
        self._resolved_joint_names,
      )
      tensor[:, indices] = torch.tensor(values, device=self.device)
      return tensor
    raise ValueError(f"Unsupported scale type: {type(scale)}")

  def _parse_offset(self, offset: float | dict[str, float]) -> torch.Tensor | float:
    if isinstance(offset, (float, int)):
      return float(offset)
    if isinstance(offset, dict):
      tensor = torch.zeros(self.num_envs, self.action_dim, device=self.device)
      indices, _, values = resolve_matching_names_values(
        offset,
        self._resolved_joint_names,
      )
      tensor[:, indices] = torch.tensor(values, device=self.device)
      return tensor
    raise ValueError(f"Unsupported offset type: {type(offset)}")

  def _parse_clip(
    self,
    clip: dict[str, tuple[float, float]] | None,
  ) -> torch.Tensor | None:
    if clip is None:
      return None
    tensor = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
      self.num_envs,
      self.action_dim,
      1,
    )
    indices, _, values = resolve_matching_names_values(clip, self._resolved_joint_names)
    tensor[:, indices] = torch.tensor(values, device=self.device)
    return tensor
