"""Reward terms for in-hand spin."""

from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg, SceneEntityCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api import math as math_utils

from .anchors import ObjectWristAnchor, drift_factor, spin_axis_unit
from .frames import object_pose_in_wrist_frame


class wrist_object_spin_finite_diff(ManagerTermBase):
  """Signed finite-difference spin rate about a wrist-frame axis, gated on drift.

  `spin_axis` is a wrist axis index (0 = palm normal x, 1 = across the finger row
  y, 2 = along the fingers z) or any wrist-frame 3-vector
  (`anchors.spin_axis_unit`); the rate is the rotation-vector component along it,
  averaged over `history_steps` control steps.

  The rate is clipped symmetrically rather than normalised against a deadband, so
  turning the wrong way is penalised by this same term and any forward progress
  earns reward immediately. `clip_max` is the rotation speed the task asks for:
  nothing is paid for spinning faster, which keeps the policy away from the
  bang-bang joint motion the hardware does not survive.

  The result is multiplied by `drift_factor`: once the object has moved more than
  `drift_position_threshold` from where it started the episode, or its spin axis
  has tilted by more than `drift_tilt_threshold`, the reward decays towards
  `drift_outside_factor`. Holding on to the object is paid for inside this term.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    history_steps = int(cfg.params.get("history_steps", 1))
    if history_steps < 1:
      raise ValueError(f"history_steps must be >= 1, got {history_steps}")
    self.cfg = cfg
    self.history_steps = history_steps
    self._env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    self._prev_quat_wrist = torch.zeros(env.num_envs, 4, device=env.device)
    self._prev_quat_wrist[:, 0] = 1.0
    self._spin_rate_hist = torch.zeros(env.num_envs, history_steps, device=env.device)
    self._hist_ptr = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._hist_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._anchor = ObjectWristAnchor(
      env,
      object_cfg=cfg.params.get("object_cfg", SceneEntityCfg("object")),
      wrist_cfg=cfg.params.get("wrist_cfg", SceneEntityCfg("robot")),
      spin_axis=cfg.params.get("spin_axis", 0),
    )
    self._axis_unit = torch.tensor(
      spin_axis_unit(cfg.params.get("spin_axis", 0)), device=env.device)
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    _, object_quat_wrist = object_pose_in_wrist_frame(
      self._env,
      object_cfg=self.cfg.params.get("object_cfg", SceneEntityCfg("object")),
      wrist_cfg=self.cfg.params.get("wrist_cfg", SceneEntityCfg("robot")),
    )
    self._prev_quat_wrist[env_ids] = object_quat_wrist[env_ids]
    self._spin_rate_hist[env_ids] = 0.0
    self._hist_ptr[env_ids] = 0
    self._hist_count[env_ids] = 0
    self._anchor.reset(self._env, env_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    wrist_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip_min: float = -0.25,
    clip_max: float = 0.25,
    axis_direction: float = -1.0,
    spin_axis=0,
    history_steps: int = 1,
    drift_position_threshold: float = 0.03,
    drift_tilt_threshold: float = 0.5,
    drift_mode: str = "exp",
    drift_inside_factor: float = 1.0,
    drift_outside_factor: float = 0.1,
  ) -> torch.Tensor:
    if history_steps != self.history_steps:
      raise ValueError(
        f"history_steps={history_steps} does not match initialized "
        f"value {self.history_steps}."
      )

    object_pos_wrist, object_quat_wrist = object_pose_in_wrist_frame(
      env,
      object_cfg=object_cfg,
      wrist_cfg=wrist_cfg,
    )

    restarted = env.episode_length_buf <= 1
    if torch.any(restarted):
      self.reset(restarted.nonzero(as_tuple=False).squeeze(-1))

    delta_quat = math_utils.quat_mul(
      object_quat_wrist,
      math_utils.quat_conjugate(self._prev_quat_wrist),
    )
    delta_axis_angle = math_utils.axis_angle_from_quat(delta_quat)
    self._prev_quat_wrist[:] = object_quat_wrist

    spin_rate = (
      axis_direction * (delta_axis_angle @ self._axis_unit) / max(env.step_dt, 1.0e-6)
    )
    hist_ptr = self._hist_ptr
    self._spin_rate_hist[self._env_ids, hist_ptr] = spin_rate
    self._hist_ptr = (hist_ptr + 1) % self.history_steps
    self._hist_count = torch.clamp(self._hist_count + 1, max=self.history_steps)

    hist_len = torch.clamp(self._hist_count, min=1).to(spin_rate.dtype)
    spin_rate = self._spin_rate_hist.sum(dim=-1) / hist_len
    spin_reward = torch.clamp(spin_rate, min=clip_min, max=clip_max)

    pos_error, tilt_error = self._anchor.errors(object_pos_wrist, object_quat_wrist)
    gate = drift_factor(
      pos_error,
      tilt_error,
      position_threshold=drift_position_threshold,
      tilt_threshold=drift_tilt_threshold,
      mode=drift_mode,
      inside_factor=drift_inside_factor,
      outside_factor=drift_outside_factor,
    )
    return spin_reward * gate


def joint_speed_excess_l2(
  env: ManagerBasedRlEnv,
  speed_limit: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """`sum(relu(|qdot| - speed_limit)^2)` over the selected joints.

  Zero below the limit, so it does not tax the slow, deliberate motion the task
  needs; only the bang-bang regime above it. The hand's SDK cannot cap joint speed,
  so the rate has to be a property of the policy.
  """
  asset = env.scene[asset_cfg.name]
  qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.square(torch.relu(qd.abs() - speed_limit)).sum(dim=-1)


def joint_speed_excess_fraction(
  env: ManagerBasedRlEnv,
  speed_limit: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Fraction of the selected joints above `speed_limit` this step (a metric)."""
  asset = env.scene[asset_cfg.name]
  qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return (qd.abs() > speed_limit).float().mean(dim=-1)


def blocked_lead_l2(
  env: ManagerBasedRlEnv,
  lead_min_deg: float = 3.0,
  vel_max: float = 0.3,
  action_name: str = "joint_pos",
) -> torch.Tensor:
  """`sum_j relu(|target_j - q_j| - lead_min)^2` over joints that are NOT moving (rad^2).

  A joint whose target leads it while it stands still is being held by something
  -- here the object. In simulation that costs nothing: the actuator stalls for as
  long as the policy likes. On the hand the motor board latches a fault after
  roughly 0.2-1 s of current without motion. Moving joints are not charged, so
  reaching toward a target is free; only pushing on a blocked joint is.
  """
  term = env.action_manager.get_term(action_name)
  asset = term._entity
  q = asset.data.joint_pos[:, term._joint_ids]
  qd = asset.data.joint_vel[:, term._joint_ids]
  lead = (term.processed_actions - q).abs()
  still = (qd.abs() < vel_max).float()
  return torch.square(torch.relu(lead - math.radians(lead_min_deg)) * still).sum(dim=-1)
