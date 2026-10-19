from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnv


class stall_latch:
  """End the episode on a sustained push against a blocked joint.

  The hand's motor boards latch a fault after a short stall: current flows while
  the joint does not move. In simulation a stalled actuator can push forever at no
  cost, so a policy may learn to hold the object by driving a blocked joint's
  target to its limit. This ends the episode when some joint's target leads it by
  more than `lead_deg` while its net motion over the last `window` steps is below
  `vel_max`, for `steps` consecutive steps. The criterion is a proxy fitted to rig
  traces; `blocked_lead_l2` (a reward term) carries most of the gradient.

  Class-based so the per-joint streak and position history reset with the episode.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._env = env
    p = cfg.params
    self.window = int(p.get("window", 8))
    term = env.action_manager.get_term(p.get("action_name", "joint_pos"))
    n = len(term._joint_ids)
    self._hist = torch.zeros(env.num_envs, self.window + 1, n, device=env.device)
    self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._run = torch.zeros(env.num_envs, n, dtype=torch.long, device=env.device)

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = slice(None)
    self._count[env_ids] = 0
    self._run[env_ids] = 0

  def __call__(self, env: ManagerBasedRlEnv, lead_deg: float = 8.0, vel_max: float = 0.3,
               steps: int = 15, window: int = 8, action_name: str = "joint_pos") -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    q = term._entity.data.joint_pos[:, term._joint_ids]
    self._hist = torch.roll(self._hist, -1, dims=1)
    self._hist[:, -1] = q
    self._count += 1
    progress = (self._hist[:, -1] - self._hist[:, 0]).abs()
    stuck = ((term.processed_actions - q).abs() > math.radians(lead_deg)) \
      & (progress < vel_max * window * env.step_dt) \
      & (self._count > window).unsqueeze(-1)
    self._run = torch.where(stuck, self._run + 1, torch.zeros_like(self._run))
    return (self._run >= steps).any(dim=-1)
