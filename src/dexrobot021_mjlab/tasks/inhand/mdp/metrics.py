"""Episode metrics (logged, not rewarded)."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import MetricsTermCfg, SceneEntityCfg
from mjlab.utils.lab_api import math as math_utils

from .anchors import ObjectWristAnchor, spin_axis_unit
from .frames import object_pose_in_wrist_frame


class wrist_object_spin_progress:
  """Per-step task progress in [0, 1] from spin rate and pose stability.

  progress = spin_score * position_score * tilt_score

  A product, not a rate: near the drift gate the position score dominates it.
  Judge spin speed with `scripts/eval.py`, not with this curve.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._object_cfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
    self._wrist_cfg = cfg.params.get("wrist_cfg", SceneEntityCfg("robot"))
    self._anchor = ObjectWristAnchor(
      env,
      self._object_cfg,
      self._wrist_cfg,
      spin_axis=cfg.params.get("spin_axis", 0),
    )
    self._axis_unit = torch.tensor(
      spin_axis_unit(cfg.params.get("spin_axis", 0)), device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    self._anchor.reset(self._env, env_ids)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    wrist_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_spin_rate: float = 0.20,
    axis_direction: float = -1.0,
    spin_axis=0,
    position_threshold: float = 0.03,
    tilt_threshold: float = 0.5,
  ) -> torch.Tensor:
    self._anchor.refresh(env)

    object_asset = env.scene[object_cfg.name]
    wrist_asset = env.scene[wrist_cfg.name]
    ang_vel_wrist = math_utils.quat_apply_inverse(
      wrist_asset.data.root_link_quat_w,
      object_asset.data.root_link_ang_vel_w,
    )
    spin_rate = axis_direction * (ang_vel_wrist @ self._axis_unit)
    spin_score = torch.clamp(
      spin_rate / max(target_spin_rate, 1.0e-6),
      min=0.0,
      max=1.0,
    )

    object_pos_wrist, object_quat_wrist = object_pose_in_wrist_frame(
      env,
      object_cfg,
      wrist_cfg,
    )
    pos_error, tilt_error = self._anchor.errors(object_pos_wrist, object_quat_wrist)
    pos_score = torch.clamp(
      1.0 - pos_error / max(position_threshold, 1.0e-6),
      min=0.0,
      max=1.0,
    )
    tilt_score = torch.clamp(
      1.0 - tilt_error / max(tilt_threshold, 1.0e-6),
      min=0.0,
      max=1.0,
    )
    return spin_score * pos_score * tilt_score
