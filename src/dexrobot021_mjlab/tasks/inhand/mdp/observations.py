"""Observation terms for DexRobot021 in-hand tasks."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api import math as math_utils

from .frames import object_pose_in_wrist_frame


def joint_pos_limit_normalized(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  return math_utils.scale_transform(joint_pos, limits[..., 0], limits[..., 1])


def _extrinsic_bias(env: ManagerBasedRlEnv):
  """The per-environment hand-eye error, or identity before one has been drawn."""
  quat = getattr(env, "extrinsic_bias_quat", None)
  if quat is None:
    ident = torch.zeros(env.num_envs, 4, device=env.device)
    ident[:, 0] = 1.0
    return ident, torch.zeros(env.num_envs, 3, device=env.device)
  return quat, env.extrinsic_bias_pos


def _measured_object_pose(
  env: ManagerBasedRlEnv,
  object_cfg: SceneEntityCfg,
  wrist_cfg: SceneEntityCfg,
  biased: bool,
):
  """The object pose as the deployment measures it, not as the simulator knows it.

  `biased=False` returns the simulator's own answer and is what the critic, the
  rewards and the metrics use: the sensing error belongs on the policy's input,
  not on the target it is scored against.
  """
  pos, quat = object_pose_in_wrist_frame(env, object_cfg, wrist_cfg)
  if not biased:
    return pos, quat
  bias_quat, bias_pos = _extrinsic_bias(env)
  return (math_utils.quat_apply(bias_quat, pos) + bias_pos,
          math_utils.quat_mul(bias_quat, quat))


def object_pos_wrist(
  env: ManagerBasedRlEnv,
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  wrist_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  extrinsic_bias: bool = False,
) -> torch.Tensor:
  pos_wrist, _ = _measured_object_pose(env, object_cfg, wrist_cfg, extrinsic_bias)
  return pos_wrist


def object_quat_wrist(
  env: ManagerBasedRlEnv,
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  wrist_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  make_quat_unique: bool = False,
  extrinsic_bias: bool = False,
) -> torch.Tensor:
  _, quat_wrist = _measured_object_pose(env, object_cfg, wrist_cfg, extrinsic_bias)
  return math_utils.quat_unique(quat_wrist) if make_quat_unique else quat_wrist


def object_lin_vel_w(
  env: ManagerBasedRlEnv,
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
  object_asset = env.scene[object_cfg.name]
  return object_asset.data.root_link_lin_vel_w


def object_ang_vel_wrist(
  env: ManagerBasedRlEnv,
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  wrist_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  object_asset = env.scene[object_cfg.name]
  wrist_asset = env.scene[wrist_cfg.name]
  return math_utils.quat_apply_inverse(
    wrist_asset.data.root_link_quat_w,
    object_asset.data.root_link_ang_vel_w,
  )
