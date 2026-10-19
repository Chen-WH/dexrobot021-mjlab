"""Frame transform helpers for wrist-frame in-hand terms."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api import math as math_utils


def object_pose_in_wrist_frame(
  env: ManagerBasedRlEnv,
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  wrist_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
  object_asset = env.scene[object_cfg.name]
  wrist_asset = env.scene[wrist_cfg.name]
  return math_utils.subtract_frame_transforms(
    wrist_asset.data.root_link_pos_w,
    wrist_asset.data.root_link_quat_w,
    object_asset.data.root_link_pos_w,
    object_asset.data.root_link_quat_w,
  )
