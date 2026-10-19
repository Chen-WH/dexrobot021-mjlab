"""Run an exported policy and turn its output into a joint position command.

The action path is `raw -> clip -> scale -> offset -> EMA -> joint limits`, which
is `mdp.ReducedJointPositionActionCfg` with `scale=0.5`, `relative=False`,
`use_default_offset=True`, `ema_alpha=0.41`.

`ema_alpha=0.41` is what the hand's own command filter does (150 Hz at alpha 0.2
in `pyzlg_dexhand`, re-expressed at this loop's 62.5 Hz). Applying it here AND
leaving the hand's filter on would apply it twice, so the driver says which side
owns it (`applies_command_filter`).

**The offset is the episode's start pose, not `default_joint_pos`.** The sim
action term re-anchors its offset and EMA state on the joint pose the reset event
wrote, so in training `action = 0` means "hold the start pose".
"""

from __future__ import annotations

import pathlib

import numpy as np

from .contract import PolicyContract


class Policy:
  """TorchScript actor, driven from numpy."""

  def __init__(self, path: pathlib.Path | str, contract: PolicyContract,
               device: str = "cpu"):
    import torch  # deferred: the runtime host may have no CUDA build

    self.torch = torch
    self.device = torch.device(device)
    self.module = torch.jit.load(str(path), map_location=self.device).eval()
    self.contract = contract

    with torch.no_grad():
      probe = torch.zeros(1, contract.observation_dim, device=self.device)
      out = self.module(probe)
    if out.shape[-1] != contract.n_joints:
      raise ValueError(
        f"policy emits {out.shape[-1]} actions, contract has "
        f"{contract.n_joints} joints -- wrong checkpoint for this task"
      )

  def __call__(self, observation: np.ndarray) -> np.ndarray:
    t = self.torch.from_numpy(np.asarray(observation, np.float32)).unsqueeze(0)
    with self.torch.no_grad():
      action = self.module(t.to(self.device))
    return action.squeeze(0).cpu().numpy().astype(np.float64)


class ActionPipeline:
  """Raw policy output -> absolute joint position targets, in radians."""

  def __init__(self, contract: PolicyContract, apply_ema: bool = True):
    self.c = contract
    self.apply_ema = bool(apply_ema)
    self.origin = contract.episode_start_pose()
    self.limits = np.asarray(contract.joint_limits, float)
    self.clip = (np.asarray(contract.action_clip, float)
                 if contract.action_clip is not None else None)
    self.reset()

  def reset(self, origin: np.ndarray | None = None) -> None:
    """Start an episode. `origin` is the pose the hand starts from; the sim uses
    the exact pose its reset event wrote, hardware uses `reset_joint_pos` (where
    `approach_default` leaves the hand; the training jitter is 0.01 rad)."""
    if origin is not None:
      self.origin = np.asarray(origin, float).copy()
    self._filtered = self.origin.copy()
    self._last_raw = np.zeros(self.c.n_joints)

  @property
  def last_action(self) -> np.ndarray:
    """What `last_action` in the observation must report: the *raw* policy output,
    as `mdp.last_action` does -- not the filtered target."""
    return self._last_raw.copy()

  def step(self, raw_action: np.ndarray,
           measured_pos: np.ndarray | None = None) -> np.ndarray:
    del measured_pos  # the command does not depend on the measured pose
    raw = np.asarray(raw_action, float)
    self._last_raw = raw.copy()
    if self.clip is not None:
      raw = np.clip(raw, self.clip[:, 0], self.clip[:, 1])

    target = raw * self.c.action_scale
    if self.c.action_use_default_offset:
      target = target + self.origin
    if self.c.action_relative:
      target = self._filtered + target

    alpha = self.c.action_ema_alpha if self.apply_ema else 1.0
    self._filtered = alpha * target + (1.0 - alpha) * self._filtered
    return np.clip(self._filtered, self.limits[:, 0], self.limits[:, 1])
