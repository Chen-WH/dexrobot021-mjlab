"""Rebuild the actor observation vector from hardware measurements.

Every term here mirrors one function in `tasks/inhand/mdp/observations.py`. The
mirroring is the whole risk of the file, so `scripts/deploy/selftest.py` runs this
encoder against a live sim env and asserts the two vectors agree element-wise.

Ordering, scales and sizes all come from the exported `PolicyContract`, never from
constants typed here. The one thing this file encodes is *what each term means*.
"""

from __future__ import annotations

import numpy as np

from .contract import PolicyContract
from .geometry import quat_to_mat
from .tracker import CubeState


class ObservationEncoder:
  """Assembles the actor observation for one hand and one object."""

  def __init__(self, contract: PolicyContract):
    self.contract = contract
    self.soft_lo = contract.soft_limits_array()[:, 0]
    self.soft_hi = contract.soft_limits_array()[:, 1]
    # `object_lin_vel_w` is the one term expressed in the simulator's *world*
    # frame rather than the wrist frame. The hand base sits at a fixed rotation
    # from world in sim, so this is a constant change of basis -- but not the
    # identity: dropping it feeds the policy a velocity rotated 90 degrees.
    self.R_world_hand = quat_to_mat(contract.notes["hand_root_quat_w"])
    self.n = contract.observation_dim

  def encode(
    self,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    cube: CubeState,
    last_action: np.ndarray,
  ) -> np.ndarray:
    """Return the (obs_dim,) float32 vector the policy consumes."""
    parts: dict[str, np.ndarray] = {
      # scale_transform: normalise to [-1, 1] across the soft limits.
      "joint_pos": (
        2.0 * np.asarray(joint_pos, float) - (self.soft_lo + self.soft_hi)
      ) / (self.soft_hi - self.soft_lo),
      # joint_vel_rel: default joint velocity is zero, so this is the raw rate.
      "joint_vel": np.asarray(joint_vel, float),
      "object_pos": cube.pos,
      "object_quat": cube.quat,
      "object_lin_vel": self.R_world_hand @ cube.lin_vel,
      "object_ang_vel": cube.ang_vel,
      "last_action": np.asarray(last_action, float),
    }

    out = np.empty(self.n, np.float32)
    offset = 0
    for term in self.contract.observation_terms:
      value = parts.get(term.name)
      if value is None:
        raise KeyError(
          f"contract wants observation term {term.name!r} but the encoder does not "
          "produce it")
      value = np.asarray(value, float).ravel()
      if value.size != term.size:
        raise ValueError(
          f"term {term.name!r}: encoder made {value.size} values, "
          f"contract says {term.size}"
        )
      value = value * term.scale
      if term.clip is not None:
        value = np.clip(value, term.clip[0], term.clip[1])
      out[offset:offset + term.size] = value
      offset += term.size
    return out
