"""The contract between a trained policy and the hardware that runs it.

Everything the deployment needs to reproduce the training-time observation and
action pipeline, exported once from the env cfg by
`scripts/deploy/export_bundle.py` and read back here, so no constant is ever
typed twice. The hand will happily run a policy whose joint order, action scale or
observation layout silently disagrees with what it was trained under, and the only
symptom is that it works badly.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field, fields

import numpy as np


@dataclass
class ObservationTerm:
  """One slice of the actor observation vector, with its training-time scaling."""

  name: str
  size: int
  scale: float = 1.0
  clip: tuple[float, float] | None = None


@dataclass
class PolicyContract:
  """Frozen description of one trained policy's interface."""

  task_id: str
  experiment_name: str
  checkpoint: str

  # timing
  control_dt: float                 # seconds per policy step
  sim_timestep: float
  decimation: int

  # joints, in the order the policy expects them
  joint_names: list[str]
  joint_limits: list[list[float]]        # hard [lo, hi] per joint, radians
  soft_joint_limits: list[list[float]]   # what `joint_pos_limit_normalized` uses
  default_joint_pos: list[float]

  # action pipeline
  action_scale: float
  action_clip: list[list[float]] | None
  action_ema_alpha: float
  action_relative: bool
  action_use_default_offset: bool

  # observation layout, in concatenation order
  observation_terms: list[ObservationTerm]
  observation_dim: int

  # frames
  T_hand_cam: list[list[float]]     # left_hand_base <- d435i_color_optical
  extrinsic_source: str
  hand_install_yaw_deg: float

  # object
  cube_edge_m: float
  cube_mass_kg: float
  cube_tag_size_m: float
  cube_tag_size_measured: bool

  #: The pose an episode STARTS from: the centre of the reset event's range, i.e.
  #: the solved grasp. It is NOT `default_joint_pos` (the MJCF keyframe, up to
  #: 0.6 rad away). It is also the action origin: the action term re-anchors its
  #: offset on the pose the reset wrote, so `ActionPipeline` adds the scaled
  #: action to THIS pose.
  reset_joint_pos: list[float]

  notes: dict = field(default_factory=dict)

  #: Which object the policy was trained on: "cube" (`cube_edge_m` / `cube_mass_kg`)
  #: or "cylinder" (`object_diameter_m` x `object_length_m`, tags 6 / 7 on the end
  #: faces). The runtime builds its tag geometry from this
  #: (`deploy.cube.object_from_contract`) and, for a cylinder, places with the
  #: open-then-close protocol because the wrap leaves no vertical corridor.
  object_kind: str = "cube"
  object_diameter_m: float | None = None
  object_length_m: float | None = None
  #: The open pose the hand holds while the operator places the object; the
  #: runtime then ramps to `reset_joint_pos` and starts the policy.
  place_joint_pos: list[float] | None = None

  #: Where the object sits in the WRIST frame at episode start, metres. The
  #: operator places by eye, and a centimetre off is enough for the closing ramp to
  #: push a finger into it, so the placement is gated on it.
  object_pos_wrist: list[float] | None = None

  #: The object's own axis in the WRIST frame at episode start, for objects that
  #: have one (the cylinder). `run.py` gates the operator's placement on it: the
  #: roll about it and which end is which are randomised in training, the
  #: direction is not.
  object_axis_wrist: list[float] | None = None

  #: Wrist-frame unit vector the spin task pays for, signed by the commanded
  #: direction: what a rig analysis projects the object's rotation on.
  spin_axis_wrist: list[float] | None = None

  # -- io ---------------------------------------------------------------------

  def save(self, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(self), indent=2) + "\n")

  @classmethod
  def load(cls, path: pathlib.Path | str) -> PolicyContract:
    raw = json.loads(pathlib.Path(path).read_text())
    known = {f.name for f in fields(cls)}
    raw = {k: v for k, v in raw.items() if k in known}
    raw["observation_terms"] = [ObservationTerm(**t) for t in raw["observation_terms"]]
    return cls(**raw)

  # -- derived ----------------------------------------------------------------

  @property
  def control_hz(self) -> float:
    return 1.0 / self.control_dt

  @property
  def n_joints(self) -> int:
    return len(self.joint_names)

  def term_slice(self, name: str) -> slice:
    """Where a named term sits in the concatenated actor observation."""
    offset = 0
    for term in self.observation_terms:
      if term.name == name:
        return slice(offset, offset + term.size)
      offset += term.size
    raise KeyError(f"{name!r} not in {[t.name for t in self.observation_terms]}")

  def soft_limits_array(self) -> np.ndarray:
    return np.asarray(self.soft_joint_limits, float)

  def episode_start_pose(self) -> np.ndarray:
    """Where to put the hand before the episode clock starts."""
    return np.asarray(self.reset_joint_pos, float)

  def T_hand_cam_array(self) -> np.ndarray:
    return np.asarray(self.T_hand_cam, float)

  def check(self) -> list[str]:
    """Internal consistency. Cheap, and run at every load."""
    problems = []
    total = sum(t.size for t in self.observation_terms)
    if total != self.observation_dim:
      problems.append(
        f"observation terms sum to {total}, dim says {self.observation_dim}")
    for name, seq in (("joint_limits", self.joint_limits),
                      ("soft_joint_limits", self.soft_joint_limits),
                      ("default_joint_pos", self.default_joint_pos),
                      ("reset_joint_pos", self.reset_joint_pos)):
      if len(seq) != self.n_joints:
        problems.append(
          f"{name} has {len(seq)} entries for {self.n_joints} joints")
    for i, (lo, hi) in enumerate(self.soft_joint_limits):
      if not hi > lo:
        problems.append(
          f"soft limit {i} ({self.joint_names[i]}) is empty: [{lo}, {hi}]")
    if not 0.0 < self.action_ema_alpha <= 1.0:
      problems.append(f"action_ema_alpha {self.action_ema_alpha} outside (0, 1]")
    T = self.T_hand_cam_array()
    if not np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6):
      problems.append("T_hand_cam rotation block is not orthonormal")
    return problems
