"""DexRobot021 (DexHand 021) left-hand MJCF binding for mjlab."""

from __future__ import annotations

import copy

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from dexrobot021_mjlab.paths import dexrobot021_left_mimic_mjcf

# The twelve motors, in the MJCF's actuator order. Each finger's second flexion
# motor drives the PIP (`*_3`); the DIP (`*_4`) follows it through <equality>.
DEXROBOT021_ACTUATED_JOINTS: tuple[str, ...] = (
  "l_f_joint1_1",
  "l_f_joint1_2",
  "l_f_joint1_3",
  "l_f_joint2_1",
  "l_f_joint2_2",
  "l_f_joint2_3",
  "l_f_joint3_2",
  "l_f_joint3_3",
  "l_f_joint4_2",
  "l_f_joint4_3",
  "l_f_joint5_2",
  "l_f_joint5_3",
)

DEXROBOT021_FINGERPAD_BODIES: tuple[str, ...] = (
  "l_f_link1_pad",
  "l_f_link2_pad",
  "l_f_link3_pad",
  "l_f_link4_pad",
  "l_f_link5_pad",
)

# Palm plus proximal phalanges.
DEXROBOT021_PALM_PROXIMAL_BODIES: tuple[str, ...] = ("left_hand_base",) + tuple(
  f"l_f_link{finger}_{link}" for finger in "12345" for link in "12"
)

# Distal phalanges plus pads.
DEXROBOT021_DISTAL_BODIES: tuple[str, ...] = tuple(
  f"l_f_link{finger}_{link}" for finger in "12345" for link in "34"
) + DEXROBOT021_FINGERPAD_BODIES

# Follower joint -> (reference joint, polycoef) exactly as declared by the
# <equality><joint polycoef=...> entries in the MJCF, ascending powers. The reset
# event evaluates these so the written pose sits on the constraint manifold;
# an off-manifold pose makes the equality solver yank the fingers back with a
# one-step torque spike.
_DIP_OF_PIP = (0.0001, 0.6446, 0.2340, -0.0631, 0.0359)
DEXROBOT021_MIMIC_JOINT_POLYNOMIALS: dict[
  str, tuple[str, tuple[float, ...]]
] = {
  "l_f_joint1_4": ("l_f_joint1_3", _DIP_OF_PIP),
  "l_f_joint2_4": ("l_f_joint2_3", _DIP_OF_PIP),
  "l_f_joint3_4": ("l_f_joint3_3", _DIP_OF_PIP),
  "l_f_joint4_1": ("l_f_joint2_1", (0.0, 1.0, 0.0, 0.0, 0.0)),
  "l_f_joint4_4": ("l_f_joint4_3", _DIP_OF_PIP),
  "l_f_joint5_1": ("l_f_joint2_1", (0.0022, 2.1798, -1.0491, 0.0, 0.0)),
  "l_f_joint5_4": ("l_f_joint5_3", _DIP_OF_PIP),
}


def _get_left_spec() -> mujoco.MjSpec:
  path = dexrobot021_left_mimic_mjcf()
  if not path.exists():
    raise FileNotFoundError(f"DexRobot021 MJCF not found: {path}")
  return mujoco.MjSpec.from_file(str(path))


# The hand is welded palm-up. Every value lies on the coupling curves: PIP 0.4705
# gives DIP 0.3504, abduction 0.05 gives the little finger 0.10857. The episode-start
# pose is not this one -- the reset event writes the solved grasp (`grasps.py`).
DEXROBOT021_LEFT_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.43),
  rot=(0.70710678, 0.0, -0.70710678, 0.0),
  joint_pos={
    "l_f_joint1_1": 1.05,
    "l_f_joint1_2": 0.35,
    "l_f_joint1_3": 0.4705,
    "l_f_joint1_4": 0.3504,
    "l_f_joint2_1": 0.05,
    "l_f_joint2_2": 0.35,
    "l_f_joint2_3": 0.4705,
    "l_f_joint2_4": 0.3504,
    "l_f_joint3_2": 0.35,
    "l_f_joint3_3": 0.4705,
    "l_f_joint3_4": 0.3504,
    "l_f_joint4_1": 0.05,
    "l_f_joint4_2": 0.35,
    "l_f_joint4_3": 0.4705,
    "l_f_joint4_4": 0.3504,
    "l_f_joint5_1": 0.10857,
    "l_f_joint5_2": 0.35,
    "l_f_joint5_3": 0.4705,
    "l_f_joint5_4": 0.3504,
  },
  joint_vel={".*": 0.0},
)


DEXROBOT021_LEFT_DCMOTOR_CFG = EntityCfg(
  # No `spec_fn` edits: the MJCF's own twelve `<dcmotor input="position">`
  # actuators are the drive model, and every drive parameter lives in the asset.
  spec_fn=_get_left_spec,
  init_state=DEXROBOT021_LEFT_INIT_STATE,
  articulation=EntityArticulationInfoCfg(
    actuators=(
      # `<dcmotor>` is not one of the actuator types mjlab auto-detects, so the
      # command field is explicit. In `input="position"` mode ctrl is a joint-angle
      # setpoint.
      XmlActuatorCfg(
        target_names_expr=DEXROBOT021_ACTUATED_JOINTS,
        command_field="position",
        # Command latency in physics steps: 1-3 at dt=0.002 is 2-6 ms (a CAN-FD
        # frame plus up to one period of the hand's 150 Hz driver loop).
        delay_min_lag=1,
        delay_max_lag=3,
      ),
    ),
    soft_joint_pos_limit_factor=1.0,
  ),
  sort_actuators=False,
)


def get_dexrobot021_left_dcmotor_cfg() -> EntityCfg:
  return copy.deepcopy(DEXROBOT021_LEFT_DCMOTOR_CFG)
