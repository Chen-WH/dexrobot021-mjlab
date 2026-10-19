"""DexRobot021 in-hand spin task factory.

The hand is welded palm-up and starts every episode already closed around the
object in a solved grasp (`assets/grasps.py`). The policy outputs absolute joint
targets for the twelve motors (EMA-filtered, anchored on the reset pose) and is
paid for the object's rotation rate about one wrist-frame axis, clipped at a target
rate and gated on the object staying where it started.

Observations are what the real rig can measure: joint positions and velocities,
the object pose and velocities from an AprilTag camera (with a per-episode
hand-eye calibration error and measured latency), and the last action. The critic
sees the same quantities without noise, bias or delay.

The recipe follows the in-hand rotation task of ruoyiqiao/mjlab_hand (absolute EMA
targets, a single task reward, a token action-rate term, startup contact and drive
randomisation) adapted to this hand's `<dcmotor>` drive model.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mjlab_mdp
from mjlab.managers import (
  EventTermCfg,
  MetricsTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise
from mjlab.viewer import ViewerConfig

from dexrobot021_mjlab.assets import (
  DEXROBOT021_ACTUATED_JOINTS,
  DEXROBOT021_MIMIC_JOINT_POLYNOMIALS,
  GRASPS,
  Grasp,
  get_cube_cfg,
  get_cylinder_cfg,
  get_dexrobot021_left_dcmotor_cfg,
)

from . import mdp

ObjectName = Literal["cube", "cylinder"]

# Drift gates, measured from the object's pose at reset.
DRIFT_POSITION_THRESHOLD = 0.03
DRIFT_TILT_THRESHOLD = 0.5

# Half-width of the randomisation on the reset joint pose, radians. The grasp's
# finger gaps are half a millimetre; on the bench the hand repeats a commanded pose
# to about a hundredth of a radian. Placement error is carried by `reset_object`.
RESET_JOINT_JITTER = 0.01

EPISODE_LENGTH_S = 20.0


def _active_joint_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    joint_names=DEXROBOT021_ACTUATED_JOINTS,
    preserve_order=True,
  )


def _build_observations(*, play: bool) -> dict[str, ObservationGroupCfg]:
  active_joint_cfg = _active_joint_cfg()
  object_cfg = SceneEntityCfg("object")
  wrist_cfg = SceneEntityCfg("robot")

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_limit_normalized,
      noise=Gnoise(std=0.005),
      params={"asset_cfg": active_joint_cfg},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      scale=0.2,
      noise=Gnoise(std=0.01),
      params={"asset_cfg": active_joint_cfg},
    ),
    "object_pos": ObservationTermCfg(
      func=mdp.object_pos_wrist,
      noise=Gnoise(std=0.002),
      params={"object_cfg": object_cfg, "wrist_cfg": wrist_cfg,
              "extrinsic_bias": not play},
    ),
    "object_quat": ObservationTermCfg(
      func=mdp.object_quat_wrist,
      params={
        "object_cfg": object_cfg,
        "wrist_cfg": wrist_cfg,
        "make_quat_unique": False,
        "extrinsic_bias": not play,
      },
    ),
    "object_lin_vel": ObservationTermCfg(
      func=mdp.object_lin_vel_w,
      noise=Gnoise(std=0.002),
      params={"object_cfg": object_cfg},
    ),
    "object_ang_vel": ObservationTermCfg(
      func=mdp.object_ang_vel_wrist,
      scale=0.2,
      noise=Gnoise(std=0.002),
      params={"object_cfg": object_cfg, "wrist_cfg": wrist_cfg},
    ),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }

  # Observation latency, measured on hardware over 625 closed-loop steps: the
  # object pose reaching the policy is 9 ms old on average and 14 ms at p95 against
  # a 16 ms control period, and the tracker's 3-sample velocity window puts the
  # object velocities further back. The loop reads the hand before it writes, so
  # joint feedback describes the state the previous command produced. Only the
  # actor is delayed; the critic is privileged.
  #
  # `play` means the clean observation contract (no noise, bias or delay), because
  # `scripts/deploy/selftest.py` compares the numpy deployment pipeline against
  # this environment element by element.
  def _delayed(term, lo: int, hi: int):
    if play:
      return term
    return replace(term, delay_min_lag=lo, delay_max_lag=hi, delay_per_env=True)

  for name in ("object_pos", "object_quat"):
    actor_terms[name] = _delayed(actor_terms[name], 0, 1)
  for name in ("object_lin_vel", "object_ang_vel"):
    actor_terms[name] = _delayed(actor_terms[name], 1, 2)
  for name in ("joint_pos", "joint_vel"):
    actor_terms[name] = _delayed(actor_terms[name], 0, 1)

  critic_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_limit_normalized,
      params={"asset_cfg": _active_joint_cfg()},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      scale=0.2,
      params={"asset_cfg": _active_joint_cfg()},
    ),
    "object_pos": ObservationTermCfg(
      func=mdp.object_pos_wrist,
      params={"object_cfg": SceneEntityCfg("object"), "wrist_cfg": SceneEntityCfg("robot")},
    ),
    "object_quat": ObservationTermCfg(
      func=mdp.object_quat_wrist,
      params={
        "object_cfg": SceneEntityCfg("object"),
        "wrist_cfg": SceneEntityCfg("robot"),
        "make_quat_unique": False,
      },
    ),
    "object_lin_vel": ObservationTermCfg(
      func=mdp.object_lin_vel_w,
      params={"object_cfg": SceneEntityCfg("object")},
    ),
    "object_ang_vel": ObservationTermCfg(
      func=mdp.object_ang_vel_wrist,
      scale=0.2,
      params={"object_cfg": SceneEntityCfg("object"), "wrist_cfg": SceneEntityCfg("robot")},
    ),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }

  return {
    "actor": ObservationGroupCfg(
      actor_terms,
      enable_corruption=not play,
      concatenate_terms=True,
    ),
    "critic": ObservationGroupCfg(
      critic_terms,
      enable_corruption=False,
      concatenate_terms=True,
    ),
  }


def _build_actions() -> dict[str, mdp.ReducedJointPositionActionCfg]:
  return {
    "joint_pos": mdp.ReducedJointPositionActionCfg(
      entity_name="robot",
      joint_names=DEXROBOT021_ACTUATED_JOINTS,
      # At scale 1.0 and init_std 1.0 the object is lost in 99 % of early episodes,
      # at 0.5 in 15 %; 0.25 keeps it but limits the policy to +-0.25 rad.
      scale=0.5,
      relative=False,
      use_default_offset=True,
      # Matched to the hand's own command filter: 150 Hz with alpha 0.2 is a 30 ms
      # time constant, which at this env's 62.5 Hz is 1 - exp(-0.016/0.030) = 0.41.
      ema_alpha=0.41,
      preserve_order=True,
    )
  }


def _build_events(grasp: Grasp):
  return {
    "reset_scene_to_default": EventTermCfg(
      func=mjlab_mdp.reset_scene_to_default,
      mode="reset",
    ),
    "reset_object": EventTermCfg(
      func=mdp.reset_object_pose_and_orientation,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("object"),
        # World axes: +z is up (the palm normal). The operator lowers the object
        # in until it rests on the palm, so the vertical error is one-sided and
        # small while the horizontal error is what the grasp leaves room for.
        "pose_range": {"x": (-0.004, 0.004), "y": (-0.004, 0.004),
                       "z": (-0.001, 0.002)},
        "symmetries": "cylinder" if grasp.obj == "cylinder" else True,
        # Tilting the object widens its footprint; against a grasp with
        # half-millimetre gaps 0.10 rad costs ~1.4 mm of interpenetration.
        "max_tilt_rad": 0.10,
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_with_mimic_followers,
      mode="reset",
      params={
        "position_range": {
          name: (lo - RESET_JOINT_JITTER, hi + RESET_JOINT_JITTER)
          for name, (lo, hi) in grasp.joint_ranges.items()
        },
        "velocity_range": {name: (0.0, 0.0) for name in DEXROBOT021_ACTUATED_JOINTS},
        "mimic_relations": DEXROBOT021_MIMIC_JOINT_POLYNOMIALS,
        "use_default_offset": False,
        "operation": "abs",
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    **_contact_events(),
    **_drive_and_sensing_events(),
  }


def _build_rewards(*, spin_axis, spin_axis_direction: float, spin_target_rate: float,
                   joint_speed_limit: float, joint_speed_weight: float,
                   blocked_lead_weight: float | None) -> dict[str, RewardTermCfg]:
  rewards = {
    "spin": RewardTermCfg(
      func=mdp.wrist_object_spin_finite_diff,
      weight=4.0,
      params={
        "object_cfg": SceneEntityCfg("object"),
        "wrist_cfg": SceneEntityCfg("robot"),
        "clip_min": -spin_target_rate,
        "clip_max": spin_target_rate,
        "axis_direction": spin_axis_direction,
        "spin_axis": spin_axis,
        "history_steps": 4,
        "drift_position_threshold": DRIFT_POSITION_THRESHOLD,
        "drift_tilt_threshold": DRIFT_TILT_THRESHOLD,
        "drift_mode": "exp",
        "drift_inside_factor": 1.0,
        "drift_outside_factor": 0.1,
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1.0e-6),
    # Joints above `joint_speed_limit` are where the drive model has no hardware
    # backing and where the hand chatters; the hand cannot cap its own rate.
    "joint_speed_excess": RewardTermCfg(
      func=mdp.joint_speed_excess_l2, weight=joint_speed_weight,
      params={"speed_limit": joint_speed_limit, "asset_cfg": _active_joint_cfg()}),
  }
  if blocked_lead_weight is not None:
    rewards["blocked_lead"] = RewardTermCfg(
      func=mdp.blocked_lead_l2, weight=blocked_lead_weight,
      params={"lead_min_deg": 3.0, "vel_max": 0.3})
  return rewards


def _build_metrics(*, spin_axis, spin_axis_direction: float, spin_target_rate: float,
                   joint_speed_limit: float) -> dict[str, MetricsTermCfg]:
  return {
    "spin_progress": MetricsTermCfg(
      func=mdp.wrist_object_spin_progress,
      params={
        "object_cfg": SceneEntityCfg("object"),
        "wrist_cfg": SceneEntityCfg("robot"),
        "target_spin_rate": spin_target_rate,
        "axis_direction": spin_axis_direction,
        "spin_axis": spin_axis,
        "position_threshold": DRIFT_POSITION_THRESHOLD,
        "tilt_threshold": DRIFT_TILT_THRESHOLD,
      },
    ),
    "joint_speed_over_limit": MetricsTermCfg(
      func=mdp.joint_speed_excess_fraction,
      params={"speed_limit": joint_speed_limit, "asset_cfg": _active_joint_cfg()}),
  }


def _build_terminations(*, play: bool, stall_latch: bool) -> dict[str, TerminationTermCfg]:
  # No drop termination: a dropped object earns nothing through the drift gate,
  # and the episode runs to its time limit.
  terminations: dict[str, TerminationTermCfg] = {}
  if not play:
    terminations["time_out"] = TerminationTermCfg(func=mdp.time_out, time_out=True)
  if stall_latch:
    terminations["stall_latch"] = TerminationTermCfg(
      func=mdp.stall_latch,
      params={"lead_deg": 8.0, "vel_max": 0.3, "steps": 15, "window": 8})
  return terminations


def make_spin_env_cfg(
  *,
  obj: ObjectName,
  play: bool = False,
  spin_axis_direction: float = 1.0,
  spin_target_rate: float = 1.5,
  joint_speed_limit: float = 1.5,
  joint_speed_weight: float = -0.1,
  blocked_lead_weight: float | None = None,
  stall_latch: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Spin `obj` about the axis its grasp holds it on (`Grasp.spin_axis_wrist`).

  `spin_target_rate` (rad/s) is where the reward clips. `blocked_lead_weight` and
  `stall_latch` price pushing on a blocked joint; they are sized against a policy
  that already spins and must be switched on only when warm-starting from one.
  """
  grasp = GRASPS[obj]
  spin_axis = tuple(grasp.spin_axis_wrist)
  spin_kw = {"spin_axis": spin_axis, "spin_axis_direction": spin_axis_direction,
             "spin_target_rate": spin_target_rate}
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=4 if play else 1024,
      env_spacing=0.8,
      entities={
        "robot": get_dexrobot021_left_dcmotor_cfg(),
        "object": get_cylinder_cfg() if obj == "cylinder" else get_cube_cfg(),
      },
      extent=1.5,
    ),
    observations=_build_observations(play=play),
    actions=_build_actions(),
    commands={},
    events=_build_events(grasp),
    rewards=_build_rewards(**spin_kw, joint_speed_limit=joint_speed_limit,
                           joint_speed_weight=joint_speed_weight,
                           blocked_lead_weight=blocked_lead_weight),
    terminations=_build_terminations(play=play, stall_latch=stall_latch),
    curriculum={},
    metrics=_build_metrics(**spin_kw, joint_speed_limit=joint_speed_limit),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="left_hand_base",
      distance=1.0,
      elevation=-20.0,
      azimuth=120.0,
      max_extra_envs=2,
    ),
    sim=SimulationCfg(
      nconmax=256,
      njmax=1100,
      mujoco=MujocoCfg(
        timestep=0.002,
        iterations=20,
        ls_iterations=20,
        impratio=10.0,
        cone="elliptic",
      ),
    ),
    decimation=8,
    episode_length_s=EPISODE_LENGTH_S,
    scale_rewards_by_dt=False,
  )


def _contact_events() -> dict[str, EventTermCfg]:
  """Startup contact randomisation.

  * tangential friction U(0.5, 1.5) around the nominal 1.0;
  * torsional friction U(0.001, 0.02): the least measured coefficient and the one
    spin is most sensitive to;
  * contact time constant (`solref[0]`) log-uniform over (0.02, 0.10) s
    (`mdp.randomize_contact_solref`).

  `shared_random=True`: friction is a property of the surface, and independent
  per-geom draws would put the elementwise max over a contact pair at the top of
  the range in every environment.
  """
  def term(axis: int, lo: float, hi: float) -> EventTermCfg:
    return EventTermCfg(
      mode="startup",
      func=mjlab_mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [axis],
        "ranges": (lo, hi),
        "shared_random": True,
      },
    )

  return {
    "contact_friction_slide": term(0, 0.5, 1.5),
    "contact_friction_torsion": term(1, 0.001, 0.02),
    "contact_solref": EventTermCfg(
      mode="startup",
      func=mdp.randomize_contact_solref,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "timeconst_range": (0.02, 0.10),
        "distribution": "log_uniform",
      },
    ),
  }


def _drive_and_sensing_events() -> dict[str, EventTermCfg]:
  """Drive-parameter and sensing randomisation.

  The no-load speeds pin one combination of stall torque, damping and friction per
  actuator group; loaded hardware bogs down more than the model, hence
  `drive_effort` skewed below 1. `frictionloss` has no independent measurement.
  """
  def term(func, params: dict, distribution: bool = True) -> EventTermCfg:
    base = {"asset_cfg": SceneEntityCfg("robot")}
    if distribution:
      base["distribution"] = "uniform"
    return EventTermCfg(mode="startup", func=func, params={**base, **params})

  return {
    # Hand-eye calibration error, redrawn per episode (`mdp.sample_extrinsic_bias`):
    # up to 4.6 deg and 10 mm.
    "extrinsic_bias": EventTermCfg(
      mode="reset",
      func=mdp.sample_extrinsic_bias,
      params={"max_angle_rad": 0.08, "max_offset_m": 0.010},
    ),
    "drive_frictionloss": term(
      mjlab_mdp.dr.joint_friction,
      {"operation": "scale", "ranges": (0.5, 2.0)},
    ),
    "drive_damping": term(
      mjlab_mdp.dr.joint_damping,
      {"operation": "scale", "ranges": (0.5, 2.0)},
    ),
    # Static joint-angle offsets on the action path: the hardware's joints settle
    # 0.8-3.5 deg off the commanded angle while reading back the commanded value.
    "encoder_bias": term(
      mjlab_mdp.dr.encoder_bias,
      {"bias_range": (-0.06, 0.06)},
      distribution=False,
    ),
    "drive_effort": term(
      mjlab_mdp.dr.effort_limits,
      {"operation": "scale", "effort_limit_range": (0.6, 1.3)},
    ),
  }
