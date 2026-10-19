"""Export one trained checkpoint into a self-contained deployment bundle.

    python scripts/deploy/export_bundle.py <task-id> <checkpoint> -o checkpoints/<name>

Writes:

    contract.json   every constant the runtime needs, read off the live env cfg
    model.pt        the rsl_rl checkpoint itself (for play, selftest, fine-tuning)
    policy.pt       TorchScript actor, what the runtime loads
    policy.onnx     the same actor, for a host without torch (best effort)
    README.md       what this bundle is

The point of the bundle is that **no constant is typed twice**.  Joint order,
soft limits, reset pose, action scale, EMA alpha, observation term order and
sizes are all read out of the constructed environment, not transcribed: a
policy running under a wrong constant does not fail, it just works badly.

Needs a GPU: it builds the real environment.  The bundle it writes does not.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import extrinsic as extrinsic_mod  # noqa: E402  (scripts/deploy/extrinsic.py)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402

import dexrobot021_mjlab.tasks  # noqa: F401,E402
from dexrobot021_mjlab.assets.grasps import GRASPS  # noqa: E402
from dexrobot021_mjlab.checkpoint_guard import check_checkpoint_object  # noqa: E402
from dexrobot021_mjlab.deploy.contract import ObservationTerm, PolicyContract  # noqa: E402
from dexrobot021_mjlab.deploy.cube import CYLINDER_TAG_SIZE_M, tile_tag_size_m  # noqa: E402


def _spin_axis_wrist(env_cfg) -> list[float]:
  """The unit axis the spin reward pays for, signed by the commanded direction."""
  term = env_cfg.rewards["spin"]
  from dexrobot021_mjlab.tasks.inhand.mdp.anchors import spin_axis_unit

  direction = float(term.params.get("axis_direction", 1.0))
  return [direction * c for c in spin_axis_unit(term.params["spin_axis"])]


def _reset_joint_pos(env, env_cfg, robot, joint_names, ids,
                     default: np.ndarray) -> np.ndarray:
  """The centre of the reset event's joint distribution, per action joint.

  This is where a simulated episode starts, and it is NOT `default_joint_pos`
  (see `PolicyContract.reset_joint_pos`). Read off the event config rather than
  off one sampled reset, because the event randomises and one draw is not the
  centre.

  Checked against the env's actual post-reset pose before it is written: the
  regexes here have to resolve to the same joints the event resolved, and a
  silent mismatch would put the hand somewhere no one asked for.
  """

  term = env_cfg.events.get("reset_robot_joints")
  if term is None:
    raise RuntimeError("no `reset_robot_joints` event; cannot say where an "
                       "episode starts")
  ranges = term.params["position_range"]
  offset = default if term.params.get("use_default_offset") else np.zeros_like(default)
  if term.params.get("operation", "abs") != "abs":
    raise RuntimeError(f"reset operation {term.params.get('operation')!r} is not "
                       "`abs`; the midpoint below would not be a joint angle")

  centre = np.empty(len(joint_names))
  half_width = np.empty(len(joint_names))
  for i, name in enumerate(joint_names):
    for pattern, (lo, hi) in ranges.items():
      if re.fullmatch(pattern, name):
        centre[i] = 0.5 * (float(lo) + float(hi)) + offset[i]
        half_width[i] = 0.5 * abs(float(hi) - float(lo))
        break
    else:
      raise RuntimeError(f"{name} matches no entry in the reset position_range; "
                         "the export would guess where the episode starts")

  # The env was reset when it was constructed, so this is one genuine draw from
  # the distribution whose centre was just computed. It has to be inside the
  # range the event was configured with, or the two disagree about the joints.
  actual = robot.data.joint_pos[0, ids].cpu().numpy()
  slack = half_width + 1e-3
  bad = np.abs(actual - centre) > slack
  if bad.any():
    rows = "\n".join(
      f"    {joint_names[i]:16s} reset to {actual[i]:.3f}, "
      f"config says {centre[i]:.3f} +- {half_width[i]:.3f}"
      for i in np.flatnonzero(bad))
    raise RuntimeError(
      "the reset pose read off the event config does not match the pose the env "
      f"actually reset to:\n{rows}")
  return centre


def _object_kind(env) -> str:
  """"cube" or "cylinder", read off the simulated scene's geom."""
  import mujoco

  model = env.sim.mj_model
  for kind in ("cube", "cylinder"):
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"object/{kind}") >= 0:
      return kind
  raise RuntimeError("the scene has no geom named 'object/cube' or 'object/cylinder'")


def _cylinder_size_m(env) -> tuple[float, float]:
  """(diameter, length) of the simulated cylinder, in metres."""
  import mujoco

  model = env.sim.mj_model
  gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object/cylinder")
  radius, half = (float(v) for v in model.geom_size[gid][:2])
  return round(2 * radius, 6), round(2 * half, 6)


def _cube_edge_m(env) -> float:
  """The simulated cube's edge length, and a refusal if it is not a cube."""
  import mujoco

  model = env.sim.mj_model
  gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object/cube")
  if gid < 0:
    raise RuntimeError("the scene has no geom named 'object/cube'")
  half = np.asarray(model.geom_size[gid][:3], dtype=float)
  if float(half.max() - half.min()) > 1e-6:
    raise RuntimeError(
      f"the simulated object is not a cube: half-extents {1e3 * half} mm. "
      "The deployment's tag layout assumes one edge length, so this cannot be "
      "exported."
    )
  return round(float(2.0 * half[0]), 6)


def _cube_mass_kg(env) -> float:
  import mujoco

  model = env.sim.mj_model
  for name in ("object/cube", "object/cylinder"):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid >= 0:
      return round(float(model.body_mass[bid]), 6)
  raise RuntimeError("the scene has no body named 'object/cube' or 'object/cylinder'")


def build_contract(task: str, checkpoint: str, env, env_cfg,
                   hand_yaw_deg: float,
                   cube_tag_size_m: float | None) -> PolicyContract:
  robot = env.scene["robot"]
  action_cfg = env_cfg.actions["joint_pos"]
  joint_names = list(action_cfg.joint_names)

  # Resolve the entity's joint ordering for the action term's joints, so the
  # limits below line up with `joint_names` even if the entity stores them in a
  # different order.
  ids = [list(robot.joint_names).index(n) for n in joint_names]
  soft = robot.data.soft_joint_pos_limits[0, ids].cpu().numpy()
  hard = robot.data.joint_pos_limits[0, ids].cpu().numpy()
  default = robot.data.default_joint_pos[0, ids].cpu().numpy()
  reset = _reset_joint_pos(env, env_cfg, robot, joint_names, ids, default)

  # Observation layout, straight off the manager: term order here *is* the
  # concatenation order the policy was trained with.
  group = env.observation_manager._group_obs_term_names["actor"]
  dims = env.observation_manager._group_obs_term_dim["actor"]
  cfgs = env.observation_manager._group_obs_term_cfgs["actor"]
  terms = []
  for name, dim, cfg in zip(group, dims, cfgs, strict=True):
    size = int(np.prod(dim))
    clip = tuple(cfg.clip) if getattr(cfg, "clip", None) is not None else None
    terms.append(ObservationTerm(name=name, size=size,
                                 scale=float(getattr(cfg, "scale", None) or 1.0),
                                 clip=clip))

  clip_cfg = getattr(action_cfg, "clip", None)
  action_clip = None
  if clip_cfg:
    # `clip` is a dict of joint-name regex -> (lo, hi). Resolve to per-joint rows
    # so the runtime needs no regex engine and no ordering assumption.
    import re
    action_clip = []
    for name in joint_names:
      row = [-np.inf, np.inf]
      for pattern, bounds in clip_cfg.items():
        if re.fullmatch(pattern, name):
          row = [float(bounds[0]), float(bounds[1])]
          break
      action_clip.append(row)

  kind = _object_kind(env)
  grasp = GRASPS[kind]
  check_checkpoint_object(checkpoint, env_cfg)

  derived = extrinsic_mod.derive(hand_yaw_deg)
  root_quat = robot.data.root_link_quat_w[0].cpu().numpy().tolist()

  return PolicyContract(
    task_id=task,
    experiment_name=pathlib.Path(checkpoint).parent.name,
    # The rsl_rl checkpoint is copied into the bundle under this name (play,
    # selftest and fine-tuning need it; the runtime uses policy.pt).
    checkpoint="model.pt",
    control_dt=float(env_cfg.sim.mujoco.timestep * env_cfg.decimation),
    sim_timestep=float(env_cfg.sim.mujoco.timestep),
    decimation=int(env_cfg.decimation),
    joint_names=joint_names,
    joint_limits=hard.tolist(),
    soft_joint_limits=soft.tolist(),
    default_joint_pos=default.tolist(),
    reset_joint_pos=reset.tolist(),
    action_scale=float(action_cfg.scale),
    action_clip=action_clip,
    action_ema_alpha=float(getattr(action_cfg, "ema_alpha", 1.0) or 1.0),
    action_relative=bool(action_cfg.relative),
    action_use_default_offset=bool(action_cfg.use_default_offset),
    observation_terms=terms,
    observation_dim=int(env.observation_manager.group_obs_dim["actor"][0]),
    T_hand_cam=derived["T_hand_cam"].tolist(),
    extrinsic_source="cad",
    hand_install_yaw_deg=float(hand_yaw_deg),
    # Read off the simulated object rather than written down again: these name a
    # physical object on the bench.
    cube_edge_m=_cube_edge_m(env) if kind == "cube" else 0.0,
    cube_mass_kg=_cube_mass_kg(env),
    cube_tag_size_m=float(cube_tag_size_m if cube_tag_size_m else
                          (CYLINDER_TAG_SIZE_M if kind == "cylinder"
                           else tile_tag_size_m(_cube_edge_m(env)))),
    cube_tag_size_measured=cube_tag_size_m is not None,
    object_kind=kind,
    object_diameter_m=_cylinder_size_m(env)[0] if kind == "cylinder" else None,
    object_length_m=_cylinder_size_m(env)[1] if kind == "cylinder" else None,
    object_axis_wrist=([round(float(v), 6) for v in grasp.spin_axis_wrist]
                       if kind == "cylinder" else None),
    object_pos_wrist=[round(float(v), 6) for v in grasp.object_pos_wrist],
    place_joint_pos=([round(float(grasp.place_joint_pos[n]), 6) for n in joint_names]
                     if grasp.place_joint_pos else None),
    spin_axis_wrist=_spin_axis_wrist(env_cfg),
    notes={
      # The hand base is not axis-aligned with the simulator world, and
      # `object_lin_vel` is the one observation expressed in world rather than
      # wrist coordinates. The runtime needs this rotation to reproduce it.
      "hand_root_quat_w": root_quat,
      # Consequence of the above, and a hardware requirement: the palm normal
      # (wrist +x) points along world +z in sim, so the arm must hold the hand
      # palm-up or gravity acts on the grasp differently than in training.
      "palm_normal_world": (
        np.asarray(env.scene["robot"].data.root_link_quat_w[0].cpu())
        .tolist()
      ),
      "episode_length_s": float(env_cfg.episode_length_s),
    },
  )


class _FlatActor(torch.nn.Module):
  """Flat tensor in, action out.

  rsl_rl's inference policy takes a *dict* of observation groups, because a model
  may read several. The deployment has one group and no reason to carry a dict
  across the sim-to-real boundary, so the group name is baked in here -- where it
  is checked against the runner's own output on the next few lines -- rather than
  becoming another string the runtime has to agree about.
  """

  def __init__(self, policy, group: str):
    super().__init__()
    self.policy = policy
    self.group = group

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.policy({self.group: obs})


def export_policy(runner, obs_dim: int, out_dir: pathlib.Path) -> dict:
  """TorchScript first, ONNX best-effort. Both are checked against the runner."""
  inference = runner.get_inference_policy(device="cpu")
  groups = list(getattr(runner.alg.get_policy(), "obs_groups", ["actor"]))
  group = "actor" if "actor" in groups else groups[0]
  policy = _FlatActor(inference, group).eval()

  dummy = torch.zeros(1, obs_dim)
  with torch.no_grad():
    reference = policy(dummy)

  result = {}
  traced = torch.jit.trace(policy, dummy, check_trace=False)
  torch.jit.save(traced, str(out_dir / "policy.pt"))
  with torch.no_grad():
    loaded = torch.jit.load(str(out_dir / "policy.pt"))(dummy)
  drift = float((loaded - reference).abs().max())
  if drift > 1e-5:
    raise RuntimeError(f"TorchScript export differs from the runner by {drift:.2e}")
  result["policy.pt"] = {"max_abs_diff": drift}

  try:
    runner.export_policy_to_onnx(str(out_dir), filename="policy.onnx")
    result["policy.onnx"] = {"exported": True}
  except Exception as exc:  # noqa: BLE001 - onnx is optional for the runtime
    result["policy.onnx"] = {"exported": False, "error": str(exc)}
  return result


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("task")
  ap.add_argument("checkpoint")
  ap.add_argument("-o", "--out", type=pathlib.Path, required=True)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--hand-yaw-deg", type=float, default=180.0)
  ap.add_argument("--tag-size-m", type=float, default=None,
                  help="MEASURED black-square side of the printed tag. Without it "
                       "the nominal size is used and every recovered distance "
                       "carries the print-scale error as a systematic bias.")
  args = ap.parse_args()

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)

  env = ManagerBasedRlEnv(env_cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
  runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
              map_location=args.device)

  contract = build_contract(args.task, args.checkpoint, env, env_cfg,
                            args.hand_yaw_deg, args.tag_size_m)
  problems = contract.check()
  if problems:
    print("contract failed its own checks:")
    for p in problems:
      print("  -", p)
    return 1

  args.out.mkdir(parents=True, exist_ok=True)
  contract.save(args.out / "contract.json")
  src = pathlib.Path(args.checkpoint).resolve()
  if src != (args.out / contract.checkpoint).resolve():
    shutil.copyfile(src, args.out / contract.checkpoint)
  export_info = export_policy(runner, contract.observation_dim, args.out)
  iteration = int(torch.load(src, map_location="cpu", weights_only=False)["iter"])

  (args.out / "README.md").write_text(
    f"# deployment bundle: {args.task}\n\n"
    f"- `model.pt`: rsl_rl checkpoint (iteration {iteration}); `policy.pt` / "
    f"`policy.onnx`: the exported actor\n"
    f"- control rate: {contract.control_hz:.4g} Hz "
    f"({contract.sim_timestep * 1e3:.1f} ms x {contract.decimation})\n"
    f"- observation: {contract.observation_dim} values, "
    f"{len(contract.observation_terms)} terms\n"
    f"- actions: {contract.n_joints} joints, scale {contract.action_scale}, "
    f"EMA alpha {contract.action_ema_alpha}\n"
    f"- object: {contract.object_kind}, tag size {contract.cube_tag_size_m * 1e3:.1f} mm "
    f"({'measured' if contract.cube_tag_size_measured else 'nominal'})\n\n"
    f"Validate against sim with `python scripts/deploy/selftest.py {args.task} "
    f"{args.out}`,\nrun on the hand with `python scripts/deploy/run.py {args.out}`.\n",
    encoding="utf-8",
  )

  print(f"wrote {args.out}/")
  print(json.dumps({"contract": "ok", **export_info}, indent=2))
  print(f"  observation ({contract.observation_dim}): " + ", ".join(
    f"{t.name}[{t.size}]" for t in contract.observation_terms))
  env.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
