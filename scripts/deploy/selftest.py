"""Check the deployment stack against the simulator it was exported from.

    python scripts/deploy/selftest.py <task-id> <bundle-dir>

`src/dexrobot021_mjlab/deploy/` reimplements the observation and action pipeline
in numpy so it can run on a machine with no mjlab, no warp and no GPU. This drives
a real environment and, at every step, rebuilds the actor observation from what
the hardware would measure and compares it element-wise with what the environment
produced. The object pose is turned into synthetic AprilTag detections that go
back through `Cube.pose_from_detections` and `CubeTracker`, and the joint targets
are compared with the sim action term's `processed_actions`.

What this can and cannot catch: the detections are synthesised with the bundle's
own extrinsic and tag layout and then fused with the same two, so an error in
either cancels exactly. This verifies term order and slicing, the tracker's hold
and velocity logic, the action path and the exported network. The tag layout on
the physical object has to be checked on the rig (`preflight.py --camera`).

Two terms are expected to differ and are reported rather than asserted:
`object_lin_vel` and `object_ang_vel`, which the hardware differentiates from the
pose stream.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402

import dexrobot021_mjlab.tasks  # noqa: F401,E402
from dexrobot021_mjlab.deploy.contract import PolicyContract  # noqa: E402
from dexrobot021_mjlab.deploy.cube import Cube, TagObservation, object_from_contract  # noqa: E402
from dexrobot021_mjlab.deploy.geometry import se3, se3_inv  # noqa: E402
from dexrobot021_mjlab.deploy.observation import ObservationEncoder  # noqa: E402
from dexrobot021_mjlab.deploy.policy import ActionPipeline, Policy  # noqa: E402
from dexrobot021_mjlab.deploy.tracker import CubeTracker  # noqa: E402

# Terms whose hardware value is derived rather than measured, so an exact match
# is not expected. Reported with their residual instead of asserted.
DERIVED_TERMS = {"object_lin_vel", "object_ang_vel"}


def synth_camera(contract, T_hand_object: np.ndarray) -> np.ndarray:
  """The wrist-frame camera pose the synthetic chain uses.

  The bundle's own `T_hand_cam` for the cube. It cancels inside this test, so for
  the cylinder -- whose end-face tags the palm camera sees edge-on -- the chain is
  hung 0.4 m out along one end's outward normal instead.
  """
  T = contract.T_hand_cam_array()
  if getattr(contract, "object_kind", "cube") != "cylinder":
    return T
  axis = np.asarray(T_hand_object, float)[:3, :3] @ np.array([0.0, 0.0, 1.0])
  centre = np.asarray(T_hand_object, float)[:3, 3] + 0.5 * contract.object_length_m * axis
  eye = centre + 0.4 * axis
  z = (centre - eye) / np.linalg.norm(centre - eye)   # optical +z looks at the tag
  x = np.cross([1.0, 0.0, 0.0], z)
  x /= np.linalg.norm(x)
  out = np.eye(4)
  out[:3, :3] = np.column_stack([x, np.cross(z, x), z])
  out[:3, 3] = eye
  return out


def synth_detections(T_hand_cube: np.ndarray, T_hand_cam: np.ndarray,
                     cube: Cube) -> list[TagObservation]:
  """What the detector would report, given an object pose in the wrist frame."""
  T_cam_cube = se3_inv(T_hand_cam) @ T_hand_cube
  out = []
  for tag_id, T_cube_tag in cube.T_cube_tag.items():
    T_cam_tag = T_cam_cube @ T_cube_tag
    out.append(TagObservation(tag_id, T_cam_tag[:3, :3], T_cam_tag[:3, 3],
                              decision_margin=60.0, z_points_outward=False))
  return out


def _term_diff(name: str, deploy: np.ndarray, sim: np.ndarray):
  """(element-wise difference, rotation angle in degrees or None).

  `q` and `-q` are the same orientation and the two sides pick their branch
  independently, so quaternion terms are sign-aligned first and the geodesic
  angle is reported alongside.
  """
  if name != "object_quat":
    return float(np.abs(deploy - sim).max()), None
  qd, qs = np.array(deploy, float), np.array(sim, float)
  if float(qd @ qs) < 0.0:
    qd = -qd
  dot = abs(float(qd @ qs)) / max(np.linalg.norm(qd) * np.linalg.norm(qs), 1e-12)
  angle = float(np.degrees(2.0 * np.arccos(min(dot, 1.0))))
  return float(np.abs(qd - qs).max()), angle


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("task")
  ap.add_argument("bundle", type=pathlib.Path)
  ap.add_argument("--steps", type=int, default=200)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--velocity-window", type=int, default=3,
                  help="tracker finite-difference window, in control steps")
  ap.add_argument("--tol", type=float, default=1e-4,
                  help="element-wise tolerance on the measured observation terms")
  args = ap.parse_args()

  contract = PolicyContract.load(args.bundle / "contract.json")
  problems = contract.check()
  if problems:
    print("contract failed its checks:", *problems, sep="\n  ")
    return 1
  if contract.task_id != args.task:
    print(f"bundle was exported for {contract.task_id!r}, not {args.task!r}; "
          "shapes match across tasks, so this is refused.")
    return 1

  cube = object_from_contract(contract)
  layout_problems = cube.verify_layout()
  if layout_problems:
    print(f"{contract.object_kind} layout failed its own check:", *layout_problems, sep="\n  ")
    return 1

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(env_cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
  runner = MjlabOnPolicyRunner(wrapped, asdict(load_rl_cfg(args.task)),
                               device=args.device)
  runner.load(str(args.bundle / contract.checkpoint), load_cfg={"actor": True},
              strict=True, map_location=args.device)
  sim_policy = runner.get_inference_policy(device=args.device)

  encoder = ObservationEncoder(contract)
  deploy_policy = Policy(args.bundle / "policy.pt", contract, device="cpu")
  pipeline = ActionPipeline(contract, apply_ema=True)
  # Hung off the object's own pose at reset, so the cylinder's tags are in view.
  _r, _o = env.scene["robot"], env.scene["object"]
  T_hand_cam_synth = synth_camera(contract, se3_inv(
    se3(_r.data.root_link_pos_w[0].cpu().numpy(), _r.data.root_link_quat_w[0].cpu().numpy())) @ se3(
    _o.data.root_link_pos_w[0].cpu().numpy(), _o.data.root_link_quat_w[0].cpu().numpy()))
  tracker = CubeTracker(cube, T_hand_cam_synth, contract.control_dt,
                        velocity_window=args.velocity_window)

  robot, obj = env.scene["robot"], env.scene["object"]
  joint_ids = [list(robot.joint_names).index(n) for n in contract.joint_names]

  action_term = env.action_manager.get_term("joint_pos")
  obs, _ = env.reset()
  # The sim anchors the action offset on the pose its reset event wrote (jittered
  # by 0.01 rad); the pipeline has to use that same pose to agree to 1e-5.
  pipeline.reset(origin=robot.data.joint_pos[0, joint_ids].cpu().numpy())
  # Self-check row: anchoring on `default_joint_pos` instead is wrong by up to
  # 0.6 rad, and the target comparison must see it.
  wrong = ActionPipeline(contract, apply_ema=True)
  wrong.reset(origin=np.asarray(contract.default_joint_pos, float))
  target_diff = wrong_diff = 0.0
  n_held = 0
  worst = {t.name: 0.0 for t in contract.observation_terms}
  worst_deg = {t.name: 0.0 for t in contract.observation_terms}
  achieved: list[float] = []
  policy_diff = 0.0
  n_compared = 0

  for step in range(args.steps):
    actor = obs["actor"][0].cpu().numpy()

    # --- what hardware could measure -----------------------------------------
    joint_pos = robot.data.joint_pos[0, joint_ids].cpu().numpy()
    joint_vel = robot.data.joint_vel[0, joint_ids].cpu().numpy()
    T_w_hand = se3(robot.data.root_link_pos_w[0].cpu().numpy(),
                   robot.data.root_link_quat_w[0].cpu().numpy())
    T_w_cube = se3(obj.data.root_link_pos_w[0].cpu().numpy(),
                   obj.data.root_link_quat_w[0].cpu().numpy())
    T_hand_cube = se3_inv(T_w_hand) @ T_w_cube

    detections = synth_detections(T_hand_cube, T_hand_cam_synth, cube)
    state = tracker.update(detections, step * contract.control_dt)
    deploy_obs = encoder.encode(joint_pos=joint_pos, joint_vel=joint_vel, cube=state,
                                last_action=pipeline.last_action)

    # --- compare --------------------------------------------------------------
    # A step where every tag was too oblique is one where the tracker HOLDS the last
    # pose by design; those are counted, not compared.
    if state.age_steps > 0:
      n_held += 1
    elif step > args.velocity_window + 1:  # let the tracker's velocity window fill
      n_compared += 1
      for term in contract.observation_terms:
        sl = contract.term_slice(term.name)
        d, s = _term_diff(term.name, deploy_obs[sl], actor[sl])
        worst[term.name] = max(worst[term.name], d)
        if s is not None:
          worst_deg[term.name] = max(worst_deg[term.name], s)
      deploy_action = deploy_policy(actor)   # same input, so only the net differs
      with torch.no_grad():
        sim_action = sim_policy(obs)[0].cpu().numpy()
      policy_diff = max(policy_diff, float(np.abs(deploy_action - sim_action).max()))

    with torch.no_grad():
      action = sim_policy(obs)
    target = pipeline.step(action[0].cpu().numpy())
    wrong_target = wrong.step(action[0].cpu().numpy())
    achieved.append(float(np.abs(joint_vel).max()))
    obs, _, dones, _, _ = env.step(action)
    if bool(dones[0]):
      tracker.reset()
      pipeline.reset(origin=robot.data.joint_pos[0, joint_ids].cpu().numpy())
      wrong.reset()
    else:
      sim_target = action_term.processed_actions[0].cpu().numpy()
      target_diff = max(target_diff, float(np.abs(target - sim_target).max()))
      wrong_diff = max(wrong_diff, float(np.abs(wrong_target - sim_target).max()))

  # --- report ----------------------------------------------------------------
  print(f"\n{args.task}\n{contract.observation_dim} observations, "
        f"{n_compared} steps compared ({n_held} held: no tag inside the obliquity "
        f"gate, so the tracker held its last pose), tolerance {args.tol:g}\n")
  failed = []
  for term in contract.observation_terms:
    diff = worst[term.name]
    if term.name in DERIVED_TERMS:
      verdict = "derived"
    elif diff <= args.tol:
      verdict = "ok"
    else:
      verdict = "FAIL"
      failed.append(term.name)
    extra = f"   rotation {worst_deg[term.name]:.3f} deg" if term.name == "object_quat" else ""
    print(f"  {term.name:16s} [{term.size:2d}]  "
          f"max |deploy - sim| = {diff:.3e}   {verdict}{extra}")
  print(f"\n  policy output    max |deploy - sim| = {policy_diff:.3e}   "
        f"{'ok' if policy_diff <= 1e-5 else 'FAIL'}")
  if policy_diff > 1e-5:
    failed.append("policy output")
  target_ok = target_diff <= 1e-5
  print(f"  joint target     max |deploy - sim| = {target_diff:.3e} rad   "
        f"{'ok' if target_ok else 'FAIL'}")
  print(f"  must FAIL: default_joint_pos origin  max |deploy - sim| = "
        f"{np.degrees(wrong_diff):.1f} deg   "
        f"{'FAIL (as it must)' if wrong_diff > 1e-3 else 'ok -- CHECK IS BLIND'}")
  print(f"  tracker: {tracker.stats}")
  print(f"  joint rate achieved in sim, rad/s: median {np.median(achieved):.2f}  "
        f"p95 {np.percentile(achieved, 95):.2f}  max {max(achieved):.2f}")

  if not target_ok:
    failed.append("joint target")
  if wrong_diff <= 1e-3:
    failed.append("target check is blind (the wrong origin agreed)")
  env.close()
  if failed:
    print(f"\nFAILED: {', '.join(failed)}")
    return 1
  print("\nPASS: the numpy deployment pipeline reproduces the simulator's "
        "observation and action path.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
