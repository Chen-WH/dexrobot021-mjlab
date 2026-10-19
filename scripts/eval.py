"""Evaluate a spin checkpoint: sustained spin rate about the task's axis, joint speeds,
actuator saturation and terminations.

    python scripts/eval.py <task-id> <checkpoint> [--envs 256] [--steps 2500]

The task id is required: cube and cylinder policies share observation and action
shapes, so a checkpoint loads into the wrong task without error (the checkpoint
guard refuses it when it can tell). The play configuration is used: no
observation noise, bias or delay, but the startup randomisation is drawn.
"""
import argparse
from dataclasses import asdict

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api import math as math_utils

import dexrobot021_mjlab.tasks  # noqa: F401
from dexrobot021_mjlab.tasks.inhand.mdp.anchors import spin_axis_unit

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("task")
p.add_argument("checkpoint")
p.add_argument("--envs", type=int, default=256)
p.add_argument("--steps", type=int, default=2500, help="control steps (62.5 Hz)")
p.add_argument("--seed", type=int, default=None)
p.add_argument("--device", default="cuda:0")
a = p.parse_args()

cfg = load_env_cfg(a.task, play=True)
cfg.scene.num_envs = a.envs
if a.seed is not None:
  cfg.seed = a.seed
spin = cfg.rewards["spin"].params
axis = np.array(spin_axis_unit(spin["spin_axis"]))
axis_dir = float(spin["axis_direction"])
agent = load_rl_cfg(a.task)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=a.device),
                         clip_actions=agent.clip_actions)
runner = MjlabOnPolicyRunner(env, asdict(agent), device=a.device)
runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)
policy = runner.get_inference_policy(device=a.device)
robot, obj = env.unwrapped.scene["robot"], env.unwrapped.scene["object"]
limit = np.abs(env.unwrapped.sim.mj_model.actuator_forcerange[:, 1])

V, T, W = [], [], []
terminations = 0
obs = env.get_observations()
with torch.inference_mode():
  for _ in range(a.steps):
    out = env.step(policy(obs))
    obs = out[0]
    terminations += int(out[2].sum())
    V.append(robot.data.joint_vel.abs().cpu().numpy())
    T.append(env.unwrapped.sim.data.actuator_force.abs().cpu().numpy())
    W.append(math_utils.quat_apply_inverse(
      robot.data.root_link_quat_w, obj.data.root_link_ang_vel_w).cpu().numpy())
V, T, W = np.concatenate(V), np.concatenate(T), np.stack(W)
sat = T / limit

# Sustained rate: 1 s moving mean of the wrist-frame angular velocity, per env.
k = np.ones(62) / 62
sm = np.stack([[np.convolve(W[:, e, i], k, "valid") for i in range(3)]
               for e in range(W.shape[1])])
signed = axis_dir * np.einsum("eat,a->et", sm, axis)
print(f"checkpoint      {a.checkpoint}")
print(f"spin axis       {np.round(axis, 3)} (wrist frame)")
print(f"sustained spin  p50={np.percentile(signed, 50):.3f}  p95={np.percentile(signed, 95):.3f}"
      " rad/s  (1 s mean, + = commanded direction)")
print(f"terminations    {terminations}")
print(f"joint |w|       p95={np.percentile(V, 95):.2f}  p99.9={np.percentile(V, 99.9):.2f}"
      f"  max={V.max():.2f} rad/s")
print(f"torque/limit    p50={np.percentile(sat, 50):.2f}  p95={np.percentile(sat, 95):.2f}"
      f"  frac>0.95={(sat > 0.95).mean() * 100:.2f}%")
