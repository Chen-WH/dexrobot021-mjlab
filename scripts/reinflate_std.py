"""Reset a checkpoint's action noise and stage it as the seed of a fine-tuning run.

A converged policy carries a collapsed action distribution. Resetting
`log_std_param` keeps everything it knows about the hand and the object while
restoring enough exploration to adapt to new reward terms.

    python scripts/reinflate_std.py <stage-1 model_N.pt> \
        --task Dexrobot021-Spin-Cylinder-Finetune-v0 --std 0.4

writes `logs/rsl_rl/<task experiment>/<run-name>/model_std04.pt` (plus the source
run's `params/`) and prints the training command that resumes from it.
"""
import argparse
import math
import pathlib
import shutil

import torch

from dexrobot021_mjlab.paths import rsl_rl_log_root
from dexrobot021_mjlab.tasks.inhand.config.dexrobot021_left import experiment_name_for

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("checkpoint", type=pathlib.Path)
p.add_argument("--task", required=True, help="the task that will be fine-tuned")
p.add_argument("--std", type=float, default=0.4)
p.add_argument("--run-name", default="seed")
a = p.parse_args()

d = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
actor = d["actor_state_dict"]
key = "distribution.log_std_param" if "distribution.log_std_param" in actor else "log_std_param"
old = actor[key]
print(f"source {a.checkpoint}  iter={d['iter']}")
print(f"  std before: {[round(v, 3) for v in old.exp().tolist()]}")
actor[key] = torch.full_like(old, math.log(a.std))
print(f"  std after : {a.std} on all {old.numel()} dims")

run_dir = rsl_rl_log_root() / experiment_name_for(a.task) / a.run_name
run_dir.mkdir(parents=True, exist_ok=True)
name = f"model_std{round(a.std * 10):02d}.pt"
torch.save(d, run_dir / name)
params = a.checkpoint.parent / "params"
if params.is_dir():
  shutil.copytree(params, run_dir / "params", dirs_exist_ok=True)
print(f"wrote {run_dir / name}\n\nnext:\n  dexrobot021-train {a.task} --env.scene.num-envs 2048 "
      f"--agent.resume True --agent.load-run {a.run_name} --agent.load-checkpoint {name}")
