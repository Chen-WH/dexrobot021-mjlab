"""DexRobot021 left-hand spin task registrations.

`experiment_name` (the log directory under `logs/rsl_rl/`) is derived from the task
id, so a run's directory always names the task it was trained on.

Cube and cylinder policies have the same observation and action shapes, so a
checkpoint loads into the wrong task without error; `checkpoint_guard` refuses it.
"""

from mjlab.tasks.registry import register_mjlab_task

from dexrobot021_mjlab.tasks.inhand.env_cfg import make_spin_env_cfg
from dexrobot021_mjlab.tasks.inhand.rl_cfg import ppo_runner_cfg


def experiment_name_for(task_id: str) -> str:
  """`Dexrobot021-Spin-Cube-v0` -> `spin_cube`."""
  stem = task_id.removeprefix("Dexrobot021-").removesuffix("-v0")
  return stem.lower().replace("-", "_")


def _register(task_id: str, *, max_iterations: int, **env_kw) -> None:
  register_mjlab_task(
    task_id=task_id,
    env_cfg=make_spin_env_cfg(play=False, **env_kw),
    play_env_cfg=make_spin_env_cfg(play=True, **env_kw),
    rl_cfg=ppo_runner_cfg(experiment_name=experiment_name_for(task_id),
                          max_iterations=max_iterations),
    runner_cls=None,
  )


# 70 mm cube, spun about the wrist y axis (across the finger row, in the palm
# plane): a hand with little abduction rolls an object across the palm with
# flexion alone. The reward clips at 1.5 rad/s, and joints above 1.5 rad/s are
# penalised. Trained from scratch; the released policy is iteration 5999.
_register(
  "Dexrobot021-Spin-Cube-v0",
  obj="cube",
  max_iterations=6000,
)

# 60 x 100 mm cylinder lying across the palm, spun about its own axis (rolled in
# the fist). Stage 1, trained from scratch with the same recipe as the cube.
_register(
  "Dexrobot021-Spin-Cylinder-v0",
  obj="cylinder",
  max_iterations=6000,
)

# Stage 2: warm-started from a stage-1 policy (`scripts/reinflate_std.py --std 0.4`)
# with two terms against pushing on a blocked joint, which on the hand latches a
# motor board: `blocked_lead_l2` at -100 (~11 % of a trained policy's spin reward
# per step) and the `stall_latch` termination. Both are sized against a policy that
# already spins -- from scratch they freeze it -- so this task is fine-tuning only.
# The released cylinder policy is iteration 8700 (2700 iterations into stage 2).
_register(
  "Dexrobot021-Spin-Cylinder-Finetune-v0",
  obj="cylinder",
  blocked_lead_weight=-100.0,
  stall_latch=True,
  max_iterations=4000,
)
