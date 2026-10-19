"""Train a DexRobot021 task with the upstream mjlab trainer."""

from __future__ import annotations

import sys
from dataclasses import replace

import mjlab
import tyro
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.scripts.train import TrainConfig, launch_training

from dexrobot021_mjlab.paths import rsl_rl_log_root
from dexrobot021_mjlab.scripts._tasks import dexrobot021_tasks


def _cli_sets_log_root(args: list[str]) -> bool:
  return any(arg == "--log-root" or arg.startswith("--log-root=") for arg in args)


def main() -> None:
  maybe_print_top_level_help("dexrobot021-train")

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(dexrobot021_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  if not _cli_sets_log_root(remaining_args):
    args = replace(args, log_root=str(rsl_rl_log_root()))
  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
