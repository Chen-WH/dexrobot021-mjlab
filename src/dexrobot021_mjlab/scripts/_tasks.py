"""Task registry helpers for the CLI wrappers."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks

import dexrobot021_mjlab.tasks  # noqa: F401


def dexrobot021_tasks() -> list[str]:
  return sorted(task for task in list_tasks() if task.startswith("Dexrobot021-"))
