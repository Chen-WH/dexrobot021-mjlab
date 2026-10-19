"""Refuse a checkpoint trained on a different object than the task builds.

Cube and cylinder policies have identical observation and action shapes, so a
cube checkpoint loads into a cylinder task with no error and plays badly. Every
loader here goes through `MjlabOnPolicyRunner.load`, so the check wraps that one
function. It is installed by importing `dexrobot021_mjlab.tasks`.

Which object a checkpoint was trained on is read from, in order:
  * `params/env.yaml` next to it (a training run): the object's spawn position;
  * `contract.json` next to it (a released bundle): `object_kind`.
With neither, the checkpoint is loaded unchecked.
"""

from __future__ import annotations

import json
import pathlib
import re

_INSTALLED = False
_SPAWN_RE = re.compile(
  r"\n    object:\n      init_state:\n        pos: !!python/tuple\n"
  r"((?:\s*-\s*-?[\d.eE+-]+\n){3})")


def _object_of_spawn(pos) -> str | None:
  from dexrobot021_mjlab.assets.grasps import GRASPS

  for name, grasp in GRASPS.items():
    if max(abs(float(a) - b) for a, b in zip(pos, grasp.object_pos_world, strict=True)) < 5e-4:
      return name
  return None


def trained_object(checkpoint: str | pathlib.Path) -> str | None:
  """"cube" / "cylinder" for a checkpoint, or None if nothing records it."""
  folder = pathlib.Path(checkpoint).parent
  params = folder / "params" / "env.yaml"
  if params.exists():
    match = _SPAWN_RE.search(params.read_text())
    if match is not None:
      pos = [float(re.sub(r"^\s*-\s+", "", row))
             for row in match.group(1).strip().splitlines()]
      return _object_of_spawn(pos)
  contract = folder / "contract.json"
  if contract.exists():
    return json.loads(contract.read_text()).get("object_kind")
  return None


def check_checkpoint_object(checkpoint, env_cfg) -> None:
  try:
    env_obj = _object_of_spawn(env_cfg.scene.entities["object"].init_state.pos)
  except (AttributeError, KeyError, TypeError):
    return
  trained = trained_object(checkpoint)
  if trained is None or env_obj is None:
    return
  if trained != env_obj:
    raise RuntimeError(
      f"{checkpoint} was trained on the {trained}, but this task builds the "
      f"{env_obj}. Use the matching task id.")


def install() -> None:
  """Wrap `MjlabOnPolicyRunner.load` with the object check. Idempotent."""
  global _INSTALLED
  if _INSTALLED:
    return
  try:
    from mjlab.rl import MjlabOnPolicyRunner
  except Exception:  # pragma: no cover - mjlab not importable is not our problem
    return

  original = MjlabOnPolicyRunner.load

  def load(self, path, *args, **kwargs):
    env = getattr(self, "env", None)
    env_cfg = getattr(getattr(env, "unwrapped", env), "cfg", None)
    if env_cfg is not None:
      check_checkpoint_object(path, env_cfg)
    return original(self, path, *args, **kwargs)

  load.__doc__ = original.__doc__
  load.__wrapped__ = original
  MjlabOnPolicyRunner.load = load
  _INSTALLED = True
