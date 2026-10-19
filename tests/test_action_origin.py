"""The deploy action path anchors on the episode start pose, as the sim action term does.

`ReducedJointPositionAction.reset` sets offset and EMA state to the pose the reset event
wrote, so `action = 0` holds the start pose -- not `default_joint_pos`, which is up to
0.6 rad away. The sim-side comparison is `scripts/deploy/selftest.py` ("joint target").

    python -m pytest tests/test_action_origin.py
"""
import pathlib

import numpy as np
import pytest

from dexrobot021_mjlab.deploy.contract import PolicyContract
from dexrobot021_mjlab.deploy.policy import ActionPipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bundle", ["spin_cube", "spin_cylinder"])
def test_zero_action_holds_the_start_pose(bundle):
  c = PolicyContract.load(ROOT / "checkpoints" / bundle / "contract.json")
  reset, default = np.asarray(c.reset_joint_pos), np.asarray(c.default_joint_pos)
  assert np.abs(reset - default).max() > 0.3, "fixture no longer tells the two poses apart"

  p = ActionPipeline(c)
  for _ in range(50):
    t = p.step(np.zeros(c.n_joints))
  assert np.allclose(t, np.clip(reset, *np.asarray(c.joint_limits).T))
  assert np.abs(t - default).max() > 0.3

  # An explicit origin (what selftest passes: the sim's jittered reset pose) is used
  # and kept across a plain reset.
  o = reset + 0.01
  p.reset(origin=o)
  assert np.allclose(p.step(np.zeros(c.n_joints)), o)
  p.reset()
  assert np.allclose(p.step(np.zeros(c.n_joints)), o)
