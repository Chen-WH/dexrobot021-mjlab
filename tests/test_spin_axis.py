"""Spin axis as an index or a vector, and the axis each grasp holds its object on.

    python -m pytest tests/test_spin_axis.py
"""
import math

import numpy as np
import pytest

from dexrobot021_mjlab.assets.dexrobot021 import DEXROBOT021_LEFT_INIT_STATE
from dexrobot021_mjlab.assets.grasps import GRASPS, _quat_to_mat
from dexrobot021_mjlab.tasks.inhand.mdp.anchors import spin_axis_unit


def test_index_equals_unit_vector():
  for i in range(3):
    assert np.allclose(spin_axis_unit(i), np.eye(3)[i])
    assert np.allclose(spin_axis_unit(tuple(np.eye(3)[i] * 2.5)), np.eye(3)[i])
  rv = np.array([0.03, -0.11, 0.07])   # a rotation vector; the rate is its projection
  assert math.isclose(float(rv @ np.array(spin_axis_unit(1))), rv[1])


def test_zero_vector_is_rejected():
  with pytest.raises(ValueError):
    spin_axis_unit((0.0, 0.0, 0.0))


def test_cube_spins_about_wrist_y():
  assert np.allclose(GRASPS["cube"].spin_axis_wrist, (0.0, 1.0, 0.0))


def test_cylinder_axis_is_its_own_z_in_the_wrist_frame():
  g = GRASPS["cylinder"]
  # The spawn orientation in the world, carried back into the wrist frame, must put
  # the cylinder's local z on the axis the reward pays for.
  R_world_wrist = _quat_to_mat(DEXROBOT021_LEFT_INIT_STATE.rot)
  R_world_obj = _quat_to_mat(g.object_quat_world)
  z_in_wrist = R_world_wrist.T @ R_world_obj[:, 2]
  assert np.allclose(z_in_wrist, g.spin_axis_wrist, atol=1e-4)
  # Lying across the palm: mostly along wrist y.
  assert abs(g.spin_axis_wrist[1]) > 0.9
  # Must differ: the cylinder's local x is not its spin axis.
  assert not np.allclose(R_world_wrist.T @ R_world_obj[:, 0], g.spin_axis_wrist, atol=1e-2)
