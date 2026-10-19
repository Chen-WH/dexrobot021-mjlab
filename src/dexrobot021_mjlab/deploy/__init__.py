"""Hardware deployment for the DexHand 021 in-hand policies.

The package is deliberately free of `mjlab`, `torch.cuda`, `warp` and ROS
imports at module scope: the control loop has to start on a machine that has the
hand and the camera on it, which is not necessarily the machine that trained the
policy.  Everything sim-side lives in `scripts/deploy/export_bundle.py`, which
writes a self-describing bundle that this package consumes.
"""

from .contract import PolicyContract
from .cube import Cube, TagObservation

__all__ = ["Cube", "PolicyContract", "TagObservation"]
