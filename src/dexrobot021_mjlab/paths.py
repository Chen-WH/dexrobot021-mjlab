"""Filesystem locations: bundled assets, the log root and the rig calibration."""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parents[1]
ASSETS_DIR = _PACKAGE_ROOT / "assets"


def project_root() -> Path:
  return _PROJECT_ROOT


def output_root() -> Path:
  """Where `logs/` goes. Defaults to the repository root; override with
  `DEXROBOT021_MJLAB_OUTPUT_ROOT`."""
  value = os.environ.get("DEXROBOT021_MJLAB_OUTPUT_ROOT")
  return Path(value).expanduser().resolve() if value else _PROJECT_ROOT


def rsl_rl_log_root() -> Path:
  return output_root() / "logs" / "rsl_rl"


def calib_dir() -> Path:
  """Rig calibration that belongs to the bench, not to a policy bundle
  (`extcal.json`, `joint_verification.json`). Written by the deploy tools."""
  return _PROJECT_ROOT / "deploy" / "calib"


def dexrobot021_left_mimic_mjcf() -> Path:
  return ASSETS_DIR / "robots" / "dexhand021" / "mjcf" / "dexhand021_left_convex_mimic.xml"


def cube_mjcf() -> Path:
  return ASSETS_DIR / "objects" / "cube" / "mjcf" / "reorientation_cube.xml"


def cylinder_mjcf() -> Path:
  return ASSETS_DIR / "objects" / "cylinder" / "mjcf" / "reorientation_cylinder.xml"
