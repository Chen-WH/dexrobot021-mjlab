"""`deploy.cube.tile_layout()` against the print generator's own mapping, tile by tile.

The tile positions in `deploy/cube.py` are DERIVED (face normal, printed-up, cross
product, a quarter of an edge), while the paper on the cube was printed from
`hardware/cube_70mm/`. Two derivations of the same geometry agree until one of them
is edited, and the runtime cannot catch a disagreement: four coplanar tiles agree
with each other by construction.

    python -m pytest tests/test_tile_layout.py
"""
import pathlib

import numpy as np
import yaml

from dexrobot021_mjlab.deploy.cube import Cube, tile_layout, tile_tag_size_m

MAPPING = (pathlib.Path(__file__).resolve().parents[1]
           / "hardware" / "cube_70mm" / "cube_tile_mapping_2x2_70mm.yaml")

_AXES = {"+X": np.array([1.0, 0, 0]), "-X": np.array([-1.0, 0, 0]),
         "+Y": np.array([0, 1.0, 0]), "-Y": np.array([0, -1.0, 0]),
         "+Z": np.array([0, 0, 1.0]), "-Z": np.array([0, 0, -1.0])}


def _mismatches(layout) -> tuple[int, int, float]:
  """(tiles whose derived origin disagrees with the generator, total, worst mm)."""
  doc = yaml.safe_load(MAPPING.read_text())
  edge_m = float(doc["edge_size_m"])
  cube = Cube(edge_m, tile_tag_size_m(edge_m), layout=layout)
  want = {}
  for face in doc["faces"].values():
    n = _AXES[face["normal_in_cube_frame"]]
    for t in face["tiles"]:
      want[int(t["id"])] = n * edge_m / 2 + np.array(t["offset_m"], float)
  assert set(cube.T_cube_tag) == set(want), sorted(set(cube.T_cube_tag) ^ set(want))
  off = {i: float(np.linalg.norm(cube.T_cube_tag[i][:3, 3] - p)) for i, p in want.items()}
  return sum(v > 1e-9 for v in off.values()), len(off), 1e3 * max(off.values())


def test_tile_positions_match_the_generator():
  doc = yaml.safe_load(MAPPING.read_text())
  edge_m = float(doc["edge_size_m"])
  assert abs(edge_m - 0.070) < 1e-9
  # the detector's tag size is derived in deploy/cube.py and printed by the generator
  assert abs(tile_tag_size_m(edge_m) - float(doc["detector_tag_size_m"])) < 1e-9
  bad, total, worst_mm = _mismatches(tile_layout())
  assert bad == 0, f"{bad}/{total} tiles disagree, worst {worst_mm:.3f} mm"


def test_must_fail_row_a_flipped_in_face_axis_moves_every_tile():
  """Break the derivation and the comparison must notice.

  Negating the in-face offset moves every tile onto a sibling's position, so the id
  set still matches and only a per-tile position comparison can see it.
  """
  broken = {name: (tag, normal, up, (-np.asarray(off, float)).tolist())
            for name, (tag, normal, up, off) in tile_layout().items()}
  bad, total, _ = _mismatches(broken)
  assert bad == total, f"flipping the in-face axis moved only {bad}/{total} tiles"
