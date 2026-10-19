# Hardware

Printable targets for the real-robot runs. Print everything at **100 % / actual
size** and measure the result: a scaled print does not fail, it silently scales every
recovered distance.

| Directory | What | Tags |
|---|---|---|
| [`cube_70mm/`](cube_70mm/) | Net for a 70 mm cube, 2x2 tag36h11 tiles per face (A3) | ids 20-43, 28 mm black square |
| [`cylinder/`](cylinder/) | Two end-face labels for a 60 mm x 100 mm cylinder (A4) | ids 6 (top), 7 (bottom), 28.8 mm black square |
| [`wrist_tag/`](wrist_tag/) | Wrist reference tag (SVG, exact scale) | id 10, 80 mm black square |

## Objects

| Object | Size | Mass used in simulation |
|---|---|---|
| Cube | 70 mm edge (pine) | 142.7 g |
| Cylinder | 60 mm diameter x 100 mm (pine) | 118.65 g |

The policies were trained on these sizes and masses. A different object needs a new
episode-start grasp (`src/dexrobot021_mjlab/assets/grasps.py`) and retraining.

## Cube

1. Print `cube_70mm/dexhand_cube_tiles_2x2_70mm_A3.pdf` on A3 at 100 %. The
   "MEASURE ME" bar must be exactly 100 mm.
2. Cut along the INNER edge of the solid outline, crease on the dashed lines and wrap
   the cube **printed side out**. The FRONT face (+Y) is the centre of the cross; the
   face labelled RIGHT (+X) is the cube's own right, i.e. on your LEFT when you face
   the FRONT face.
3. `cube_tile_mapping_2x2_70mm.yaml` is the machine-readable layout;
   `tests/test_tile_layout.py` checks the code against it.

## Cylinder

Cut the two 58 mm circles from `cylinder/cylinder_end_tags_tag36h11_A4.pdf` and stick
one on each end face, centred. Which end gets id 6 and the in-plane angle of each
sticker do not matter: training randomises the cylinder's roll about its axis and
which end faces the thumb. Measure the printed black square; if it is not 28.8 mm,
re-export the bundle with `--tag-size-m`.

## Wrist tag

Print `wrist_tag/tag36h11_id10_80mm.svg` at 100 % (black square 80 mm; measure it and
pass the value as `--wrist-tag-size` to `extcal.py` and `run.py`). Mount it rigidly
on the hand's mounting bracket where the external camera sees it together with the
object. The object pose is measured relative to this tag in every frame, so it has to
stay put after `extcal.py` has been run.
