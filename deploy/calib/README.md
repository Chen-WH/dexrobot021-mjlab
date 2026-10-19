# Rig calibration

Written by the deployment tools and read by `run.py` / `preflight.py`. These files
describe one particular bench (camera placement, lens, hand wiring), so they are not
versioned (see `.gitignore`); produce your own:

| File | Written by | What |
|---|---|---|
| `daheng_intrinsics.json` | `scripts/deploy/dahengcal.py --square-mm <measured>` | Daheng intrinsics and distortion from a checkerboard |
| `joint_verification.json` | `scripts/deploy/verify_hand.py <bundle>` | Per-joint sign / zero check of the hand |
| `extcal.json` | `scripts/deploy/extcal.py checkpoints/spin_cube` | Daheng pose and wrist-tag pose in the hand frame |

Re-run `extcal.py` whenever the Daheng, the wrist tag or the hand mount moves, and
`dahengcal.py` whenever the lens or its focus changes.
