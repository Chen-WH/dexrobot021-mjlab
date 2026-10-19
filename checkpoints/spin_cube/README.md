# deployment bundle: Dexrobot021-Spin-Cube-v0

- `model.pt`: rsl_rl checkpoint (iteration 5999); `policy.pt` / `policy.onnx`: the exported actor
- control rate: 62.5 Hz (2.0 ms x 8)
- observation: 49 values, 7 terms
- actions: 12 joints, scale 0.5, EMA alpha 0.41
- object: cube, tag size 28.0 mm (measured)

Validate against sim with `python scripts/deploy/selftest.py Dexrobot021-Spin-Cube-v0 checkpoints/spin_cube`,
run on the hand with `python scripts/deploy/run.py checkpoints/spin_cube`.
