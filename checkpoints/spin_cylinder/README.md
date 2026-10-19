# deployment bundle: Dexrobot021-Spin-Cylinder-Finetune-v0

- `model.pt`: rsl_rl checkpoint (iteration 8700); `policy.pt` / `policy.onnx`: the exported actor
- control rate: 62.5 Hz (2.0 ms x 8)
- observation: 49 values, 7 terms
- actions: 12 joints, scale 0.5, EMA alpha 0.41
- object: cylinder, tag size 28.8 mm (nominal)

Validate against sim with `python scripts/deploy/selftest.py Dexrobot021-Spin-Cylinder-Finetune-v0 checkpoints/spin_cylinder`,
run on the hand with `python scripts/deploy/run.py checkpoints/spin_cylinder`.
