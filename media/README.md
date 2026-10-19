# Media

Demo clips shown in the top-level README. All four are 400 x 400 px, 15 fps.

| File | Content |
|---|---|
| `cube_sim.gif` | Cube spin, simulation (`checkpoints/spin_cube`) |
| `cube_real.gif` | Cube spin, real hand |
| `cylinder_sim.gif` | Cylinder spin, simulation (`checkpoints/spin_cylinder`) |
| `cylinder_real.gif` | Cylinder spin, real hand |

The simulation clips were rendered offscreen with the released checkpoints, from a free
camera placed like the rig's external camera, in the `play` configuration. Render-only
changes: debug overlays and the object's axis markers hidden, and three stripes plus
end-face squares drawn on the cylinder so its spin is visible. None of these change the
physics. The real clips are from the rig's external camera.

To record a clip yourself:

```bash
uv run dexrobot021-play Dexrobot021-Spin-Cylinder-Finetune-v0 \
    --checkpoint-file checkpoints/spin_cylinder/model.pt --video --video-length 1250
```

The mp4 (1250 steps = one 20 s episode) is written to `checkpoints/spin_cylinder/videos/play/`.
The viewer stays open after recording; close it or press Ctrl-C. To convert the mp4 to a GIF:

```bash
ffmpeg -i in.mp4 -vf "fps=15,scale=400:-1:flags=lanczos,palettegen=max_colors=128" pal.png
ffmpeg -i in.mp4 -i pal.png -lavfi "fps=15,scale=400:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle" out.gif
```
