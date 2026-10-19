# dexrobot021-mjlab

[English](README.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![mjlab](https://img.shields.io/badge/mjlab-1.6.0-orange.svg)](https://github.com/mujocolab/mjlab)

> 用 **DexRobot DexHand021**（左手，12 个电机）做方块与圆柱的手内自旋：在
> [mjlab](https://github.com/mujocolab/mjlab) 中用 PPO 训练，通过外部 AprilTag
> 相机部署到真手上。

## 演示

<table width="100%">
  <tr>
    <th width="50%">方块 · 70 mm</th>
    <th width="50%">圆柱 · 60 x 100 mm</th>
  </tr>
  <tr>
    <td><img src="media/cube_sim.gif" width="100%" alt="方块自旋，仿真" /><br /><img src="media/cube_real.gif" width="100%" alt="方块自旋，真机" /></td>
    <td><img src="media/cylinder_sim.gif" width="100%" alt="圆柱自旋，仿真" /><br /><img src="media/cylinder_real.gif" width="100%" alt="圆柱自旋，真机" /></td>
  </tr>
</table>

上：仿真；下：真机。

## 概述

手掌朝上固定，每个回合开始时手已经握住物体。策略输出 12 个电机的绝对关节目标（与手自身指令滤波器一致的 EMA 平滑），
奖励是物体绕某一轴的转速（在 1.5 rad/s 处截断），并以物体不偏离初始位置为门控。
观测只用真机能测到的量：关节角与角速度、AprilTag 相机测得的物体位姿与速度（含随机的手眼误差和实测延迟）、上一步动作。

能迁移到真机的关键：

- **驱动模型写在 MJCF 里**：每个电机是 MuJoCo `<dcmotor>`，空载转速按真手实测；DIP 通过手的多项式连杆关系（`<equality>`）跟随 PIP。
- **开局抓握针对实物求解**，真机上用同样的“张开放入 → 合拢”流程（`src/dexrobot021_mjlab/assets/grasps.py`）。
- **域随机化**：接触摩擦（切向与扭转）与刚度、关节阻尼/摩擦、执行器力矩、编码器零偏、相机外参、观测延迟。
- **面向硬件的惩罚项**：关节速度超过 1.5 rad/s 受罚；圆柱策略额外微调，避免顶住被挡的关节（会让电机驱动板锁死）。

## 结果

| 策略 | 物体 | 仿真持续转速（p50） | 真机 |
|---|---|---|---|
| [`checkpoints/spin_cube`](checkpoints/spin_cube) | 70 mm 方块，142.7 g | 1.7 rad/s | 净转速 15-26 °/s |
| [`checkpoints/spin_cylinder`](checkpoints/spin_cylinder) | 60 x 100 mm 圆柱，118.65 g | 1.9 rad/s | 绕自身轴 45-62 °/s，多数回合跑满 20 s |

仿真数值来自 `scripts/eval.py`（256 个环境，play 配置）。

## 安装

需要 Linux x86_64、NVIDIA GPU（使用 PyTorch CUDA 12.8 wheel）和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/Chen-WH/dexrobot021-mjlab.git
cd dexrobot021-mjlab
uv sync                    # 训练、回放、评估
uv run dexrobot021-list    # 列出已注册任务
```

真机部署再加 `uv sync --extra deploy`，见[真机部署](#真机部署)。

## 任务

| 任务 ID | 物体 | 说明 |
|---|---|---|
| `Dexrobot021-Spin-Cube-v0` | 70 mm 方块 | 绕手腕 y 轴（横跨指列方向）自旋，从零训练 |
| `Dexrobot021-Spin-Cylinder-v0` | 60 x 100 mm 圆柱 | 阶段 1：圆柱横放在掌上，绕自身轴自旋，从零训练 |
| `Dexrobot021-Spin-Cylinder-Finetune-v0` | 同上 | 阶段 2：加入“顶住被挡关节”惩罚与堵转终止，**只能热启动** |

任务同时注册为 `mjlab.tasks` entry point，mjlab 自带的 `train` / `play` / `list-envs` 也能看到。

## 回放与评估发布的策略

```bash
uv run dexrobot021-play Dexrobot021-Spin-Cube-v0 --checkpoint-file checkpoints/spin_cube/model.pt
uv run dexrobot021-play Dexrobot021-Spin-Cylinder-Finetune-v0 --checkpoint-file checkpoints/spin_cylinder/model.pt

# 持续转速、关节速度、执行器饱和
uv run python scripts/eval.py Dexrobot021-Spin-Cube-v0 checkpoints/spin_cube/model.pt
uv run python scripts/eval.py Dexrobot021-Spin-Cylinder-Finetune-v0 checkpoints/spin_cylinder/model.pt
```

无显示器的机器给 `play` 加 `--viewer viser`。方块和圆柱策略观测维度相同，装错任务不会报错，
代码里有检查会拒绝这种情况。

## 训练

日志写到 `logs/rsl_rl/<experiment>/`。发布的策略用 2048 个环境、单张 RTX 4070 Ti SUPER 训练（约 0.8 s/迭代）。

**方块**（6000 次迭代，约 1.3 小时；发布的是第 5999 次迭代）：

```bash
uv run dexrobot021-train Dexrobot021-Spin-Cube-v0 --env.scene.num-envs 2048
```

**圆柱**分两阶段。阶段 2 的惩罚项是按“已经会转”的策略定标的，从零训练会把策略罚到不动。

```bash
# 阶段 1（6000 次迭代）
uv run dexrobot021-train Dexrobot021-Spin-Cylinder-v0 --env.scene.num-envs 2048

# 把动作噪声重置为 0.4，并放到阶段 2 的日志目录下
uv run python scripts/reinflate_std.py logs/rsl_rl/spin_cylinder/<run>/model_<N>.pt \
    --task Dexrobot021-Spin-Cylinder-Finetune-v0 --std 0.4

# 阶段 2（最多再训 4000 次迭代）
uv run dexrobot021-train Dexrobot021-Spin-Cylinder-Finetune-v0 --env.scene.num-envs 2048 \
    --agent.resume True --agent.load-run seed --agent.load-checkpoint model_std04.pt
```

发布的圆柱策略是第 8700 次迭代（阶段 2 的第 2700 次）。原始训练在这之后不久因观测出现 NaN
（偶发的仿真数值发散）而中断；若你的训练也遇到，从最后一个 checkpoint 续训即可。可以用 `scripts/eval.py` 比较各 checkpoint。

## 真机部署

### 硬件

- DexRobot **DexHand021 左手**，通过 CAN 用 DexRobot 的 `pyzlg_dexhand` 驱动，**掌心朝上**固定
  （我们用 UR10 当支架，不对它下指令）。
- **大恒 MER2-301-125U3C**（2048x1536，全局快门）+ 8 mm C 口镜头，从侧面看手，用于跟踪物体。需要大恒 Galaxy SDK（`gxipy`）。
- 手的安装支架上的 **RealSense D435i**，只用于标定大恒（它相对手的 CAD 位姿写在 `scripts/deploy/extrinsic.py`，支架不同需要改这里的数）。
- [`hardware/`](hardware/) 里打印的方块、圆柱标签和手腕 tag。

`pyzlg_dexhand` 和 `gxipy` 不在 PyPI 上，需按厂商说明装进环境。

### 步骤

```bash
# 1. 换镜头/调焦后：棋盘格标定大恒内参（用卡尺量方格边长）
uv run python scripts/deploy/dahengcal.py --square-mm <实测值>

# 2. 每只手一次：逐关节验证方向和零点（手必须空着，会动）
uv run python scripts/deploy/verify_hand.py checkpoints/spin_cube

# 3. 相机、手腕 tag 或支架动过之后：拿方块让两台相机同时看到，标定大恒和手腕 tag 在手坐标系下的位姿
#    （方块和圆柱都用方块 bundle 标定）
uv run python scripts/deploy/extcal.py checkpoints/spin_cube

# 4. 验证 numpy 部署管线与仿真一致（不需要硬件）
uv run python scripts/deploy/selftest.py Dexrobot021-Spin-Cube-v0 checkpoints/spin_cube

# 5. 每次开工前
uv run python scripts/deploy/preflight.py checkpoints/spin_cube --camera --hand

# 6. 运行（先用 --mock 空跑一遍）
uv run python scripts/deploy/run.py checkpoints/spin_cube --episodes 3 --trace
uv run python scripts/deploy/run.py checkpoints/spin_cylinder --episodes 3 --trace
```

每个回合：手张开到放置姿态（空手）；把物体放到张开的手指上后松手（圆柱横放，大致沿拇指到小指方向）；
静止 2 s 且通过放置检查后手合拢到开局抓握，策略最多运行 20 s；结束后把物体取出。
物体要放深、靠近掌心——搁在指尖上会被拒绝（`--place-z-tol-mm`）。每次运行会在 bundle 目录下写
JSON 记录（加 `--trace` 另存逐步 `.npz`）。

## 目录结构

```
src/dexrobot021_mjlab/
  assets/        手与物体的 MJCF 和网格、开局抓握
  tasks/inhand/  环境配置、奖励/观测/事件、任务注册
  deploy/        numpy 运行时：contract、观测、策略、跟踪、相机、手驱动
scripts/         eval.py、reinflate_std.py、deploy/（导出、自检、标定、运行）
checkpoints/     发布的 bundle：model.pt（rsl_rl）、policy.pt / policy.onnx、contract.json
hardware/        方块、圆柱和手腕的打印标签
```

从自己的 checkpoint 导出 bundle：
`uv run python scripts/deploy/export_bundle.py <task> <model_N.pt> -o checkpoints/<name>`。

## 致谢

- 仿真基于 [mjlab](https://github.com/mujocolab/mjlab) 与 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)，PPO 来自 [rsl_rl](https://github.com/leggedrobotics/rsl_rl)。
- 任务配方参考 [mjlab_hand](https://github.com/ruoyiqiao/mjlab_hand) 的手内旋转任务。
- DexHand021 网格来自 DexRobot 的 [dexrobot_urdf](https://github.com/DexRobot/dexrobot_urdf)（Apache-2.0）；方块模型与贴图来自 [MuJoCo MPC](https://github.com/google-deepmind/mujoco_mpc)（Apache-2.0）；驱动参数参考 Yuan et al., "Development of the Bioinspired Tendon-Driven DexHand 021 With Proprioceptive Compliance Control", IEEE RA-L, 2026。

## 引用

```bibtex
@software{dexrobot021mjlab2026,
  title  = {dexrobot021-mjlab: In-Hand Spinning with the DexHand021 in mjlab},
  author = {Chen, Weihao},
  year   = {2026},
  url    = {https://github.com/Chen-WH/dexrobot021-mjlab}
}
```

## 许可证

Apache 2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
