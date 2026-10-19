"""RSL-RL PPO config for DexRobot021 in-hand tasks."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def ppo_runner_cfg(*, experiment_name: str, max_iterations: int) -> RslRlOnPolicyRunnerCfg:
  """PPO settings of ruoyiqiao/mjlab_hand's Allegro in-hand rotation task.

  `init_std=1.0`: the absolute EMA action target has a restoring force, so full
  exploration noise does not integrate into a drift.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.0,
      value_loss_coef=2.0,
      use_clipped_value_loss=False,
      learning_rate=5.0e-4,
      schedule="adaptive",
    ),
    experiment_name=experiment_name,
    save_interval=100,
    num_steps_per_env=16,
    max_iterations=max_iterations,
    clip_actions=1.0,
    logger="tensorboard",
    upload_model=False,
  )
