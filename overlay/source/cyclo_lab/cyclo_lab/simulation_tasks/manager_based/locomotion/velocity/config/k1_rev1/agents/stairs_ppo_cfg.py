"""RSL-RL PPO configuration dedicated to blind K1 stair locomotion."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import K1Rev1FlatPPORunnerCfg


@configclass
class K1Rev1StairsPPORunnerCfg(K1Rev1FlatPPORunnerCfg):
    experiment_name = "k1_rev1_stairs_ascent_velocity"
    max_iterations = 10000
    save_interval = 100
    num_steps_per_env = 24

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.8
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.008
        self.algorithm.learning_rate = 1.0e-3
