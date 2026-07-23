"""Visualize the K1 stair environment without training or a checkpoint."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Cyclo-Velocity-Stairs-K1-Rev1-Play-v0",
    help="Registered Cyclo Lab environment to visualize.",
)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--max_steps", type=int, default=0, help="Exit after N steps; 0 keeps the GUI running.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import cyclo_lab  # noqa: F401,E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def main():
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    print(f"[INFO] Visualizing {args_cli.task} with {args_cli.num_envs} environments.")
    print("[INFO] This script does not train or load a policy.")
    print(
        f"[INFO] PPO config: experiment={agent_cfg.experiment_name}, "
        f"max_iterations={agent_cfg.max_iterations}, save_interval={agent_cfg.save_interval}"
    )

    step_count = 0
    while simulation_app.is_running() and (args_cli.max_steps <= 0 or step_count < args_cli.max_steps):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
        step_count += 1

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
