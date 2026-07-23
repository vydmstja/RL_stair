"""Goal, reward, and termination terms for a blind stair course."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


def _goal_vector_w(env, goal_offset: tuple[float, float, float]) -> torch.Tensor:
    offset = torch.tensor(goal_offset, device=env.device, dtype=torch.float32)
    goal_position_w = env.scene.env_origins + offset
    robot: Articulation = env.scene["robot"]
    return goal_position_w - robot.data.root_pos_w


def goal_position_b(
    env,
    goal_offset: tuple[float, float, float] = (5.0, 0.0, 0.0),
) -> torch.Tensor:
    """Relative XY goal position in the robot body frame."""
    robot: Articulation = env.scene["robot"]
    goal_vector_b = quat_apply_inverse(robot.data.root_quat_w, _goal_vector_w(env, goal_offset))
    return goal_vector_b[:, :2]


def goal_progress_velocity(
    env,
    goal_offset: tuple[float, float, float] = (5.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward world velocity projected toward the target position."""
    asset: Articulation = env.scene[asset_cfg.name]
    goal_xy = _goal_vector_w(env, goal_offset)[:, :2]
    direction = goal_xy / torch.linalg.vector_norm(goal_xy, dim=1).clamp_min(1.0e-6).unsqueeze(1)
    progress = torch.sum(asset.data.root_lin_vel_w[:, :2] * direction, dim=1)
    return progress.clamp(min=-0.5, max=1.0)


def goal_distance_exp(
    env,
    goal_offset: tuple[float, float, float] = (5.0, 0.0, 0.0),
    std: float = 1.0,
) -> torch.Tensor:
    """Dense position-tracking reward that increases near the course goal."""
    distance = torch.linalg.vector_norm(_goal_vector_w(env, goal_offset)[:, :2], dim=1)
    return torch.exp(-torch.square(distance / std))


def goal_reached(
    env,
    goal_offset: tuple[float, float, float] = (5.0, 0.0, 0.0),
    distance_threshold: float = 0.30,
) -> torch.Tensor:
    """Return true once the robot reaches the flat exit platform."""
    distance = torch.linalg.vector_norm(_goal_vector_w(env, goal_offset)[:, :2], dim=1)
    return distance < distance_threshold
