"""Goal, reward, and termination terms for a blind stair-ascent course."""

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


def _course_position_x(env, asset: Articulation) -> torch.Tensor:
    return asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]


def _terrain_difficulty(env) -> torch.Tensor:
    terrain = getattr(env.scene, "terrain", None)
    levels = getattr(terrain, "terrain_levels", None)
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if levels is None or generator is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    denom = max(float(generator.num_rows - 1), 1.0)
    return levels.to(env.device, dtype=torch.float32) / denom


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


def forward_velocity_clipped(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_reward: float = 0.8,
) -> torch.Tensor:
    """Reward forward course velocity and ignore lateral/turning shortcuts."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_w[:, 0], min=0.0, max=max_reward)


def backward_velocity_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_velocity: float = 0.8,
) -> torch.Tensor:
    """Penalize backing away from the stair course."""
    asset: Articulation = env.scene[asset_cfg.name]
    backward_speed = torch.clamp(-asset.data.root_lin_vel_w[:, 0], min=0.0, max=max_velocity)
    return torch.square(backward_speed)


def goal_distance_linear(
    env,
    goal_offset: tuple[float, float, float] = (5.0, 0.0, 0.0),
    start_distance: float = 5.0,
) -> torch.Tensor:
    """Dense course-completion reward that is nonzero from the first step."""
    distance = torch.linalg.vector_norm(_goal_vector_w(env, goal_offset)[:, :2], dim=1)
    return torch.clamp(1.0 - distance / start_distance, min=0.0, max=1.0)


def goal_distance_exp(
    env,
    goal_offset: tuple[float, float, float] = (5.0, 0.0, 0.0),
    std: float = 1.0,
) -> torch.Tensor:
    """Dense position-tracking reward that increases near the course goal."""
    distance = torch.linalg.vector_norm(_goal_vector_w(env, goal_offset)[:, :2], dim=1)
    return torch.exp(-torch.square(distance / std))


def lateral_position_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Keep the blind policy centered on the straight stair lane."""
    asset: Articulation = env.scene[asset_cfg.name]
    lateral_error = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return torch.square(lateral_error)


def stair_ascent_height_tracking(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    nominal_base_height: float = 0.78,
    stair_start_x: float = 1.0,
    stair_length: float = 0.96,
    num_steps: int = 3,
    step_height_range: tuple[float, float] = (0.03, 0.14),
    std: float = 0.10,
) -> torch.Tensor:
    """Reward the base being high enough for the current ascent phase.

    This is a privileged reward only. The actor still receives no terrain,
    camera, ray, or height-scan observation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    x_course = _course_position_x(env, asset)
    z_course = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    difficulty = _terrain_difficulty(env)
    step_height = step_height_range[0] + difficulty * (step_height_range[1] - step_height_range[0])
    climb_phase = torch.clamp((x_course - stair_start_x) / stair_length, min=0.0, max=1.0)
    expected_rise = climb_phase * (num_steps * step_height)
    target_base_height = nominal_base_height + expected_rise

    below_target_error = torch.clamp(target_base_height - z_course, min=0.0)
    active = x_course > (stair_start_x - 0.20)
    reward = torch.exp(-torch.square(below_target_error / std))
    return reward * active.float()


def stair_ascent_vertical_velocity(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stair_start_x: float = 1.0,
    stair_length: float = 0.96,
    max_reward: float = 0.30,
) -> torch.Tensor:
    """Reward upward base motion only while the robot is on the stair section."""
    asset: Articulation = env.scene[asset_cfg.name]
    x_course = _course_position_x(env, asset)
    in_stair_zone = (x_course > stair_start_x) & (x_course < stair_start_x + stair_length + 0.30)
    moving_forward = asset.data.root_lin_vel_w[:, 0] > 0.05
    upward_speed = torch.clamp(asset.data.root_lin_vel_w[:, 2], min=0.0, max=max_reward)
    return upward_speed * (in_stair_zone & moving_forward).float()


def selected_joint_vel_l2(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Mean squared velocity for a selected joint group."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def goal_reached(
    env,
    goal_offset: tuple[float, float, float] = (5.0, 0.0, 0.0),
    distance_threshold: float = 0.30,
) -> torch.Tensor:
    """Return true once the robot reaches the flat exit platform."""
    distance = torch.linalg.vector_norm(_goal_vector_w(env, goal_offset)[:, :2], dim=1)
    return distance < distance_threshold
