"""Isaac Lab sub-terrain with flat approach, stairs, and flat exit."""

from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def stair_course_terrain(
    difficulty: float,
    cfg: "StairCourseTerrainCfg",
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a straight course: flat approach, three stairs, flat exit."""
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    total_stair_length = cfg.num_steps * cfg.step_width
    if cfg.approach_length + total_stair_length >= cfg.size[0]:
        raise ValueError("Stair course does not leave space for the exit platform.")

    if cfg.direction == "up":
        approach_height = 0.0
        stair_heights = [(index + 1) * step_height for index in range(cfg.num_steps)]
        exit_height = cfg.num_steps * step_height
    else:
        approach_height = cfg.num_steps * step_height
        stair_heights = [(cfg.num_steps - index - 1) * step_height for index in range(cfg.num_steps)]
        exit_height = 0.0

    segments = [(0.0, cfg.approach_length, approach_height)]
    segment_start = cfg.approach_length
    for top_height in stair_heights:
        segment_end = segment_start + cfg.step_width
        segments.append((segment_start, segment_end, top_height))
        segment_start = segment_end
    segments.append((segment_start, cfg.size[0], exit_height))

    meshes = []
    bottom_height = -cfg.base_depth
    for x_start, x_end, top_height in segments:
        length = x_end - x_start
        height = top_height - bottom_height
        center = (
            (x_start + x_end) / 2.0,
            cfg.size[1] / 2.0,
            bottom_height + height / 2.0,
        )
        meshes.append(
            trimesh.creation.box(
                extents=(length, cfg.size[1], height),
                transform=trimesh.transformations.translation_matrix(center),
            )
        )

    origin = np.array([cfg.start_offset, cfg.size[1] / 2.0, approach_height])
    return meshes, origin


@configclass
class StairCourseTerrainCfg(SubTerrainBaseCfg):
    function = stair_course_terrain
    direction: Literal["up", "down"] = "up"
    step_height_range: tuple[float, float] = (0.03, 0.14)
    step_width: float = 0.30
    num_steps: int = 3
    approach_length: float = 1.50
    start_offset: float = 0.50
    base_depth: float = 0.50
