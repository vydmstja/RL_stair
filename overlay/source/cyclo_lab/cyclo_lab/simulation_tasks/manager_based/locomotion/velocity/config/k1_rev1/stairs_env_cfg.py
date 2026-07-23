"""Blind K1 course with flat approach followed by upward stairs."""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import cyclo_lab.simulation_tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import K1Rev1FlatEnvCfg, K1Rev1Rewards, K1Rev1TerminationsCfg
from .stairs_course_terrain import StairCourseTerrainCfg


GOAL_OFFSET = (5.0, 0.0, 0.0)


def _course_terrains():
    terrains = {}
    tread_depths = (0.25, 0.30, 0.38)
    for column in range(12):
        tread_depth = tread_depths[column % len(tread_depths)]
        terrains[f"column_{column:02d}_up_{int(tread_depth * 100)}cm"] = StairCourseTerrainCfg(
            proportion=1.0 / 12.0,
            direction="up",
            step_height_range=(0.03, 0.14),
            step_width=tread_depth,
            num_steps=3,
            approach_length=1.50,
            start_offset=0.50,
            base_depth=0.50,
        )
    return terrains


K1_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    size=(6.0, 3.0),
    border_width=8.0,
    num_rows=8,
    num_cols=12,
    color_scheme="height",
    horizontal_scale=0.05,
    vertical_scale=0.005,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains=_course_terrains(),
)


@configclass
class K1Rev1StairsRewards(K1Rev1Rewards):
    """Ascent-first velocity, progress, stability, contact, and smoothness objectives."""

    goal_progress = RewTerm(
        func=mdp.goal_progress_velocity,
        weight=1.25,
        params={"goal_offset": GOAL_OFFSET},
    )
    goal_position = RewTerm(
        func=mdp.goal_distance_linear,
        weight=2.0,
        params={"goal_offset": GOAL_OFFSET, "start_distance": GOAL_OFFSET[0]},
    )
    goal_reached = RewTerm(
        func=mdp.goal_reached,
        weight=75.0,
        params={"goal_offset": GOAL_OFFSET, "distance_threshold": 0.30},
    )
    forward_velocity = RewTerm(
        func=mdp.forward_velocity_clipped,
        weight=1.0,
        params={"max_reward": 0.8},
    )
    backward_velocity = RewTerm(
        func=mdp.backward_velocity_l2,
        weight=-2.0,
        params={"max_velocity": 0.8},
    )
    stair_height_tracking = RewTerm(
        func=mdp.stair_ascent_height_tracking,
        weight=3.0,
        params={
            "nominal_base_height": 0.78,
            "stair_start_x": 1.0,
            "stair_length": 0.96,
            "num_steps": 3,
            "step_height_range": (0.03, 0.14),
            "std": 0.10,
        },
    )
    stair_vertical_velocity = RewTerm(
        func=mdp.stair_ascent_vertical_velocity,
        weight=1.0,
        params={"stair_start_x": 1.0, "stair_length": 0.96, "max_reward": 0.30},
    )
    lateral_position = RewTerm(
        func=mdp.lateral_position_l2,
        weight=-0.75,
    )
    upper_body_joint_vel = RewTerm(
        func=mdp.selected_joint_vel_l2,
        weight=-0.025,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_yaw_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                ],
            )
        },
    )


@configclass
class K1Rev1StairsTerminations(K1Rev1TerminationsCfg):
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.85})
    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"goal_offset": GOAL_OFFSET, "distance_threshold": 0.30},
    )


@configclass
class K1Rev1StairsEnvCfg(K1Rev1FlatEnvCfg):
    """Walk on flat ground, ascend stairs, and reach the exit platform."""

    rewards: K1Rev1StairsRewards = K1Rev1StairsRewards()
    terminations: K1Rev1StairsTerminations = K1Rev1StairsTerminations()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = K1_STAIRS_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.debug_vis = False
        self.scene.env_spacing = 7.0
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

        # No camera, ray caster, depth, or explicit stair geometry observation.
        self.scene.height_scanner = None
        self.observations.policy.goal_position = ObsTerm(
            func=mdp.goal_position_b,
            params={"goal_offset": GOAL_OFFSET},
            clip=(-6.0, 6.0),
        )
        self.observations.critic.goal_position = ObsTerm(
            func=mdp.goal_position_b,
            params={"goal_offset": GOAL_OFFSET},
            clip=(-6.0, 6.0),
        )
        self.observations.policy.history_length = 15
        self.observations.critic.history_length = 15

        # The course always runs along +X; keep the blind command simple.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # Start with one meter of flat approach before the first stair.
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.08, 0.08),
            "y": (-0.08, 0.08),
            "yaw": (-0.08, 0.08),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.03, 0.03),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        }

        self.events.physics_material.params["static_friction_range"] = (0.60, 1.20)
        self.events.physics_material.params["dynamic_friction_range"] = (0.50, 1.00)
        self.events.physics_material.params["num_buckets"] = 64
        self.events.push_robot.interval_range_s = (10.0, 14.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}

        # Stability and natural contact behavior on both flat and stair areas.
        self.rewards.alive.weight = 0.03
        self.rewards.base_height = None
        self.rewards.track_lin_vel_xy_exp.weight = 0.8
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.2
        self.rewards.feet_air_time.weight = 0.8
        self.rewards.feet_single_contact.weight = 0.3
        self.rewards.feet_clearance.weight = 0.0
        self.rewards.feet_slide.weight = -0.4
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.dof_acc_l2.weight = -2.0e-7
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.joint_deviation_arms.weight = -0.25
        self.rewards.joint_deviation_shoulder_pitch.weight = -0.25
        self.rewards.joint_deviation_elbow.weight = -0.25
        self.rewards.joint_deviation_arm_lateral.weight = -0.75
        self.episode_length_s = 20.0


@configclass
class K1Rev1StairsEnvCfg_PLAY(K1Rev1StairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 12
        self.scene.terrain.max_init_terrain_level = 1
        self.observations.policy.enable_corruption = False
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
