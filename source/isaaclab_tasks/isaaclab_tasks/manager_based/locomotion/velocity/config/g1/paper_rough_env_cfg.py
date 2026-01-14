# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import paper_LocomotionVelocityRoughEnvCfg, paper_RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip

import numpy as np
from scipy.stats import vonmises

# def compute_von_mises_values(self, offset: float = 0.0) -> dict:
#     """Compute the Von Mises values for a given number of discrete timesteps L."""
#     # 1) Fase normalizzata e in radianti
#     phi = np.linspace(0.0, 1.0, self.L, endpoint=False)
#     phi = (phi + offset) % 1.0  # apply offset

#     phi_rad = phi * (2.0 * np.pi)

#     # 2) Boundaries in radianti
#     start_rad_swing = (2.0 * np.pi) * float(self.swing_start)
#     end_rad_swing   = (2.0 * np.pi) * float(self.swing_end)
#     start_rad_stance= (2.0 * np.pi) * float(self.stance_start)
#     end_rad_stance  = (2.0 * np.pi) * float(self.stance_end)

#     kappa = float(self.kappa)

#     # 3) “Finestre” smooth tramite differenza di CDF von Mises
#     #    Nota: usiamo la CDF con "loc" settato ai rispettivi start/end.
#     vm_start_swing  = vonmises(kappa, loc=start_rad_swing)
#     vm_end_swing    = vonmises(kappa, loc=end_rad_swing)
#     vm_start_stance = vonmises(kappa, loc=start_rad_stance)
#     vm_end_stance   = vonmises(kappa, loc=end_rad_stance)

#     # CDF vettoriali
#     cdf_start_swing  = vm_start_swing.cdf(phi_rad)
#     cdf_end_swing    = vm_end_swing.cdf(phi_rad)
#     cdf_start_stance = vm_start_stance.cdf(phi_rad)
#     cdf_end_stance   = vm_end_stance.cdf(phi_rad)

#     # 4) Gestione wrap-around: se end < start, somma le due code
#     swing_wrap  = (end_rad_swing   < start_rad_swing)
#     stance_wrap = (end_rad_stance  < start_rad_stance)

#     if swing_wrap:
#         swing_values = cdf_start_swing + (1.0 - cdf_end_swing)
#     else:
#         swing_values = cdf_start_swing - cdf_end_swing

#     if stance_wrap:
#         stance_values = cdf_start_stance + (1.0 - cdf_end_stance)
#     else:
#         stance_values = cdf_start_stance - cdf_end_stance

#     # 5) Clipping numerico per stare in [0, 1]
#     swing_values  = np.clip(swing_values,  0.0, 1.0)        # probabilità che la fase sia attiva al tempo ciclico phi
#     stance_values = np.clip(stance_values, 0.0, 1.0)

    
#     dict = { float(phi[i]) : (float(swing_values[i]), float(stance_values[i]))
#       for i in range(len(phi)) }

#     return dict

@configclass
class paper_G1Rewards(paper_RewardsCfg):
    """Reward terms for the MDP."""

    # bipedal_reward
    bipedal_reward = RewTerm(
        func=mdp.bipedal_reward,
        weight=0.5,
        params={
            "left_foot_cfg": SceneEntityCfg(
                "robot", body_names="left_ankle_roll_link"
            ),
            "left_foot_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="left_ankle_roll_link"
            ),
            "right_foot_cfg": SceneEntityCfg(
                "robot", body_names="right_ankle_roll_link"
            ),
            "right_foot_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="right_ankle_roll_link"
            ),
        },
    )

    # cmd_reward
    cmd_reward = RewTerm(
        func=mdp.cmd_reward,
        weight=0.35,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),   ### check if is correct ###
        },

    )
    # smooth_reward

    smooth_reward = RewTerm(
        func=mdp.smooth_reward,
        weight=0.15,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "asset_root_cfg": SceneEntityCfg("robot", body_names="pelvis")},
    )
                
    # # bias term
    # bias = RewTerm(
    #     func=mdp.bias,
    #     weight=1.0 )

    # is alive term
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0 )

    # termination penalty
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0 )

@configclass
class paper_G1RoughEnvCfg(paper_LocomotionVelocityRoughEnvCfg):
    """Rough terrain environment configuration for the G1 robot following the paper settings."""

    rewards: paper_G1Rewards = paper_G1Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # self.L = 40 # number of discrete timesteps in the period
        # self.ratio = 0.5  
        # self.swing_start, self.swing_end = 0.0, self.ratio
        # self.stance_start, self.stance_end = self.ratio, 1.0
        # self.kappa = 45 # concentration parameter for Von Mises distribution
        # self.right_offset = 0.5  # phase offset for the right leg
        # self.left_offset = 0.0   # phase offset for the left leg
        # self.Von_Mises_Values_right = compute_von_mises_values(self, self.right_offset)     # is a dict [phi index (from 0 to L-1)->(swing_value, stance_value)]
        # self.Von_Mises_Values_left  = compute_von_mises_values(self, self.left_offset)      # is a dict [phi index (from 0 to L-1)->(swing_value, stance_value)]
        # Scene
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards

        ### here set rewards weights and params if different from default ###

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"

@configclass
class paper_G1RoughEnvCfg_PLAY(paper_G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None