from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg
from .paper_rough_env_cfg import paper_G1RoughEnvCfg


# from .recorders_cfg import (
#     ActionStateRecorderManagerCfg,
#     InitialStateRecorderCfg,
#     PostStepStatesRecorderCfg,
#     PreStepActionsRecorderCfg,
#     PreStepFlatPolicyObservationsRecorderCfg,
#     PostStepProcessedActionsRecorderCfg,
# )



@configclass
class paper_G1FlatEnvCfg(paper_G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.variable_L = True



        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards

        # ### here set rewards weights and params if different from default ###
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        # )
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


class paper_G1FlatEnvCfg_PLAY(paper_G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # #Recorder settings
        # self.recorders = ActionStateRecorderManagerCfg(
        #     recorders=[
        #         InitialStateRecorderCfg(),
        #         PreStepActionsRecorderCfg(),
        #         PostStepProcessedActionsRecorderCfg(),
        #         PreStepFlatPolicyObservationsRecorderCfg(),
        #         PostStepStatesRecorderCfg(),
        #     ],
        #     data
        # )

        self.save_quantities = True

        self.episode_length_s = 40.0


        # self.vel_list = [(0.3, 0.0, 0.0),  # in teoria non serve
        #                 (0.8, 0.0, 0.0),
        #                 (1.2, 0.0, 0.0)]



        # self.ratio=0.5
        self.variable_L = True
        self.seed = 0
        # self.kappa=20
        # self.right_offset=0.5
        # self.left_offset=0.0

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)
        self.commands.base_velocity.resampling_time_range = (1,1)



        # Viewer
        self.viewer = ViewerCfg(eye=(2.5, 2.5, 1.0), origin_type="asset_root", env_index=0, asset_name="robot")
        # viewer: ViewerCfg = ViewerCfg(
        # eye=(0.0, 3.0, 1.25), lookat=(0.0, 0.0, 0.5), origin_type="asset_body", asset_name="robot", body_name="pelvis") 
        # 5.5, 5.5, 1.3
