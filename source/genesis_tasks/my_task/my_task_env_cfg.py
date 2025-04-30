
import genesis as gs

from genesis_study.source.genesis.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from genesis_study.source.genesis.assets.asset_base_cfg import EntityAssetCfg, LightAssetCfg, CameraAssetCfg, FluidEmitterAssetCfg
# 안쓰는게 더 좋아보임 나중에 삭제/ 그냥 gs 명령어로 하지 않는 이유가 있었음, gs 명령어를 하면 gs.init() 으로 하기
from genesis_study.source.genesis.assets.morph_cfg import URDFCfg, BoxCfg
from genesis_study.source.genesis.assets.surface_cfg import SurfaceCfg
from genesis_study.source.genesis.assets.material_cfg import MaterialsCfg
from genesis_study.source.genesis.managers.manager_term_cfg import TerminationTermCfg as DoneTerm
from genesis_study.source.genesis.managers.manager_term_cfg import ObservationGroupCfg as ObsGroup
from genesis_study.source.genesis.managers.manager_term_cfg import ObservationTermCfg as ObsTerm
from genesis_study.source.genesis.managers.manager_term_cfg import SceneEntityCfg
from genesis_study.source.genesis.utils import configclass


@configclass
class ObjectTableSceneCfg:

    # # robots
    # robot = EntityAssetCfg

    # # end-effector sensor
    # ee_frame = EntityAssetCfg

    # # Table
    # table = EntityAssetCfg

    # Cube
# 각 색상 큐브 생성
    cube_red = EntityAssetCfg(
        morph=BoxCfg(
            pos=(-0.0, 0.0, 1.0),
            size=(0.1, 0.1, 0.1),
            collision=True
        ),
        material=MaterialsCfg(rho=1000.0),
        surface=SurfaceCfg(
            color=(1.0, 0.0, 0.0),
            vis_mode='visual'
        )
    )
    cube_green = EntityAssetCfg(
        morph=BoxCfg(
            pos=( 0.0, 0.0, 0.5),
            size=(0.1, 0.1, 0.1),
            collision=True
        ),
        material=MaterialsCfg(rho=1000.0),
        surface=SurfaceCfg(
            color=(0.0, 1.0, 0.0),
            vis_mode='visual'
        )
    )
    cube_blue = EntityAssetCfg(
        morph=BoxCfg(
            pos=( 0.0, 0.0, 1.5),
            size=(0.1, 0.1, 0.1),
            collision=True
        ),
        material=MaterialsCfg(rho=1000.0),
        surface=SurfaceCfg(
            color=(0.0, 0.0, 1.0),
            vis_mode='visual'
        )
    )
    # plane
    plane = EntityAssetCfg(morph=URDFCfg(file="urdf/plane/plane.urdf", fixed=True))

    # light
    # light = LightAssetCfg(morph=URDFCfg)


# @configclass
# class ActionCfg:
#     arm_action: 
#     gripper_action: 

# @configclass
# class ObservationsCfg:
#     @configclass
#     class PolicyCfg(ObsGroup):
#         """Observations for policy group with state values."""

#         actions = ObsTerm(func=mdp.last_action)
#         joint_pos = ObsTerm(func=mdp.joint_pos_rel)
#         joint_vel = ObsTerm(func=mdp.joint_vel_rel)
#         object = ObsTerm(func=mdp.object_obs)
#         cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
#         cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
#         eef_pos = ObsTerm(func=mdp.ee_frame_pos)
#         eef_quat = ObsTerm(func=mdp.ee_frame_quat)
#         gripper_pos = ObsTerm(func=mdp.gripper_pos)

#         def __post_init__(self):
#             self.enable_corruption = False
#             self.concatenate_terms = False

#     @configclass
#     class RGBCameraPolicyCfg(ObsGroup):
#         """Observations for policy group with RGB images."""

#         def __post_init__(self):
#             self.enable_corruption = False
#             self.concatenate_terms = False

#     @configclass
#     class SubtaskCfg(ObsGroup):
#         """Observations for subtask group."""

#         grasp_1 = ObsTerm(
#             func=mdp.object_grasped,
#             params={
#                 "robot_cfg": SceneEntityCfg("robot"),
#                 "ee_frame_cfg": SceneEntityCfg("ee_frame"),
#                 "object_cfg": SceneEntityCfg("cube_2"),
#             },
#         )
#         stack_1 = ObsTerm(
#             func=mdp.object_stacked,
#             params={
#                 "robot_cfg": SceneEntityCfg("robot"),
#                 "upper_object_cfg": SceneEntityCfg("cube_2"),
#                 "lower_object_cfg": SceneEntityCfg("cube_1"),
#             },
#         )
#         grasp_2 = ObsTerm(
#             func=mdp.object_grasped,
#             params={
#                 "robot_cfg": SceneEntityCfg("robot"),
#                 "ee_frame_cfg": SceneEntityCfg("ee_frame"),
#                 "object_cfg": SceneEntityCfg("cube_3"),
#             },
#         )

#         def __post_init__(self):
#             self.enable_corruption = False
#             self.concatenate_terms = False

#     # observation groups
#     policy: PolicyCfg = PolicyCfg()
#     rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
#     subtask_terms: SubtaskCfg = SubtaskCfg()

# @configclass
# class EventCfg:
#     # init_indy_arm_pose = EventTerm

# @configclass
# class TerminationsCfg:

#     time_out = DoneTerm(func=mdp.time_out, time_out=True)

#     cube_1_dropping = DoneTerm(
#         func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")}
#     )

#     cube_2_dropping = DoneTerm(
#         func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")}
#     )

#     cube_3_dropping = DoneTerm(
#         func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_3")}
#     )

#     success = DoneTerm(func=mdp.cubes_stacked)

@configclass
class TaskEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = None

    def __post_init__(self):
        if self.scene is None:
            self.scene = ObjectTableSceneCfg()
    #     self.dt = 0.01
    #     self.substeps = 2
    #     self.episode_length_s = 30.0








# class mytask2:
#     def __init__(self, cfg: TaskEnvCfg):
#         self.cfg = cfg
#         self.define()

#     def define(self):
#         if isinstance(self.cfg.scene, dict):
#             print(1)
#             cfg_items = self.cfg.scene.items()
#         else:
#             print(2)
#             cfg_items = self.cfg.scene.__dict__.items()

#         for key, item in cfg_items:
#             print(f'{key}: {item}')
#             print(item.morph.__dict__.items(),'\n')


# mytask2(TaskEnvCfg())
