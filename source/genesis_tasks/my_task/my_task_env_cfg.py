from genesis_study.source.genesis.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from genesis_study.source.genesis.assets.asset_base_cfg import EntityAssetCfg, LightAssetCfg, CameraAssetCfg, FluidEmitterAssetCfg
from genesis_study.source.genesis.assets.morph_cfg import URDFCfg, MJCFCfg, BoxCfg
from genesis_study.source.genesis.assets.surface_cfg import SurfaceCfg
from genesis_study.source.genesis.assets.material_cfg import MaterialsCfg
from genesis_study.source.genesis.managers.manager_term_cfg import TerminationTermCfg as DoneTerm
from genesis_study.source.genesis.managers.manager_term_cfg import ObservationGroupCfg as ObsGroup
from genesis_study.source.genesis.managers.manager_term_cfg import ObservationTermCfg as ObsTerm
from genesis_study.source.genesis.managers.manager_term_cfg import SceneEntityCfg
from genesis_study.source.genesis.utils import configclass
from dataclasses import MISSING

from genesis_study.source.genesis import mdp

@configclass
class ObjectTableSceneCfg:

    # robots
    robot = EntityAssetCfg(
        MJCFCfg(file="xml/franka_emika_panda/panda.xml"),)

    # end-effector sensor
    # ee_frame = EntityAssetCfg = MISSING

    # Table
    # table = EntityAssetCfg

    # Cube
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


@configclass
class ActionCfg:
    arm_action = MISSING
    gripper_action = MISSING

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")}
    )

    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")}
    )

    cube_3_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_3")}
    )

    success = DoneTerm(func=mdp.cubes_stacked)


@configclass
class TaskEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg()
    actions : ActionCfg = ActionCfg()
    

    def __post_init__(self):
        self.dt = 0.01
        self.substeps = 2
        self.episode_length_s = 30.0


        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_finger.*"],
        #     open_command_expr={"panda_finger_.*": 0.04},
        #     close_command_expr={"panda_finger_.*": 0.0},
        # )

        # self.actions.arm_action = InverseKinematicsActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_joint.*"],
        #     body_name="panda_hand",
        #     controller=IKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        #     scale=0.5,
        #     body_offset=InverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
