
import genesis as gs

from genesis_study.source.genesis.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from genesis_study.source.genesis.assets.asset_base_cfg import EntityAssetCfg, LightAssetCfg, CameraAssetCfg, FluidEmitterAssetCfg
from genesis_study.source.genesis.assets.from_file_cfg import URDFCfg
from genesis_study.source.genesis.managers.manager_term_cfg import TerminationTermCfg as DoneTerm
from genesis_study.source.genesis.managers.manager_term_cfg import ObservationGroupCfg as ObsGroup
from genesis_study.source.genesis.managers.manager_term_cfg import ObservationTermCfg as ObsTerm
from genesis_study.source.genesis.utils import configclass

gs.init()

@configclass
class ObjectTableSceneCfg:

    # robots
    robot = 

    # end-effector sensor
    ee_frame = 

    # Table
    table = 

    # plane
    plane = EntityAssetCfg(morph=URDFCfg(file="urdf/plane/plane.urdf", fixed=True))
    # light
    light = LightAssetCfg(morph=URDFCfg)


@configclass
class ActionCfg:
    arm_action: 
    gripper_action: 

@configclass
class ObservationsCfg:
    pass

@configclass
class EventCfg:
    # init_indy_arm_pose = EventTerm

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

class mytask2:
    def __init__(self, cfg: TaskEnvCfg):
        self.cfg = cfg
        self.define()

    def define(self):
        if isinstance(self.cfg.scene, dict):
            print(1)
            cfg_items = self.cfg.scene.items()
        else:
            print(2)
            cfg_items = self.cfg.scene.__dict__.items()

        for key, item in cfg_items:
            print(f'{key}: {item}')
            print(item.morph.__dict__.items(),'\n')


mytask2(TaskEnvCfg())
