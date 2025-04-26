
import genesis as gs

from genesis_study.source.genesis.assets.asset_base_cfg import EntityAssetCfg, LightAssetCfg, CameraAssetCfg, FluidEmitterAssetCfg
from genesis_study.source.genesis.assets.from_file_cfg import UrdfFileCfg
from genesis_study.source.genesis.utils import configclass

@configclass
class ObjectTableSceneCfg:

    # robots
    # robot

    # end-effector sensor
    # ee_frame

    # Table
    # table = 

    # plane
    plane = EntityAssetCfg(
        morph = gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    # light = 


@configclass
class ActionCfg:
    arm_action: 
    gripper_action: 

@configclass
class ObservationsCfg:
    pass

@configclass
class TerminationsCfg:
    pass

@configclass
class TaskEnvCfg:
    pass

