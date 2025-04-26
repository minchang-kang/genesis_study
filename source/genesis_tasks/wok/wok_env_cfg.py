
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
    plane = URDFAssetCfg(UrdfFileCfg)

    # light = 


@configclass
class ActionCfg:
    pass

@configclass
class ObservationsCfg:
    pass

@configclass
class TerminationsCfg:
    pass

@configclass
class WokEnvCfg:
    pass

