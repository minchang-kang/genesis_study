from dataclasses import MISSING
from genesis_study.source.genesis.utils import configclass
from genesis_study.source.genesis.managers.action_manager import GenesisActionTerm


@configclass
class ActionTermCfg:
    class_type: type[GenesisActionTerm] = MISSING

    asset_name: str = MISSING

    clip: dict[str, tuple] | None = None

@configclass
class InverseKinematicsActionCfg(ActionTermCfg):

    @configclass
    class OffsetCfg:
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    class_type: type[GenesisActionTerm] = task_space_actions.DifferentialInverseKinematicsAction
    joint_names: list[str] = MISSING
    body_name: str = MISSING
    body_offset: OffsetCfg | None = None
    scale: float | tuple[float, ...] = 1.0
    controller: DifferentialIKControllerCfg = MISSING