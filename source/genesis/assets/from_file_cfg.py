from dataclasses import MISSING
from genesis_study.source.genesis.utils import configclass

@configclass
class FileCfg:
    file: str = MISSING
    scale: float | tuple = 1.0
    pos: tuple = (0.0, 0.0, 0.0)
    euler: tuple = (0.0, 0.0, 0.0)
    quat: tuple | None = None
    convexify: bool | None = None
    visualization: bool = True
    colision: bool = True
    requires_jac_and_IK: bool = True


@configclass
class UrdfFileCfg(FileCfg):
    fixed: bool = False
    prioritize__urdf_material: bool = False
    merge_fixed_links: bool = True
    links_to_keep: list[str] = []