from dataclasses import MISSING
from genesis_study.source.genesis.utils import configclass


@configclass
class SurfaceCfg:
    color: tuple | None = None
    vis_mode: str | None = None