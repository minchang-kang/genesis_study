from dataclasses import MISSING
from genesis_study.source.genesis.utils import configclass


@configclass
class MaterialsCfg:
    rho: float | None = None