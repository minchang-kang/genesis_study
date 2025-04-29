import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from genesis_study.source.genesis.utils import configclass
from .manager_base import ManagerTermBase

# from .scene_entity_cfg import SceneEntityCfg




@configclass
class ManagerTermBaseCfg:

    func: Callable | ManagerTermBase = MISSING

    params: dict[str, Any] = dict()


@configclass
class EvnetTermCfg(ManagerTermBaseCfg):

    func: Callable[..., None] = MISSING

    mode: str = MISSING

    interval_range_s: tuple[float, float] | None = None

    is_global_time: bool = False

    min_step_count_between_reset: int = 0

@configclass
class RewardTermCfg(ManagerTermBaseCfg):

    func: Callable[..., torch.Tensor] = MISSING

    weight: float = MISSING


@configclass
class TerminationTermCfg(ManagerTermBaseCfg):

    func: Callable[..., torch.Tensor] = MISSING

    time_out: bool = False
    