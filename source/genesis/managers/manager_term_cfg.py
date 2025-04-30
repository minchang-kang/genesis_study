from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from genesis_study.source.genesis.utils import configclass
from genesis_study.source.genesis.utils.noise.noise_cfg import NoiseCfg
from genesis_study.source.genesis.utils.modifiers.modifier_cfg import ModifierCfg

if TYPE_CHECKING:
    from genesis_study.source.genesis.managers.manager_base import ManagerTermBase

# from .scene_entity_cfg import SceneEntityCfg




@configclass
class ManagerTermBaseCfg:

    func: Callable | ManagerTermBase = MISSING

    params: dict[str, Any] = dict()


@configclass
class ObservationTermCfg(ManagerTermBaseCfg):

    func: Callable[..., torch.Tensor] = MISSING

    modifiers: list[ModifierCfg] | None = None

    noise: NoiseCfg | None= None

    clip: tuple[float, float] | float | None = None

    scale: tuple[float, ...] | float | None = None

    history_length: int = 0


@configclass
class ObservationGroupCfg:
    concatenate_terms: bool = True
    enable_corruption: bool = False
    history_length: int | None = None
    flatten_history_dim: bool = True


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


@configclass
class SceneEntityCfg:
    name: str = MISSING