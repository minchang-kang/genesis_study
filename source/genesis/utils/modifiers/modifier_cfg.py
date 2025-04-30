import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

from genesis_study.source.genesis.utils import configclass


@configclass
class ModifierCfg:

    func: Callable[..., torch.Tensor] = MISSING

    params: dict[str, Any] = dict()