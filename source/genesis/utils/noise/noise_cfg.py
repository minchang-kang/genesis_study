from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal
from genesis_study.source.genesis.utils import configclass

@configclass
class NoiseCfg:
    func: Callable[[torch.Tensor, NoiseCfg], torch.Tensor] = MISSING

    operation: Literal["add", "scale", "abs"] = "add"