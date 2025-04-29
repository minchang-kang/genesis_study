import torch
from typing import Dict

# ================== types ===================
VecEnvObs = Dict[str, torch.Tensor | Dict[str, torch.Tensor]]

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]