from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from genesis_study.source.genesis.managers.manager_term_cfg import ManagerTermBaseCfg

if TYPE_CHECKING:
    from genesis_study.source.genesis.envs.manager_based_rl_env import ManagerBasedRLEnv


class ManagerTermBase(ABC):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        
        self.cfg = cfg
        self._env = env


    @property
    def num_envs(self) -> int:
        return self._env.num_envs
    


    @property
    def device(self) -> str:
        return self._env.device
    


    def reset(self, env_idx: Sequence[int] | None = None) -> None:
        pass

    def __call__(self, *args) -> Any:
        raise NotImplementedError("The method '__call__' should be implemented by the subclass.")