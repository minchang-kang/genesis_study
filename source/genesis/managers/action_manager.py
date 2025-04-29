reset 정의 필요
process_action
apply_action

import torch

from ..envs.manager_based_rl_env import ManagerBasedRLEnv


class ActionManager(MangerBase):
    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        if cfg is None:
            raise ValueError("Action manager configuration is None. Please provide a valid configuration.")
        
        self._action = torch.zeros((self.num_envs, self.totoal_action_dim), device=self.device)







    def reset(self):
        pass

    def process_action(self):
        pass

    def apply_action(self):
        pass




    def _prepare_terms(self):
        pass