import torch
import math

import gymnasium as gym

from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg


class ManagerBasedRLEnv(gym.Env):

    def __init__(self, cfg:ManagerBasedRLEnvCfg):
        self.device = torch.device(cfg.device)
        self.max_episode_length = 
        self.cfg = cfg

        self._is_closed = False

        self.extras = {}
        pass

    def reset(self):
        pass

    def step(self): #, action:):
        pass
        # return self.obs_buf, self.reward_buf, self.reset_termination, self.reset_time_outs, self.extras

    def render(self):
        pass

    def close(self):
        # if not self._is_closed:
        #     del self.command_manager
        #     del self.reward_manager
        #     del self.termination_manager
        #     del self.curriculum_manager
            
        #     super().close()
        pass