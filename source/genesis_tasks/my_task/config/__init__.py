import gymnasium as gym
import os

from . import (
    agents,
    my_task_joint_pos_env_cfg
)
from .. import my_task_env_cfg

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Genesis-My-Task-v0",
    entry_point="genesis_study.source.genesis.envs.manager_based_rl_env:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": my_task_joint_pos_env_cfg.MyTaskEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)


gym.register(
    id = "my_task_env_v0",
    entry_point = "genesis_study.source.genesis.envs.manager_based_rl_env:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": my_task_env_cfg.TaskEnvCfg
    },
    disable_env_checker=True,
)