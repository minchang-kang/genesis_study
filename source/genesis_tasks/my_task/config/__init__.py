import gymnasium as gym
import os

from . import (
    agents,
    my_task_joint_pos_env_cfg
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Genesis-My-Task-v0",
    entry_point="genesis_study.genesis.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": my_task_joint_pos_env_cfg.MyTaskEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)