import genesis as gs
import gymnasium as gym

from genesis_study.source.genesis.envs.manager_based_rl_env import ManagerBasedRLEnv
from genesis_study.source.genesis_tasks.my_task.my_task_env_cfg import TaskEnvCfg

from genesis_study.source.genesis_tasks.my_task.config import my_task_register

# main.py 쪽
def main():
    gs.init(logging_level="warning")

    env = ManagerBasedRLEnv(cfg=TaskEnvCfg())

    # env = gym.make("my_task_env_v0")
    # obs, info = env.reset()
    print("환경 초기화 완료!")
    env.robot.get_links_pos()
    for _ in range(1000):
        env.scene.step()

if __name__ == "__main__":
    main()