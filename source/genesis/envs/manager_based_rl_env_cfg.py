import genesis as gs

from dataclasses import MISSING
from typing import Optional
from genesis_study.source.genesis.utils import configclass

@configclass
class ManagerBasedRLEnvCfg:
    # =========== Scene setting ===========
    # sim_options
    dt: float = 0.01
    substeps: int = 1

    # rigid_options
    constraint_solver: gs.constraint_solver = gs.constraint_solver.Newton
    enable_collision: bool = True
    enable_joint_limit: bool = True

    # vis_options
    rendered_envs_idx: Optional[list] = None

    # viewer_options
    camera_pos: tuple[float, float, float] = (2.0, 0.0, 2.5)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)
    camera_fov: float = 40.0

    # show_viewer
    show_viewer: bool = True

    # show_FPS
    show_FPS: bool = True

    # ========== build setting ==========
    num_envs: int = 0
    env_spacing: tuple[float, float] = (0.0, 0.0)
    device: str = 'cuda:0'

    # ========== Episode setting ==========
    episode_length_s: float = 20.0
    resampling_time_s: float = 2.0
    action_scale: float = 0.25
    simulate_action_latency: bool = True
    clip_actions: float = 1.0


    # ========== Enviroment setting ==========
    # 밑에 처럼 object로 한다면 isaaclab처럼 maanger를 만들어서 관리 해야 한다
    scene: object = MISSING

    commands: object = None

    actions: object = MISSING

    observations: object = MISSING

    evnets: object = None

    rewards: object = MISSING

    terminations: object = MISSING

    curriculum: object = None