import math
import torch
import numpy as np
import gymnasium as gym
from typing import Any, ClassVar, Sequence

import genesis as gs
from genesis_study.source.genesis.assets.asset_base_cfg import *
from genesis_study.source.genesis.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from genesis_study.source.genesis.utils.asset_parser import create_scene_entity_from_cfg
from genesis_study.source.genesis.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager, RewardManager, ActionManager, ObservationManager, EventManager
from genesis_study.source.genesis.envs.common import VecEnvStepReturn, VecEnvObs


class ManagerBasedRLEnv(gym.Env):
    """Genesis 스타일 Manager 기반 강화학습 환경 (벡터라이즈드 환경)."""

    is_vector_env: ClassVar[bool] = True
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
    }

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """환경 초기화"""
        # 기본 config 저장
        self.cfg = cfg
        self.render_mode = render_mode
        self._is_closed = False

        # simulation counter
        self._sim_step_counter = 0

        # curriculum counter
        self.common_step_counter = 0

        # info 저장 공간 설정
        self.extras = {}

        # Scene, add, build 도 모듈화 가능해 보임
        # Scene 설정 
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.cfg.dt, substeps=self.cfg.substeps),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=self.cfg.constraint_solver,
                enable_collision=self.cfg.enable_collision,
                enable_joint_limit=self.cfg.enable_joint_limit,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.cfg.dt),
                camera_pos=self.cfg.camera_pos,
                camera_lookat=self.cfg.camera_lookat,
                camera_fov=self.cfg.camera_fov,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=self.cfg.rendered_envs_idx),
            show_viewer=self.cfg.show_viewer,
            show_FPS=self.cfg.show_FPS,
        )

        # Scene 안에 Plane, Robot, Light 추가
        self._create_scene_entities()

        # 디바이스 설정
        self.device = torch.device(self.cfg.device)

        # Step 관련 시간 계산
        self.step_dt = self.cfg.dt * self.cfg.substeps
        self.physics_dt = self.cfg.dt

        # Agent num 설정
        self.num_envs = self.cfg.num_envs

        # Buffer 초기화 -
        """  
        이렇게 하는 이유?
        수천 수만번 반복하는 환경에서는 메소드 내에서 생성하는것 보다는 init에서 정의를 해두는것이
        속도 측면에서 더 빠르다고 한다. -- chat.gpt --
        """
        # self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.reward_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        # Manager 초기화
        # self.load_managers()

        # Physics 빌드
        self.scene.build(n_envs=self.cfg.num_envs)

        # self.event_manager = EventManager(self.cfg.events, self)

        print("[INFO] Genesis ManagerBasedRLEnv 초기화 완료.")

        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tEnvironment step-size : {self.step_dt}")

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)




    def _create_scene_entities(self):
        for name, asset_cfg in self.cfg.scene.__dict__.items():
            if asset_cfg is None:
                continue
            entity = create_scene_entity_from_cfg(self.scene, name, asset_cfg)
            if entity is not None:
                setattr(self, name, entity)


    # def load_managers(self):
    #     # note: this order is important since observation manager needs to know the command and action managers
    #     # and the reward manager needs to know the termination manager
    #     print("[INFO] Event Manager: ", self.event_manager)

    #     # -- command manager
    #     self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
    #     print("[INFO] Command Manager: ", self.command_manager)

    #     # prepare the managers
    #     # -- termination manager
    #     self.termination_manager = TerminationManager(self.cfg.terminations, self)
    #     print("[INFO] Termination Manager: ", self.termination_manager)

    #     # -- reward manager
    #     self.reward_manager = RewardManager(self.cfg.rewards, self)
    #     print("[INFO] Reward Manager: ", self.reward_manager)

    #     # -- curriculum manager
    #     self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
    #     print("[INFO] Curriculum Manager: ", self.curriculum_manager)

    #     # print("[INFO] Recorder Manager: ", self.recorder_manager)

    #     # -- action manager
    #     self.action_manager = ActionManager(self.cfg.actions, self)
    #     print("[INFO] Action Manager: ", self.action_manager)

    #     # -- observation manager
    #     self.observation_manager = ObservationManager(self.cfg.observations, self)
    #     print("[INFO] Observation Manager:", self.observation_manager)

    #     # setup the action and observation spaces for Gym
    #     self._configure_gym_env_spaces()

    #     # perform events at the start of the simulation
    #     if "startup" in self.event_manager.available_modes:
    #         self.event_manager.apply(mode="startup")


    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            # check if group is concatenated or not
            # if not concatenated, then we need to add each term separately as a dictionary
            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                self.single_observation_space[group_name] = gym.spaces.Dict({
                    term_name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=term_dim)
                    for term_name, term_dim in zip(group_term_names, group_dim)
                })
        # action space (unbounded since we don't impose any limits)
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)


    # articulation kenematics, recorder 부분 일단 스킵
    def reset(self, env_ids: Sequence[int] | None = None) -> tuple[VecEnvObs, dict]:
        """Reset the entire environment (Genesis version)."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        
        # Reset buffer (모든 env를 리셋 준비)
        self.reset_buf[:] = True

        # Reset environments
        self._reset_idx(env_ids)


        # Update observations
        self.obs_buf = self.observation_manager.compute()

        return self.obs_buf, self.extras


    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset only selected environments (Genesis version)."""

        self.command_manager.compute(dt=self.step_dt)

        self.scene.reset(env_ids)

        # 추가 설정 필요 eventmanager 작설할 때
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(mode="reset", env_ids=torch.arange(self.num_envs, device=self.device))

        self.extras["log"] = dict()

        managers = [
            self.observation_manager,
            self.action_manager,
            self.reward_manager,
            self.curriculum_manager,
            self.command_manager,
            self.event_manager,
            self.termination_manager,
            self.recorder_manager,
        ]

        for manager in managers:
            info = manager.reset(env_ids)
            self.extras["log"].update(info)

        self.episode_length_buf[env_ids] = 0

        # 4. Post-reset, Physics Update
        # self.scene.write_data_to_sim()
        # self.scene.forward()

        # 5. Rerender if needed
        # if self.scene.has_sensors() and self.cfg.rerender_on_reset:
        #     self.scene.render()


    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment dynamics (Genesis version)."""

        # 1. Action processing
        self.action_manager.process_action(action.to(self.device))

        # 2. Pre-Step record (optional, recording)
        self.recorder_manager.record_pre_step()

        # 3. Physics stepping
        is_rendering = self.scene.show_viewer or self.scene.has_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            # Apply action -> Write to sim -> Step physics
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.scene.step()

            # Optional rendering if necessary
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.scene.render()

            # Update all entities (states, sensors)
            self.scene.update(dt=self.physics_dt)

        # 4. Update counters
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # 5. Check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # 6. Compute rewards
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # 7. Reset environments if terminated
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.scene.forward()
            if self.scene.has_sensors() and self.cfg.rerender_on_reset:
                self.scene.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # 8. Update commands
        self.command_manager.compute(dt=self.step_dt)

        # 9. Interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 10. Update observations
        self.obs_buf = self.observation_manager.compute()

        # 11. Post-step record (optional)
        if len(self.recorder_manager.active_terms) > 0:
            self.recorder_manager.record_post_step()

        # 12. Return standard outputs
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    # 일단 보류
    def render(self):
        pass


    def close(self):
        if not self._is_closed:
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager
            del self.action_manager
            del self.observation_manager
            del self.event_manager
            del self.scene
            # del self.viewport_camera_controller
            # del self.recorder_manager
            # # clear callbacks and instance
            # self.sim.clear_all_callbacks()
            # self.sim.clear_instance()
            # # destroy the window
            # if self._window is not None:
            #     self._window = None
            self._is_closed = True