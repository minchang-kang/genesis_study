# genesis_action_manager.py

import torch
from abc import ABC, abstractmethod


class GenesisActionTerm(ABC):
    """
    Base class for individual action terms in Genesis.
    Each term controls a part of the environment (e.g., robot arm, gripper).
    """
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.asset = env.scene[cfg.asset_name]

    def reset(self, env_ids=None):
        pass

    @abstractmethod
    def process_actions(self, actions: torch.Tensor):
        pass

    @abstractmethod
    def apply_actions(self):
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        pass


class GenesisActionManager:
    """
    Genesis-style ActionManager without full manager-based overhead.
    """
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.terms = {}
        self.term_order = []

        total_dim = 0
        for name, term_cfg in cfg.__dict__.items():
            if term_cfg is None:
                continue
            term = term_cfg.class_type(term_cfg, env)
            self.terms[name] = term
            self.term_order.append(name)
            total_dim += term.action_dim

        self.total_dim = total_dim
        self.action = torch.zeros((env.num_envs, total_dim), device=env.device)
        self.prev_action = torch.zeros_like(self.action)

    def reset(self, env_ids=None):
        self.action[:] = 0.0
        self.prev_action[:] = 0.0
        for term in self.terms.values():
            term.reset(env_ids)

    def process_action(self, action: torch.Tensor):
        assert action.shape[1] == self.total_dim, (
            f"Invalid action shape. Expected {self.total_dim}, got {action.shape[1]}")
        self.prev_action[:] = self.action
        self.action[:] = action.to(self.env.device)

        idx = 0
        for name in self.term_order:
            term = self.terms[name]
            term_actions = action[:, idx:idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def apply_action(self):
        for term in self.terms.values():
            term.apply_actions()

    @property
    def action_term_dim(self):
        return [term.action_dim for term in self.terms.values()]

    @property
    def total_action_dim(self):
        return self.total_dim
