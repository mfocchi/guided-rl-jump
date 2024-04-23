from typing import Any

import gymnasium as gym
import math
import numpy as np
import torch

from omni.isaac.version import get_version
from omni.isaac.orbit.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager

from .base_env import BaseEnv, VecEnvObs
from .rl_task_env import RLTaskEnv
from .rl_task_env_cfg import RLTaskEnvCfg

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]

class RLPlanningTaskEnv(RLTaskEnv):
    def __init__(self, cfg: RLTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        # Additional initialization code specific to RLPlanningTaskEnv

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Implement your custom step method here
        # This method will override the step method from RLTaskEnv
        # You can access the parent class's step method using super()
        # and customize the behavior as needed

        # Example:
        # Custom step logic goes here
        # Call the parent class's step method if necessary using super()
        print("Sto usando la nuova classeee")
        return super().step(action)
