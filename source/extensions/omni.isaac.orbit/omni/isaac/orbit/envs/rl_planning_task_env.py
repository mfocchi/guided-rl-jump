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
        # ATTENTION: step_dt = max_episode_length_s
        # -> max_episode_length = 1

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps (remove the need for decimation)."""
        return math.ceil(self.max_episode_length_s / self.cfg.sim.dt)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # process actions
        self.action_manager.process_action(action)

        # perform physics stepping until the timeout
        for _ in range(self.max_episode_length):
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            # TODO: Enable rendering in case of play? How does this affect the performance during training?
            self.sim.step(render=True)
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            # TODO: add running costs
            self.episode_length_buf += 1  # sim step in current episode (per env)

            # -- step interval events
            # dt = self.cfg.sim.dt
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.cfg.sim.dt)

        # perform rendering if gui is enabled
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            self.sim.render()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations (always in timeout)
        self.reset_buf = self.termination_manager.compute()

        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        # dt=1, because reward is multiplied dt ->
        # value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
        self.reward_buf = self.reward_manager.compute(dt=1)

        # -- reset envs that terminated/timed-out and log the episode information
        # ATTENTION: for us it will always be timeout
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
        # -- update command
        # dt = self.cfg.sim.dt, this is the time pased
        self.command_manager.compute(dt=self.cfg.sim.dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        # TODO: check if it's needed before the reset in our case
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
