from typing import Any

import gymnasium as gym
import math
import numpy as np
import torch

from omni.isaac.version import get_version
from omni.isaac.orbit.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager

from .base_env import BaseEnv, VecEnvObs
from .rl_task_env import RLTaskEnv
from .rl_planning_task_env_cfg import RLPlanningTaskEnvCfg

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]


class RLPlanningTaskEnv(RLTaskEnv):
    def __init__(self, cfg: RLPlanningTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        # ATTENTION: step_dt = max_episode_length_s
        # -> max_episode_length = 1

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps (remove the need for decimation)."""
        return math.ceil(self.max_episode_length_s / self.cfg.sim.dt)

    def load_managers(self):
        super().load_managers()

        self.running_reward_manager = RewardManager(self.cfg.running_rewards, self)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:

        print("#" * 25)
        print("New episode")

        # process actions
        self.action_manager.process_action(action)
        self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        print(f"Executing")
        # perform physics stepping until the timeout
        for i in range(self.max_episode_length):
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            # Enable rendering once every 5 steps
            self.sim.step(render=i % 5 == 0)

            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

            self.episode_length_buf += 1  # sim step in current episode (per env)
            # -- final reward computation
            running_reward = self.running_reward_manager.compute(dt=1).clone()
            # Ignore running reward after t_th
            after_t_th_ids = self.extras.get('after_t_th')
            if len(after_t_th_ids) > 0:
                running_reward[after_t_th_ids] = 0

            self.reward_buf += running_reward

            # -- step interval events
            # dt = self.cfg.sim.dt
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.cfg.sim.dt)

        # perform rendering if gui is enabled
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            self.sim.render()

        print(f"Post execution")

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations (always in timeout)
        self.reset_buf = self.termination_manager.compute()

        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- final reward computation
        # dt=1, because reward is multiplied dt ->
        # value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
        self.reward_buf += self.reward_manager.compute(dt=1)

        # TODO: test if leaving the reward like that allow you to have better results
        self.reward_buf = torch.clip(self.reward_buf, 0)

        # -- reset envs that terminated/timed-out and log the episode information
        # ATTENTION: for us it will always be timeout
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.scene.update(dt=0)
        # -- update command
        # dt = self.cfg.sim.dt, this is the time pased
        self.command_manager.compute(dt=self.cfg.sim.dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
