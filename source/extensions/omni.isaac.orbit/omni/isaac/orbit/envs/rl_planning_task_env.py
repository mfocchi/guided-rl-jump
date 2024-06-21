from typing import Any

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence

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
        self.running_reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps (remove the need for decimation)."""
        return math.ceil(self.max_episode_length_s / self.cfg.sim.dt)

    def load_managers(self):
        super().load_managers()

        self.running_reward_manager = RewardManager(self.cfg.running_rewards, self)
        self.negative_rewards_manager = RewardManager(self.cfg.negative_rewards, self)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:

        print("#" * 25)
        print("New episode")

        # process actions
        self.action_manager.process_action(action)
        self.running_reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

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
            self.running_reward_manager.reset()

            # Ignore running reward after t_th
            after_t_th_ids = self.extras.get('after_t_th')
            if len(after_t_th_ids) > 0:
                running_reward[after_t_th_ids] = 0

            self.running_reward_buf += running_reward

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

        # WARNINGGGG!!!! RESET IS NOT TERMINATED FOR A BAD BEHAVIOUR BUT IT'S ALWAYS IN TIME OUT
        # SINCE IN TD3 I NEED ALWAYS TERMINATED ==  1, clone the time_out flag
        self.reset_time_outs = self.termination_manager.time_outs
        # self.reset_terminated = self.termination_manager.terminated
        self.reset_terminated = self.reset_time_outs
        # -- final reward computation
        # dt=1, because reward is multiplied dt ->
        # value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
        self.reward_buf = self.reward_manager.compute(dt=1)
        self.negative_reward_buf = self.negative_rewards_manager.compute(dt=1)

        self.final_reward_buff = self.reward_buf * torch.exp(-torch.pow(self.running_reward_buf + self.negative_reward_buf, 2))

        # TODO: test if leaving the reward like that allow you to have better results
        # self.reward_buf = torch.clip(self.reward_buf, 0)

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

        # print(f"Reward: {self.final_reward_buff}")
        # print(f"Reward positive: {self.reward_buf}")
        # print(f"Reward negative: {self.negative_reward_buf}")

        # return observations, rewards, resets and extras
        return self.obs_buf, self.final_reward_buff, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(env_ids=env_ids, mode="reset")

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)

        info = self.negative_rewards_manager.reset(env_ids)
        self.extras["log"].update(info)

        _ = self.running_reward_manager.reset(env_ids)
        # Try to log cumulative running costs
        info = {}
        episodic_sum_avg = torch.mean(self.running_reward_buf)
        info["Episode Reward/" + "running costs"] = episodic_sum_avg / self.max_episode_length_s
        self.extras["log"].update(info)

        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0
