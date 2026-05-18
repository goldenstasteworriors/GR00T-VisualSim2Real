# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
from typing_extensions import override

from gr00t.rl.envs.legged_base_task.legged_robot_base import LeggedRobotBase


class DeltaActionBase(LeggedRobotBase):
    def __init__(self, config, device):
        self._delta_action_indices = torch.tensor(
            config.delta_action_indices, dtype=torch.long, device=device
        )
        self._delta_action_scale = config.delta_action_scale
        super().__init__(config, device)

    @override
    def _init_buffers(self):
        super()._init_buffers()
        self._delta_actions = torch.zeros(
            self.num_envs, len(self._delta_action_indices), device=self.device, requires_grad=False
        )
        self._last_delta_actions = torch.zeros(
            self.num_envs, len(self._delta_action_indices), device=self.device, requires_grad=False
        )

    # @override
    # def step(self, actor_state):
    #     actions = actor_state["actions"]
    #     delta_actions_clip = self.config.get("delta_action_clip", 100.0)
    #     self._last_delta_actions[:] = actions[:, self._delta_action_indices]
    #     if self.config.get("no_delta_vel", False):
    #         self._delta_actions[:, :3] = actions[:, :3]
    #         self._delta_actions[:, 3:11] += actions[:, 3:11] * self._delta_action_scale
    #     else:
    #         self._delta_actions += actions[:, self._delta_action_indices] * self._delta_action_scale
    #     self._delta_actions = torch.clamp(self._delta_actions, -delta_actions_clip, delta_actions_clip)
    #     if self.config.get("print_delta_actions", False):
    #         logger.info(f"delta_actions: {actions[:, self._delta_action_indices]}")
    #         logger.info(f"cumulative_actions: {self._delta_actions}")
    #         logger.info(f"episode_length: {self.episode_length_buf}")
    #         logger.warning(f"stage: {self.stage_buf}")
    #     actor_state["actions"][:, self._delta_action_indices] = self._delta_actions

    #     if self.config.get("zero_vel", False):
    #         zero_env_ids = ((self.stage_buf >= 1) & (self.stage_buf < 4)).nonzero(as_tuple=False).squeeze(-1)
    #         actor_state["actions"][zero_env_ids, :3] = 0.0
    #         # xy_zero_env_ids = (self.stage_buf == 4).nonzero(as_tuple=False).squeeze(-1)
    #         # actor_state["actions"][xy_zero_env_ids, :2] = 0.0
    #     return super().step(actor_state)

    @override
    def step(self, actor_state):
        actions = actor_state["actions"]
        delta_actions_clip = self.config.get("delta_action_clip", 100.0)
        self._last_delta_actions[:] = actions[:, self._delta_action_indices]
        self._delta_actions += actions[:, self._delta_action_indices] * self._delta_action_scale
        self._delta_actions = torch.clamp(
            self._delta_actions, -delta_actions_clip, delta_actions_clip
        )

        # Configration - 0
        # self._delta_actions[:, :3] *= 0.0
        # self._delta_actions[:, 3:10] *= 0.0
        # self._delta_actions[:, 3] = -4.0
        # self._delta_actions[:, 4] = -2.0
        # self._delta_actions[:, 6] = 3.0
        # self._delta_actions[:, 10] = 1.0

        # Configration - 1
        # self._delta_actions[:, :3] *= 0.0
        # self._delta_actions[:, 3:10] *= 0.0
        # self._delta_actions[:, 10] = 1.0

        # Configration - 2
        # self._delta_actions[:, :3] *= 0.0
        # self._delta_actions[:, 3:10] *= 0.0
        # self._delta_actions[:, 3] = -3.0
        # self._delta_actions[:, 4] = -1.0
        # self._delta_actions[:, 6] = 2.0
        # self._delta_actions[:, 10] = 1.0

        actor_state["actions"][:, self._delta_action_indices] = self._delta_actions

        # if self.config.get("no_delta_vel", False):
        #     actor_state["actions"][:, :3] = actions[:, :3]
        if self.config.get("zero_vel", False):
            zero_env_ids = (
                ((self.stage_buf >= 1) & (self.stage_buf < 4)).nonzero(as_tuple=False).squeeze(-1)
            )

            # For teacher training mode: zero out policy actions
            if "gt_actions" not in actor_state:
                # Teacher training mode: zero out policy actions
                actor_state["actions"][zero_env_ids, :3] = 0.0
                logger.warning(
                    "overwriting policy actions with zero vel in delta action base STEP"
                )
            # For distillation mode: actions need to be zeroed out if enforce_teacher_rollout is True
            # else:
            #     # Check if enforce_teacher_rollout is True (set by trainer from algo config)
            #     if getattr(self, "enforce_teacher_rollout", False):
            #         actor_state["actions"][zero_env_ids, :3] = 0.0

        if self.config.get("zero_finger", False):
            if "gt_actions" not in actor_state:
                zero_env_ids = (
                    ((self.stage_buf == 0) | (self.stage_buf == 1))
                    .nonzero(as_tuple=False)
                    .squeeze(-1)
                )
                actor_state["actions"][zero_env_ids, 10] = 0.5

        if self.config.get("print_delta_actions", False):
            logger.info(f"policy actions: {self._last_delta_actions}")
            logger.info(f"cumulative_actions: {self._delta_actions}")
            logger.info(f"effective actions on envs: {actor_state['actions']}")
            logger.info(f"episode_length: {self.episode_length_buf}")
            logger.warning(f"stage: {self.stage_buf}")
        return super().step(actor_state)

    def _reward_penalty_delta_action_rate(self):
        return torch.sum(torch.square(self._last_delta_actions), dim=-1)

    def _get_obs_delta_actions(self):
        return self._last_delta_actions

    def _store_delta_actions_buffer(self, env_ids):
        return self._delta_actions[env_ids]

    def _load_delta_actions_buffer(self, env_ids, data):
        self._delta_actions[env_ids] = data

    def _get_delta_actions_buffer_shape(self):
        return self._delta_actions.shape

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        self._delta_actions[env_ids] = 0.0
        self._last_delta_actions[env_ids] = 0.0
        return super()._reset_buffers_callback(env_ids, target_buf)

    def _post_reset_callback(self, env_ids):
        if self.config.get("reset_delta_actions_with_backmap", False):
            # BUG BUG BUG
            pass

            # xx, yy = torch.meshgrid(env_ids, self._delta_action_indices)
            # self._delta_actions[env_ids] = self._action_backmap()[xx, yy]

            # logger.warning(f"reset_delta_actions_with_backmap: {self._delta_actions[env_ids]}")

            # import ipdb; ipdb.set_trace()

            # logger.warning(f"done")
