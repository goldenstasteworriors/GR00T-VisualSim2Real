# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from typing_extensions import override

from gr00t.rl.envs.legged_base_task.legged_robot_base import LeggedRobotBase


class WarpedActionBase(LeggedRobotBase):
    def __init__(self, config, device):
        super().__init__(config, device)

        self._k = config.warped_action.k
        self._s = config.warped_action.s
        self._warped_action_indices = torch.tensor(
            config.warped_action.indices, dtype=torch.long, device=device
        )

    @override
    def _init_buffers(self):
        super()._init_buffers()
        self._unwarped_actions = torch.zeros(
            self.num_envs,
            len(self.config.warped_action.indices),
            device=self.device,
            requires_grad=False,
        )

    @override
    def step(self, actor_state):
        self._unwarped_actions[:] = actor_state["actions"][:, self._warped_action_indices].clone()
        if self._k == 0 and self._s == 0:
            return super().step(actor_state)
        actor_state["actions"][:, self._warped_action_indices] = self._warp_actions(
            actor_state["actions"][:, self._warped_action_indices]
        )
        return super().step(actor_state)

    def _warp_actions(self, actions):
        abs_actions = actions.abs()
        return actions * (self._k + (1 - self._k) * abs_actions / (self._s + abs_actions))

    def _get_obs_unwarped_actions(self):
        return self._unwarped_actions
