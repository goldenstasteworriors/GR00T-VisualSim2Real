# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from typing_extensions import override

from gr00t.rl.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from gr00t.rl.utils.teleop_reset_lib.dataset import MotionDataset


class ResetFromDataset(LeggedRobotBase):
    def __init__(self, config, device):
        super().__init__(config, device)
        config.reset_from_dataset.device = device
        self.reset_dataset = MotionDataset(config.reset_from_dataset)
        self.reset_count = 0

        self.groot2motion_dof_idx = [
            self.simulator.dof_names.index(dof_name) for dof_name in self.reset_dataset.dof_names
        ]
        self.groot2motion_body_idx = [
            self.simulator.body_names.index(body_name)
            for body_name in self.reset_dataset.body_names
        ]

    def _post_compute_observations_callback(self):
        self.reset_count += 1
        if self.reset_count % self.config.reset_from_dataset.resample_every == 0:
            self.reset_dataset.resample()
            self.reset_count = 0
        return super()._post_compute_observations_callback()

    @override
    def _reset_robot_states_callback(self, env_ids, target_states=None):
        sample_idx = torch.randint(0, len(self.reset_dataset), (len(env_ids),), device=self.device)
        sample_idx_in_seq = torch.randint(
            0, self.reset_dataset.num_per_sample, (len(env_ids),), device=self.device
        )
        dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel = self.reset_dataset[
            sample_idx
        ]
        dof_pos = dof_pos[torch.arange(len(env_ids)), sample_idx_in_seq]
        dof_vel = dof_vel[torch.arange(len(env_ids)), sample_idx_in_seq]
        body_pos = body_pos[torch.arange(len(env_ids)), sample_idx_in_seq]
        body_rot = body_rot[torch.arange(len(env_ids)), sample_idx_in_seq]
        body_lin_vel = body_lin_vel[torch.arange(len(env_ids)), sample_idx_in_seq]
        body_ang_vel = body_ang_vel[torch.arange(len(env_ids)), sample_idx_in_seq]

        root_states = torch.zeros((len(env_ids), 13), device=self.device)
        root_states[:, :3] = (
            body_pos[:, 0] + self.env_origins[env_ids, :3]
        )  # TODO(haoru): more rigorous check on if the first body is the root
        root_states[:, 3:7] = body_rot[:, 0]

        groot_dof_pos = torch.tile(self.default_dof_pos, (len(env_ids), 1))
        groot_dof_pos[:, self.groot2motion_dof_idx] = dof_pos

        self.target_robot_dof_state[env_ids, :, 0] = groot_dof_pos
        self.target_robot_root_states[env_ids, :] = root_states

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
