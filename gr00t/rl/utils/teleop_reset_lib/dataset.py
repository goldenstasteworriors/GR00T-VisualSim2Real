# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Optional

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import Dataset

from gr00t.rl.utils.teleop_reset_lib.motion_loader import MotionLoader


class MotionDataset(Dataset):
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.motion_files = config.motion_files
        self.device = config.device
        # make sure motion_
        if getattr(self.config, "use_motion_file_dir", False):
            import os

            motion_dir = self.config.motion_file_dir
            all_motion_files_under_dir = [
                os.path.join(motion_dir, f) for f in os.listdir(motion_dir) if f.endswith(".npz")
            ]
            logger.info(
                f"number of motion files under dir {motion_dir}: {len(all_motion_files_under_dir)}"
            )
            logger.info(f"motion files: {all_motion_files_under_dir}")
            logger.info(f"overriding motion files with motion files under dir {motion_dir}")
            self.motion_files = all_motion_files_under_dir
        self.motion_loaders = [
            MotionLoader.from_file(motion_file, self.device) for motion_file in self.motion_files
        ]
        logger.info(f"number of motion loaders: {len(self.motion_loaders)}")
        assert len(self.motion_loaders) > 0, "No motion files provided"
        self.body_names = self.motion_loaders[0].body_names
        self.dof_names = self.motion_loaders[0].dof_names
        self.num_per_sample = config.num_per_sample
        self.sample_interval_s = config.sample_interval_s
        self.samples_robot: Optional[
            tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            ]
        ] = None
        self.samples_table_grid: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self.samples_bottle: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self.samples_homie_command: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self.resample()

    def resample(self) -> None:
        samples_robot = []
        samples_table_grid = []
        samples_bottle = []
        samples_homie_command = []
        for i, motion_loader in enumerate(self.motion_loaders):
            duration_per_sample = self.num_per_sample * self.sample_interval_s
            if duration_per_sample * 2 > motion_loader.duration:
                logging.warning(
                    f"Motion loader {i} has duration {motion_loader.duration}s, which is less than the sample duration {duration_per_sample}s * 2"
                )
                continue
            num_samples = int((motion_loader.duration - duration_per_sample) / duration_per_sample)
            sample_times = (
                np.arange(0, num_samples * self.num_per_sample) * self.sample_interval_s
                + np.random.rand() * duration_per_sample
            )
            dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel = motion_loader.sample(
                0, sample_times
            )
            dof_pos = dof_pos.reshape(num_samples, self.num_per_sample, -1)
            dof_vel = dof_vel.reshape(num_samples, self.num_per_sample, -1)
            body_pos = body_pos.reshape(num_samples, self.num_per_sample, -1, 3)
            body_rot = body_rot.reshape(num_samples, self.num_per_sample, -1, 4)
            body_lin_vel = body_lin_vel.reshape(num_samples, self.num_per_sample, -1, 3)
            body_ang_vel = body_ang_vel.reshape(num_samples, self.num_per_sample, -1, 3)
            samples_robot.append((dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel))

            (
                table_grid_body_pos,
                table_grid_body_rot,
                table_grid_body_lin_vel,
                table_grid_body_ang_vel,
            ) = motion_loader.sample_table_grid()
            table_grid_body_pos = table_grid_body_pos.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            table_grid_body_rot = table_grid_body_rot.reshape(
                num_samples, self.num_per_sample, -1, 4
            )
            table_grid_body_lin_vel = table_grid_body_lin_vel.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            table_grid_body_ang_vel = table_grid_body_ang_vel.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            samples_table_grid.append(
                (
                    table_grid_body_pos,
                    table_grid_body_rot,
                    table_grid_body_lin_vel,
                    table_grid_body_ang_vel,
                )
            )

            bottle_body_pos, bottle_body_rot, bottle_body_lin_vel, bottle_body_ang_vel = (
                motion_loader.sample_bottle()
            )
            bottle_body_pos = bottle_body_pos.reshape(num_samples, self.num_per_sample, -1, 3)
            bottle_body_rot = bottle_body_rot.reshape(num_samples, self.num_per_sample, -1, 4)
            bottle_body_lin_vel = bottle_body_lin_vel.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            bottle_body_ang_vel = bottle_body_ang_vel.reshape(
                num_samples, self.num_per_sample, -1, 3
            )
            samples_bottle.append(
                (bottle_body_pos, bottle_body_rot, bottle_body_lin_vel, bottle_body_ang_vel)
            )

            homie_commands, target_dof_positions, finger_primitive_actions = (
                motion_loader.sample_homie_command()
            )
            homie_commands = homie_commands.reshape(num_samples, self.num_per_sample, 7)
            target_dof_positions = target_dof_positions.reshape(
                num_samples, self.num_per_sample, -1
            )  # 43dof
            finger_primitive_actions = finger_primitive_actions.reshape(
                num_samples, self.num_per_sample, -1
            )  # 2dof
            samples_homie_command.append(
                (homie_commands, target_dof_positions, finger_primitive_actions)
            )

        self.samples_robot = tuple(
            torch.cat([sample[i] for sample in samples_robot], dim=0) for i in range(6)
        )
        self.samples_table_grid = tuple(
            torch.cat([sample[i] for sample in samples_table_grid], dim=0) for i in range(4)
        )
        self.samples_bottle = tuple(
            torch.cat([sample[i] for sample in samples_bottle], dim=0) for i in range(4)
        )
        self.samples_homie_command = tuple(
            torch.cat([sample[i] for sample in samples_homie_command], dim=0) for i in range(3)
        )

    def __len__(self) -> int:
        return self.samples_robot[0].shape[0]

    def __getitem__(self, index):
        """Index into robot samples (dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel)."""
        return tuple(sample[index] for sample in self.samples_robot)

    def get_item_robot_sample(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(sample[index] for sample in self.samples_robot)

    def get_item_table_grid_sample(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(sample[index] for sample in self.samples_table_grid)

    def get_item_bottle_sample(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(sample[index] for sample in self.samples_bottle)

    def get_item_homie_command_sample(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(sample[index] for sample in self.samples_homie_command)


@hydra.main(config_path="../configs", config_name="train_dataset.yaml", version_base=None)
def test_motion_dataset(config: DictConfig) -> None:
    dataset = MotionDataset(config.dataset)
    print(dataset.samples[0].shape)


if __name__ == "__main__":
    test_motion_dataset()
