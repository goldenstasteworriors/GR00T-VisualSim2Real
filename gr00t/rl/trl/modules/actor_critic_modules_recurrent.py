# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch
from tensordict import TensorDict
from torch.distributions import Normal

from gr00t.rl.trl.modules.actor_critic_modules import Actor, Critic
from gr00t.rl.trl.modules.memory import Memory
from gr00t.rl.trl.utils.common import custom_instantiate


class RecurrentActor(Actor):
    """Recurrent Actor that adds LSTM/GRU memory to the base Actor class."""

    is_recurrent = True

    def __init__(
        self,
        env_config,
        algo_config,
        backbone,
        module_dim_dict={},
        running_mean_std=False,
        max_rollout_history=1,
        input_key="actor_obs",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    ):
        super(RecurrentActor, self).__init__(
            env_config=env_config,
            algo_config=algo_config,
            backbone=backbone,
            module_dim_dict=module_dim_dict,
            running_mean_std=running_mean_std,
            max_rollout_history=max_rollout_history,
            input_key=input_key,
        )

        obs_dim_dict = env_config.robot.algo_obs_dim_dict
        input_dim = obs_dim_dict[self.input_key]

        # Add memory module
        self.memory = Memory(
            input_size=input_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        # Replace the actor module to take rnn_hidden_dim as input instead of obs_dim
        module_dim_dict_recurrent = module_dim_dict.copy()
        module_dim_dict_recurrent[self.input_key] = rnn_hidden_dim
        self.actor_module = custom_instantiate(
            backbone,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict={self.input_key: rnn_hidden_dim},
            module_dim_dict=module_dim_dict_recurrent,
            _resolve=False,
        )

        print(f"RecurrentActor RNN: {self.memory}")
        print(f"RecurrentActor MLP: {self.actor_module}")

    def reset(self, dones=None):
        """Reset memory hidden states for done environments."""
        self.memory.reset(dones)

    def forward(
        self,
        obs_dict,
        masks=None,
        hidden_states=None,
        episode_attnmask=None,
        original_dones=None,
        **kwargs,
    ):
        obs_dict = obs_dict.copy()
        if self.running_mean_std is not None:
            with torch.no_grad():
                obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])

        # Pass through memory module
        # For batch training mode, obs_dict[self.input_key] should have shape [batch_size, seq_len, obs_dim]
        # For rollout mode, it should have shape [batch_size, obs_dim]
        input_obs = obs_dict[self.input_key]

        if len(input_obs.shape) == 2:  # Rollout mode: [batch_size, obs_dim]
            memory_out = self.memory(input_obs)
        else:  # Batch training mode: [batch_size, seq_len, ...] where ... might be multi-dimensional
            batch_size = input_obs.shape[0]
            seq_len = input_obs.shape[1]
            # Flatten all dimensions after seq_len into a single obs_dim
            # [batch_size, seq_len, *extra_dims] -> [batch_size, seq_len, obs_dim]
            input_obs = input_obs.reshape(batch_size, seq_len, -1)
            # Reshape for RNN: [seq_len, batch_size, obs_dim]
            input_obs = input_obs.transpose(0, 1)

            # Process sequences through Memory module
            # If masks provided (trajectory-based training), uses proper episode boundaries
            # If masks=None (non-trajectory training), processes full sequences
            if masks is not None and original_dones is not None:
                # Trajectory-based: use masks and initialize hidden states to zeros
                # Input is already in trajectory format: [seq=max_traj_len, batch=num_trajectories, obs_dim]
                memory_out = self.memory(input_obs, masks=masks, hidden_states=hidden_states)
                # memory_out: [num_trajectories, max_traj_len, hidden_dim]

                # Unsplit back to original [num_envs, num_steps, hidden_dim] format
                from gr00t.rl.trl.utils.rl import unsplit_trajectories

                memory_out = unsplit_trajectories(memory_out, masks, original_dones)
                # memory_out: [num_envs, num_steps, hidden_dim]
            else:
                # CRITICAL: This path should NOT be taken during training!
                # If masks is None during training, hidden states are reset to zeros for every batch
                # which breaks LSTM's ability to learn temporal dependencies
                if self.training:
                    raise RuntimeError(
                        "RecurrentActor: masks and original_dones must be provided during training! "
                        f"Got masks={masks}, original_dones={original_dones}. "
                        "This indicates a bug in _get_mb_rollout_data where trajectory splitting failed."
                    )
                # Non-trajectory: only for inference/rollout
                memory_out = self.memory(input_obs)  # hx=None as positional arg
                # Reshape back: [batch_size, seq_len, hidden_dim]
                memory_out = memory_out.transpose(0, 1)

        if self.algo_config.get("use_clampped_std", False):
            self.std.clamp_(min=self.algo_config.std_clamp_min, max=self.algo_config.std_clamp_max)

        return self.actor_module(memory_out, **kwargs)

    def get_hidden_states(self):
        """Get current hidden states from memory."""
        return self.memory.hidden_states

    def clear_rollout(self):
        """Clear rollout state and detach hidden states for proper TBPTT.

        CRITICAL for TBPTT: After rollout, hidden states must be detached to prevent
        backpropagation through the entire rollout history during training.
        """
        super().clear_rollout()
        # Detach all hidden states to truncate backprop through time
        self.memory.detach_hidden_states()

    def act(
        self,
        obs_dict,
        episode_attnmask=None,
        masks=None,
        hidden_states=None,
        original_dones=None,
        **kwargs,
    ):
        """Forward pass for sampling actions during training."""
        self.update_distribution(
            obs_dict,
            episode_attnmask=episode_attnmask,
            last_step_only=False,
            masks=masks,
            hidden_states=hidden_states,
            original_dones=original_dones,
            **kwargs,
        )
        actions = self.distribution.sample()
        return TensorDict(
            {
                "actions": actions,
                "action_mean": self.action_mean,
                "action_sigma": self.action_std,
            }
        )

    def update_distribution(
        self,
        obs_dict,
        episode_attnmask=None,
        last_step_only=False,
        masks=None,
        hidden_states=None,
        original_dones=None,
        **kwargs,
    ):
        mean = self.forward(
            obs_dict,
            masks=masks,
            hidden_states=hidden_states,
            episode_attnmask=episode_attnmask,
            original_dones=original_dones,
            **kwargs,
        )
        if last_step_only:
            mean = mean[:, -1]

        if self.clamp_noise_std:
            with torch.no_grad():
                self.std.clamp_(max=self.max_noise_std)

        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def rollout(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        """Rollout method for recurrent actor during environment interaction."""
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)

        if self.running_mean_std is not None:
            with torch.no_grad():
                obs_dict = obs_dict.copy()
                obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])

        # For recurrent models, we use the memory hidden states rather than obs buffer
        # Pass current obs through memory and get output
        memory_out = self.memory(obs_dict[self.input_key])

        # Remove sequence dimension added by memory module during inference
        if len(memory_out.shape) == 3:  # [seq_len=1, batch_size, hidden_dim]
            memory_out = memory_out.squeeze(0)  # [batch_size, hidden_dim]

        # Update distribution using memory output
        mean = self.actor_module(memory_out, **kwargs)
        if self.clamp_noise_std:
            with torch.no_grad():
                self.std.clamp_(max=self.max_noise_std)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

        self.steps += 1
        return TensorDict(
            {
                "actions": self.distribution.sample(),
                "action_mean": self.action_mean,
                "action_sigma": self.action_std,
            }
        )

    def act_inference(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        """Inference method for recurrent actor."""
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)

        # For recurrent models, pass through memory
        if self.running_mean_std is not None:
            with torch.no_grad():
                obs_dict = obs_dict.copy()
                obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])

        memory_out = self.memory(obs_dict[self.input_key])

        # Remove sequence dimension added by memory module during inference
        if len(memory_out.shape) == 3:  # [seq_len=1, batch_size, hidden_dim]
            memory_out = memory_out.squeeze(0)  # [batch_size, hidden_dim]

        actions_mean = self.actor_module(memory_out, **kwargs)

        self.steps += 1
        return actions_mean


class RecurrentCritic(Critic):
    """Recurrent Critic that adds LSTM/GRU memory to the base Critic class."""

    is_recurrent = True

    def __init__(
        self,
        env_config,
        algo_config,
        backbone,
        module_dim_dict={},
        running_mean_std=False,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    ):
        super(RecurrentCritic, self).__init__(
            env_config=env_config,
            algo_config=algo_config,
            backbone=backbone,
            module_dim_dict=module_dim_dict,
            running_mean_std=running_mean_std,
        )

        obs_dim_dict = env_config.robot.algo_obs_dim_dict
        input_dim = obs_dim_dict["critic_obs"]

        # Add memory module
        self.memory = Memory(
            input_size=input_dim,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        # Replace the critic module to take rnn_hidden_dim as input instead of obs_dim
        module_dim_dict_recurrent = module_dim_dict.copy()
        module_dim_dict_recurrent["critic_obs"] = rnn_hidden_dim
        self.critic_module = custom_instantiate(
            backbone,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict={"critic_obs": rnn_hidden_dim},
            module_dim_dict=module_dim_dict_recurrent,
            _resolve=False,
        )

        print(f"RecurrentCritic RNN: {self.memory}")
        print(f"RecurrentCritic MLP: {self.critic_module}")

    def reset(self, dones=None):
        """Reset memory hidden states for done environments."""
        self.memory.reset(dones)

    def get_hidden_states(self):
        """Get current hidden states from memory."""
        return self.memory.hidden_states

    def clear_rollout(self):
        """Clear rollout state and detach hidden states for proper TBPTT.

        CRITICAL for TBPTT: After rollout, hidden states must be detached to prevent
        backpropagation through the entire rollout history during training.
        """
        super().clear_rollout()
        # Detach all hidden states to truncate backprop through time
        self.memory.detach_hidden_states()

    def evaluate(
        self,
        obs_dict,
        masks=None,
        hidden_states=None,
        episode_attnmask=None,
        original_dones=None,
        **kwargs,
    ):
        obs_dict = obs_dict.copy()
        if self.running_mean_std is not None:
            with torch.no_grad():
                obs_dict["critic_obs"] = self.running_mean_std(obs_dict["critic_obs"])

        # Pass through memory module
        # For batch training mode, obs_dict['critic_obs'] should have shape [batch_size, seq_len, obs_dim]
        # For rollout mode, it should have shape [batch_size, obs_dim]
        input_obs = obs_dict["critic_obs"]

        if len(input_obs.shape) == 2:  # Rollout mode: [batch_size, obs_dim]
            memory_out = self.memory(input_obs)
            # Remove sequence dimension added by memory module during inference
            if len(memory_out.shape) == 3:  # [seq_len=1, batch_size, hidden_dim]
                memory_out = memory_out.squeeze(0)  # [batch_size, hidden_dim]
        else:  # Batch training mode: [batch_size, seq_len, ...] where ... might be multi-dimensional
            batch_size = input_obs.shape[0]
            seq_len = input_obs.shape[1]
            # Flatten all dimensions after seq_len into a single obs_dim
            # [batch_size, seq_len, *extra_dims] -> [batch_size, seq_len, obs_dim]
            input_obs = input_obs.reshape(batch_size, seq_len, -1)
            # Reshape for RNN: [seq_len, batch_size, obs_dim]
            input_obs = input_obs.transpose(0, 1)

            # Process sequences through Memory module
            # If masks provided (trajectory-based training), uses proper episode boundaries
            # If masks=None (non-trajectory training), processes full sequences
            if masks is not None and original_dones is not None:
                # Trajectory-based: use masks and initialize hidden states to zeros
                # Input is already in trajectory format: [seq=max_traj_len, batch=num_trajectories, obs_dim]
                memory_out = self.memory(input_obs, masks=masks, hidden_states=hidden_states)
                # memory_out: [num_trajectories, max_traj_len, hidden_dim]

                # Unsplit back to original [num_envs, num_steps, hidden_dim] format
                from gr00t.rl.trl.utils.rl import unsplit_trajectories

                memory_out = unsplit_trajectories(memory_out, masks, original_dones)
                # memory_out: [num_envs, num_steps, hidden_dim]
            else:
                # CRITICAL: This path should NOT be taken during training!
                # If masks is None during training, hidden states are reset to zeros for every batch
                # which breaks LSTM's ability to learn temporal dependencies
                if self.training:
                    raise RuntimeError(
                        "RecurrentCritic: masks and original_dones must be provided during training! "
                        f"Got masks={masks}, original_dones={original_dones}. "
                        "This indicates a bug in _get_mb_rollout_data where trajectory splitting failed."
                    )
                # Non-trajectory: only for inference/rollout
                memory_out = self.memory(input_obs)  # hx=None as positional arg
                # Reshape back: [batch_size, seq_len, hidden_dim]
                memory_out = memory_out.transpose(0, 1)

        value = self.critic_module(memory_out, **kwargs)
        return value
