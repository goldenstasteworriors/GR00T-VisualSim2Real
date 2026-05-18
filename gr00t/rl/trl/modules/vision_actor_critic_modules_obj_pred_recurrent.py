# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch
from hydra.utils import instantiate
from torch.distributions import Normal

from gr00t.rl.trl.modules.vision_actor_critic_modules_recurrent import VisionRecurrentActor


class VisionRecurrentActorObjPred(VisionRecurrentActor):
    """
    Vision Recurrent Actor with object position prediction capability.
    Combines LSTM memory with vision processing and object position prediction.
    """

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
        state_history_fusion_strategy="transformer",  # "concat" or "transformer"
        use_temporal_attention=False,  # Enable temporal attention for vision features
    ):
        # Initialize the VisionRecurrentActor
        super(VisionRecurrentActorObjPred, self).__init__(
            env_config=env_config,
            algo_config=algo_config,
            backbone=backbone,
            module_dim_dict=module_dim_dict,
            running_mean_std=running_mean_std,
            max_rollout_history=max_rollout_history,
            input_key=input_key,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
        )

        # Initialize object position prediction MLP
        obs_dim_dict = env_config.robot.algo_obs_dim_dict

        # Create object position prediction MLP
        self.obj_pred_mlp = instantiate(
            backbone.obj_pred_mlp,
            env_config=env_config,
            algo_config=algo_config,
            obs_dim_dict=obs_dim_dict,
            module_dim_dict=module_dim_dict,
            _recursive_=False,
        )

        self.state_history_fusion_strategy = state_history_fusion_strategy
        self.state_history_module = None
        self.use_temporal_attention = use_temporal_attention

        state_history_module_config = getattr(backbone, "state_history_module", None)
        if state_history_module_config is not None:
            self.state_history_module = instantiate(
                state_history_module_config,
                env_config=env_config,
                algo_config=algo_config,
                obs_dim_dict=obs_dim_dict,
                module_dim_dict=module_dim_dict,
                _recursive_=False,
            )

        # Initialize temporal attention module for vision features
        self.temporal_attention_module = None
        temporal_attention_config = getattr(backbone, "temporal_attention_module", None)
        if self.use_temporal_attention and temporal_attention_config is not None:
            # Get vision feature dimension from the vision module
            vision_feature_dim = module_dim_dict.get("vision_feature_dim", 512)

            self.temporal_attention_module = instantiate(
                temporal_attention_config,
                env_config=env_config,
                algo_config=algo_config,
                mode="temporal_vision",
                query_dim=vision_feature_dim,
                key_value_dim=vision_feature_dim,
                _recursive_=False,
            )

        print(f"VisionRecurrentActorObjPred RNN: {self.memory}")
        print(f"VisionRecurrentActorObjPred MLP: {self.mlp_module}")
        print(f"VisionRecurrentActorObjPred ObjPred MLP: {self.obj_pred_mlp}")

    def forward(self, obs_dict, masks=None, hidden_states=None, episode_attnmask=None, **kwargs):
        """
        Forward pass that returns both action predictions and object position predictions.
        Handles state_history_obs if present.
        """
        if self.running_mean_std is not None:
            obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])

        if (
            self.state_history_fusion_strategy is not None
            and self.state_history_module is not None
            and any(key for key in obs_dict.keys() if "state_history_obs" in key)
        ):

            obs_dict_copy = {
                "actor_obs": obs_dict["actor_obs"],
                "state_history_obs": obs_dict["state_history_obs"],
            }

            # Use state history module to process proprioceptive history
            processed_obs = self.state_history_module(obs_dict_copy)
            obs_dict[self.input_key] = processed_obs

        # Handle both vision_obs and rgb_image_history
        if "rgb_image_history" in obs_dict:
            image_input = obs_dict["rgb_image_history"].clone()
        elif "vision_obs" in obs_dict:
            image_input = obs_dict["vision_obs"].clone()
        else:
            raise ValueError("Neither 'rgb_image_history' nor 'vision_obs' found in obs_dict")

        original_shape = image_input.shape

        if len(original_shape) == 5:
            batch_size, seq_len = original_shape[0], original_shape[1]
            image_reshaped = image_input.reshape(-1, *original_shape[2:])
            effective_batch_size = batch_size * seq_len
        elif len(original_shape) == 6:  # Batch training mode: [batch_size, seq_len, H, W, C]
            batch_size, seq_len = original_shape[0], original_shape[1]
            image_reshaped = image_input.reshape(-1, *original_shape[2:])
            effective_batch_size = batch_size * seq_len
        else:
            batch_size = original_shape[0]
            seq_len = None
            image_reshaped = image_input
            effective_batch_size = batch_size

        encoder_type = self.vision_module_config_dict.layer_config.type

        if encoder_type in ["CNN", "ResNet"]:
            image_permuted = image_reshaped.permute(0, 3, 1, 2)
            latent = self.vision_module(image_permuted).reshape(effective_batch_size, -1)
        else:
            flat_image = image_reshaped.reshape(effective_batch_size, -1).contiguous()
            latent = self.vision_module(flat_image).reshape(effective_batch_size, -1)

        if seq_len is not None:
            latent = latent.reshape(batch_size, seq_len, -1).contiguous()
            actor_obs = obs_dict[self.input_key]  # Should have shape [batch_size, seq_len, obs_dim]
        else:
            actor_obs = obs_dict[self.input_key]  # Should have shape [batch_size, obs_dim]

        # Apply temporal attention to vision features if enabled
        if (
            self.use_temporal_attention
            and self.temporal_attention_module is not None
            and seq_len is not None
        ):
            # Apply temporal attention across the sequence dimension
            # latent shape: [batch_size, seq_len, vision_feature_dim]
            latent_attended = self.temporal_attention_module(latent, latent, latent)
            latent = latent_attended

        # Concatenate observations and vision features
        concated_obs = torch.cat([actor_obs, latent], dim=-1)

        # Pass through memory module
        if len(concated_obs.shape) == 2:  # Rollout mode: [batch_size, concat_dim]
            memory_out = self.memory(concated_obs)
        else:  # Batch training mode: [batch_size, seq_len, concat_dim]
            batch_size, seq_len, concat_dim = concated_obs.shape
            # Reshape for RNN: [seq_len, batch_size, concat_dim]
            concated_obs = concated_obs.transpose(0, 1)
            memory_out = self.memory(concated_obs, masks=masks, hidden_states=hidden_states)
            # Reshape back: [batch_size, seq_len, hidden_dim]
            # memory_out = memory_out.transpose(0, 1)

        # Get action predictions from MLP
        actions = self.mlp_module(memory_out)

        obj_pos_pred = self.obj_pred_mlp(latent)
        return {"actions": actions, "obj_pred": obj_pos_pred}

    def predict_object_position(self, obs_dict, **kwargs):
        """
        Predict object position from vision features.
        """
        # Handle both vision_obs and rgb_image_history
        if "rgb_image_history" in obs_dict:
            image_input = obs_dict["rgb_image_history"].clone()
        elif "vision_obs" in obs_dict:
            image_input = obs_dict["vision_obs"].clone()
        else:
            raise ValueError("Neither 'rgb_image_history' nor 'vision_obs' found in obs_dict")

        original_shape = image_input.shape

        if len(original_shape) == 5:
            batch_size, seq_len = original_shape[0], original_shape[1]
            image_reshaped = image_input.reshape(-1, *original_shape[2:])
            effective_batch_size = batch_size * seq_len
        elif len(original_shape) == 6:  # Batch training mode: [batch_size, seq_len, H, W, C]
            batch_size, seq_len = original_shape[0], original_shape[1]
            image_reshaped = image_input.reshape(-1, *original_shape[2:])
            effective_batch_size = batch_size * seq_len
        else:
            batch_size = original_shape[0]
            seq_len = None
            image_reshaped = image_input
            effective_batch_size = batch_size

        encoder_type = self.vision_module_config_dict.layer_config.type

        if encoder_type in ["CNN", "ResNet"]:
            image_permuted = image_reshaped.permute(0, 3, 1, 2)
            latent = self.vision_module(image_permuted).reshape(effective_batch_size, -1)
        else:
            flat_image = image_reshaped.reshape(effective_batch_size, -1).contiguous()
            latent = self.vision_module(flat_image).reshape(effective_batch_size, -1)

        if seq_len is not None:
            latent = latent.reshape(batch_size, seq_len, -1).contiguous()
            # Take the last timestep for object prediction
            latent = latent[:, -1, :]

        # Get object position prediction from vision features
        obj_pos_pred = self.obj_pred_mlp(latent)
        return obj_pos_pred

    def act_with_obj_pred(
        self, obs_dict, masks=None, hidden_states=None, episode_attnmask=None, **kwargs
    ):
        """
        Action forward pass that also returns object position predictions.
        """
        result = self.forward(
            obs_dict,
            masks=masks,
            hidden_states=hidden_states,
            episode_attnmask=episode_attnmask,
            **kwargs,
        )
        action_mean = result["actions"]
        obj_pos_pred = result["obj_pred"]

        if self.clamp_noise_std:
            with torch.no_grad():
                self.std.clamp_(max=self.max_noise_std)

        self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)

        actions = self.distribution.sample()
        return {
            "actions": actions,
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
            "predicted_obj_pos": obj_pos_pred,
        }

    def action_mean_with_obj_pred(self, obs_dict, episode_attnmask=None, **kwargs):
        """
        Returns deterministic action mean and object position predictions for distillation.
        """
        # Single forward pass to get both action and object position predictions
        result = self.forward(obs_dict, episode_attnmask=episode_attnmask, **kwargs)
        action_mean = result["actions"]
        obj_pos_pred = result["obj_pred"]

        # Create distribution using the cached action mean
        if self.clamp_noise_std:
            with torch.no_grad():
                self.std.clamp_(max=self.max_noise_std)

        self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)

        actions = self.distribution.sample()
        return {
            "actions": self.action_mean,
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
            "predicted_obj_pos": obj_pos_pred,
        }

    def rollout_with_obj_pred(self, obs_dict, episode_attnmask=None, cur_dones=None, **kwargs):
        """
        Rollout method that includes object position prediction.
        Compatible with distillation trainers that expect this method.
        """
        episode_attnmask = self._update_obs_buffer(obs_dict, episode_attnmask, cur_dones)

        if self.running_mean_std is not None:
            with torch.no_grad():
                obs_dict = obs_dict.copy()
                obs_dict[self.input_key] = self.running_mean_std(obs_dict[self.input_key])

        # Get action predictions (similar to rollout but with object prediction)
        # Handle vision processing
        if "rgb_image_history" in obs_dict:
            image_input = obs_dict["rgb_image_history"].clone()
        elif "vision_obs" in obs_dict:
            image_input = obs_dict["vision_obs"].clone()
        else:
            raise ValueError("Neither 'rgb_image_history' nor 'vision_obs' found in obs_dict")

        # Process vision
        original_shape = image_input.shape
        batch_size = original_shape[0]
        image_reshaped = image_input
        effective_batch_size = batch_size

        encoder_type = self.vision_module_config_dict.layer_config.type

        if encoder_type in ["CNN", "ResNet"]:
            image_permuted = image_reshaped.permute(0, 3, 1, 2)
            latent = self.vision_module(image_permuted).reshape(effective_batch_size, -1)
        else:
            flat_image = image_reshaped.reshape(effective_batch_size, -1).contiguous()
            latent = self.vision_module(flat_image).reshape(effective_batch_size, -1)

        # Concatenate and pass through memory
        concated_obs = torch.cat([obs_dict[self.input_key], latent], dim=-1)
        memory_out = self.memory(concated_obs)

        # Remove sequence dimension added by memory module during inference
        if len(memory_out.shape) == 3:  # [seq_len=1, batch_size, hidden_dim]
            memory_out = memory_out.squeeze(0)  # [batch_size, hidden_dim]

        # Get action mean from MLP
        mean = self.mlp_module(memory_out)

        # Get object position prediction
        obj_pos_pred = self.obj_pred_mlp(latent)

        # Update distribution
        if self.clamp_noise_std:
            with torch.no_grad():
                self.std.clamp_(max=self.max_noise_std)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

        self.steps += 1
        return {
            "actions": self.distribution.sample(),
            "action_mean": self.action_mean,
            "action_sigma": self.action_std,
            "predicted_obj_pos": obj_pos_pred,
        }
