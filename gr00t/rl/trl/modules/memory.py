# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


class Memory(nn.Module):
    """Memory module for recurrent networks.

    This module is used to store the hidden states of the policy.
    Currently only supports GRU and LSTM.
    """

    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None, return_new_hidden_states=False):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode: for training with split trajectories
            # This path is for when observations are restructured via split_and_pad_trajectories
            # Initialize hidden states to zeros if not provided
            if hidden_states is None:
                num_layers = self.rnn.num_layers
                batch_size = input.shape[1]  # input is [seq, batch, dim]
                hidden_size = self.rnn.hidden_size

                if isinstance(self.rnn, nn.LSTM):
                    # LSTM needs (h0, c0) tuple
                    h0 = input.new_zeros(num_layers, batch_size, hidden_size)
                    c0 = input.new_zeros(num_layers, batch_size, hidden_size)
                    hidden_states = (h0, c0)
                else:  # GRU
                    hidden_states = input.new_zeros(num_layers, batch_size, hidden_size)
            else:
                # Validate hidden state shapes
                if isinstance(self.rnn, nn.LSTM):
                    assert (
                        isinstance(hidden_states, tuple) and len(hidden_states) == 2
                    ), f"LSTM requires (h, c) tuple, got {type(hidden_states)}"
                    assert hidden_states[0].shape == (
                        self.rnn.num_layers,
                        input.shape[1],
                        self.rnn.hidden_size,
                    ), f"Hidden state shape mismatch: expected {(self.rnn.num_layers, input.shape[1], self.rnn.hidden_size)}, got {hidden_states[0].shape}"

            out, new_hidden_states = self.rnn(input, hidden_states)

            # Check for numerical issues
            if torch.isnan(out).any():
                raise RuntimeError(
                    "NaN detected in LSTM output! This indicates numerical instability."
                )
            if torch.isinf(out).any():
                raise RuntimeError(
                    "Inf detected in LSTM output! This indicates numerical instability."
                )

            # Return with padding - loss computation will handle masking
            # Transpose from [seq, batch, hidden_dim] to [batch, seq, hidden_dim]
            out = out.transpose(0, 1)
            if return_new_hidden_states:
                return out, new_hidden_states
        else:
            # inference mode: processes sequences using maintained hidden states
            # For rollout, maintains LSTM state across timesteps
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None, hidden_states=None):
        if dones is None:  # reset all hidden states
            if hidden_states is None:
                self.hidden_states = None
            else:
                self.hidden_states = hidden_states
        elif self.hidden_states is not None:  # reset hidden states of done environments
            if hidden_states is None:
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = 0.0
                else:
                    self.hidden_states[..., dones == 1, :] = 0.0
            else:
                NotImplementedError(
                    "Resetting hidden states of done environments with custom hidden states is not implemented"
                )

    def detach_hidden_states(self, dones=None):
        if self.hidden_states is not None:
            if dones is None:  # detach all hidden states
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    self.hidden_states = tuple(
                        hidden_state.detach() for hidden_state in self.hidden_states
                    )
                else:
                    self.hidden_states = self.hidden_states.detach()
            else:  # detach hidden states of done environments
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:
                    self.hidden_states[..., dones == 1, :] = self.hidden_states[
                        ..., dones == 1, :
                    ].detach()
