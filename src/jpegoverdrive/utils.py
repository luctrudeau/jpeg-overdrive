# Copyright 2026 Luc Trudeau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared tensor utilities for JPEG Overdrive."""

import torch


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round values in the forward pass while preserving unit gradients."""
    return x + (torch.round(x) - x).detach()


def array_to_blocks(array: torch.Tensor) -> torch.Tensor:
    """Convert a 2D array into 8x8 blocks.

    Args:
        array: Tensor with shape (H, W), with dimensions divisible by 8.

    Returns:
        Tensor with shape (H // 8, W // 8, 8, 8).
    """
    height, width = array.shape

    return array.reshape(height // 8, 8, width // 8, 8).transpose(1, 2)


def blocks_to_array(blocks: torch.Tensor) -> torch.Tensor:
    """Reassemble 8x8 blocks into a 2D array.

    Args:
        blocks: Tensor with shape (block_rows, block_cols, 8, 8).

    Returns:
        Tensor with shape (block_rows * 8, block_cols * 8).
    """
    block_rows, block_cols = blocks.shape[:2]

    return blocks.transpose(1, 2).reshape(block_rows * 8, block_cols * 8)
