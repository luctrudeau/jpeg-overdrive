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

"""Tests for shared tensor utilities."""

import pytest
import torch

from jpegoverdrive.utils import (
    array_to_blocks,
    blocks_to_array,
    ste_round,
)


@pytest.mark.parametrize("height,width", [(8, 8), (16, 16), (16, 24)])
def test_array_to_blocks(height, width):
    """Verify the shape and ordering of 8x8 blocks."""
    array = torch.arange(height * width).reshape(height, width)

    blocks = array_to_blocks(array)

    assert blocks.shape == (height // 8, width // 8, 8, 8)

    for row in range(height // 8):
        for col in range(width // 8):
            expected = array[
                row * 8 : (row + 1) * 8,
                col * 8 : (col + 1) * 8,
            ]

            torch.testing.assert_close(blocks[row, col], expected)

    print(f"\nArray to blocks: {tuple(array.shape)} -> {tuple(blocks.shape)}")


@pytest.mark.parametrize("height,width", [(8, 8), (16, 16), (16, 24)])
def test_blocks_to_array(height, width):
    """Verify that block conversion is reversible."""
    array = torch.arange(height * width).reshape(height, width)

    blocks = array_to_blocks(array)
    reconstructed = blocks_to_array(blocks)

    torch.testing.assert_close(reconstructed, array)

    print(
        f"\nBlock round-trip: {tuple(array.shape)} "
        f"-> {tuple(blocks.shape)} "
        f"-> {tuple(reconstructed.shape)}"
    )


def test_ste_round_forward():
    """Verify that STE rounding matches torch.round in the forward pass."""
    values = torch.tensor([-2.5, -1.5, -0.6, 0.5, 1.5, 2.5])

    rounded = ste_round(values)
    expected = torch.round(values)

    torch.testing.assert_close(rounded, expected)

    print(f"\nSTE rounding: {values.tolist()} -> {rounded.tolist()}")


def test_ste_round_gradient():
    """Verify that STE rounding preserves unit gradients."""
    values = torch.tensor(
        [-1.7, -0.2, 0.3, 1.8],
        requires_grad=True,
    )

    rounded = ste_round(values)
    rounded.sum().backward()

    assert values.grad is not None

    torch.testing.assert_close(
        values.grad,
        torch.ones_like(values),
    )

    print(f"\nSTE gradients: {values.grad.tolist()}")
