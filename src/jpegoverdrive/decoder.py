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

"""Differentiable JPEG decoder using libjpeg's ISLOW inverse DCT."""

import jpegio as jio
import torch

from jpegoverdrive.dct import (
    ISLOW_CONST_BITS,
    ISLOW_PASS1_BITS,
    idct_islow_1d,
)


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round values in the forward pass while preserving unit gradients."""
    return x + (torch.round(x) - x).detach()


def coef_array_to_blocks(coef: torch.Tensor) -> torch.Tensor:
    """Convert jpegio's coefficient array to 8x8 DCT blocks.

    Args:
        coef: Coefficient array with shape (H, W), where both
            dimensions are multiples of 8.

    Returns:
        Tensor with shape (H // 8, W // 8, 8, 8).
    """
    height, width = coef.shape

    return coef.reshape(height // 8, 8, width // 8, 8).permute(0, 2, 1, 3)


def blocks_to_image(blocks: torch.Tensor) -> torch.Tensor:
    """Reassemble 8x8 blocks into a two-dimensional image.

    Args:
        blocks: Tensor with shape (block_rows, block_cols, 8, 8).

    Returns:
        Tensor with shape (block_rows * 8, block_cols * 8).
    """
    block_rows, block_cols = blocks.shape[:2]

    return blocks.permute(0, 2, 1, 3).reshape(block_rows * 8, block_cols * 8)


def decode_component(
    qcoeff: torch.Tensor,
    qtable: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a JPEG component from quantized DCT coefficients.

    Performs inverse quantization, a two-pass ISLOW inverse DCT,
    JPEG's +128 level shift, and clipping to [0, 255].

    Args:
        qcoeff: Quantized DCT coefficients with shape (..., 8, 8).
        qtable: Quantization table with shape (8, 8).

    Returns:
        Reconstructed component plane with shape (H, W).
    """
    # Inverse quantization.
    coeff = qcoeff * qtable

    # First IDCT pass (columns).
    workspace = idct_islow_1d(
        coeff.transpose(-1, -2),
        ISLOW_CONST_BITS - ISLOW_PASS1_BITS,
    ).transpose(-1, -2)

    # Second IDCT pass (rows).
    blocks = idct_islow_1d(
        workspace,
        ISLOW_CONST_BITS + ISLOW_PASS1_BITS + 3,
    )

    # Reassemble blocks and apply the JPEG level shift.
    return torch.clamp(blocks_to_image(blocks) + 128.0, 0.0, 255.0)


def ycbcr_to_rgb(
    y: torch.Tensor,
    cb: torch.Tensor,
    cr: torch.Tensor,
) -> torch.Tensor:
    """Convert JPEG YCbCr component planes to RGB.

    Args:
        y: Luma component in [0, 255].
        cb: Blue-difference chroma component in [0, 255].
        cr: Red-difference chroma component in [0, 255].

    Returns:
        RGB tensor with shape (H, W, 3) and values in [0, 255].
    """
    cb = cb - 128.0
    cr = cr - 128.0

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    return torch.stack([r, g, b], dim=-1).clamp(0.0, 255.0)


def decode_jpeg_components(
    jpeg: jio.DecompressedJpeg,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, ...]:
    """Decode JPEG DCT coefficients into reconstructed component planes.

    Currently supports three-component JPEG images with 4:4:4 sampling.

    Args:
        jpeg: JPEG representation returned by jpegio.read().
        device: PyTorch device used for reconstruction.

    Returns:
        Tuple containing the reconstructed Y, Cb, and Cr planes.

    Raises:
        ValueError: If the JPEG uses an unsupported component count
            or chroma subsampling configuration.
    """
    if len(jpeg.comp_info) != 3:
        raise ValueError("Only three-component JPEG images are supported.")

    if any(
        component.h_samp_factor != 1 or component.v_samp_factor != 1
        for component in jpeg.comp_info
    ):
        raise ValueError("Only 4:4:4 JPEG images are currently supported.")

    components = []

    for i, component in enumerate(jpeg.comp_info):
        # TODO: Investigate occasional ±1 differences in reconstructed components
        # compared with libjpeg-turbo's ISLOW IDCT. These may result from float32
        # precision differences relative to libjpeg's integer arithmetic.
        coef_array = torch.as_tensor(
            jpeg.coef_arrays[i],
            dtype=torch.float32,
            device=device,
        )

        qcoeff = coef_array_to_blocks(coef_array)

        qtable = torch.as_tensor(
            jpeg.quant_tables[component.quant_tbl_no],
            dtype=torch.float32,
            device=device,
        )

        plane = decode_component(qcoeff, qtable)
        plane = plane[: jpeg.image_height, : jpeg.image_width]

        components.append(plane)

    return tuple(components)


def decode_jpeg_rgb(
    jpeg: jio.DecompressedJpeg,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Decode a JPEG image into a differentiable RGB tensor.

    Args:
        jpeg: JPEG representation returned by jpegio.read().
        device: PyTorch device used for reconstruction.

    Returns:
        RGB tensor with shape (H, W, 3), containing rounded
        pixel values in [0, 255].
    """
    y, cb, cr = decode_jpeg_components(jpeg, device)

    rgb = ycbcr_to_rgb(y, cb, cr)

    return ste_round(rgb).clamp(0.0, 255.0)
