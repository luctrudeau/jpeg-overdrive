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

"""Tests for the differentiable JPEG decoder."""

from pathlib import Path

import jpegio as jio
import numpy as np
import pytest
import torch
from PIL import Image

from jpegoverdrive.decoder import (
    blocks_to_image,
    coef_array_to_blocks,
    decode_component,
    decode_jpeg_rgb,
)

LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "logo.jpg"


@pytest.fixture
def source_image():
    """Create a deterministic RGB test image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


def validate_decoder(path):
    """Compare the differentiable decoder against Pillow."""
    jpeg = jio.read(str(path))
    reconstructed = decode_jpeg_rgb(jpeg, device="cpu")

    with Image.open(path) as image:
        reference = np.asarray(image.convert("RGB"))

    actual = reconstructed.detach().cpu().numpy()
    error = np.abs(actual - reference.astype(np.float32))

    mae = float(error.mean())
    max_error = float(error.max())
    different_samples = int(np.count_nonzero(error))
    matching = 100.0 * (1.0 - different_samples / error.size)

    print(f"MAE={mae:.6f}, max_error={max_error:.0f}, matching={matching:.4f}%")

    # Show significant differences for debugging.
    positions = np.argwhere(error > 1)

    for y, x, channel in positions[:10]:
        print(
            f"  Pixel ({x}, {y}), channel {channel}: "
            f"actual={actual[y, x, channel]:.0f}, "
            f"reference={reference[y, x, channel]}, "
            f"error={error[y, x, channel]:.0f}"
        )

    return mae, max_error, matching


@pytest.mark.parametrize("quality", [75, 90, 99])
@pytest.mark.parametrize("image_type", ["random", "logo"])
def test_decoder_matches_reference(tmp_path, source_image, quality, image_type):
    """Validate JPEG reconstruction against a conventional decoder."""
    if image_type == "random":
        source = source_image
    else:
        with Image.open(LOGO_PATH) as image:
            source = np.asarray(image.convert("RGB"))

    jpeg_path = tmp_path / f"{image_type}_q{quality}.jpg"

    Image.fromarray(source).save(
        jpeg_path,
        format="JPEG",
        quality=quality,
        subsampling=0,  # 4:4:4
    )

    print(f"\n{image_type} Q{quality} ({source.shape[1]}x{source.shape[0]}):")
    mae, max_error, matching = validate_decoder(jpeg_path)

    assert mae < 0.01
    assert max_error <= 1.0


def test_coef_array_to_blocks():
    """Verify coefficient block ordering."""
    coefficients = torch.arange(256).reshape(16, 16)

    blocks = coef_array_to_blocks(coefficients)

    print(f"\nCoefficient layout: {tuple(coefficients.shape)}")
    print(f"Block layout:       {tuple(blocks.shape)}")

    assert blocks.shape == (2, 2, 8, 8)

    torch.testing.assert_close(blocks[0, 0], coefficients[:8, :8])
    torch.testing.assert_close(blocks[0, 1], coefficients[:8, 8:])
    torch.testing.assert_close(blocks[1, 0], coefficients[8:, :8])
    torch.testing.assert_close(blocks[1, 1], coefficients[8:, 8:])


def test_blocks_to_image():
    """Verify that block conversion is reversible."""
    image = torch.arange(256).reshape(16, 16)

    blocks = coef_array_to_blocks(image)
    reconstructed = blocks_to_image(blocks)

    print(
        f"\nBlock round-trip: "
        f"{tuple(image.shape)} -> {tuple(blocks.shape)} "
        f"-> {tuple(reconstructed.shape)}"
    )

    torch.testing.assert_close(reconstructed, image)


def test_decode_component_dc_only():
    """A DC-only block should reconstruct to a constant value."""
    qcoeff = torch.zeros((1, 1, 8, 8))
    qcoeff[0, 0, 0, 0] = 80.0

    qtable = torch.ones((8, 8))

    reconstructed = decode_component(qcoeff, qtable)

    # A JPEG DC coefficient of 80 contributes 80 / 8 = 10.
    expected = torch.full((8, 8), 138.0)

    print(
        f"\nDC-only reconstruction: "
        f"expected={expected[0, 0].item():.0f}, "
        f"actual={reconstructed[0, 0].item():.0f}"
    )

    torch.testing.assert_close(reconstructed, expected)


def test_decoder_coefficient_gradient():
    """Verify gradients propagate to quantized DCT coefficients."""
    torch.manual_seed(42)

    qcoeff = torch.randn(
        (2, 2, 8, 8),
        requires_grad=True,
    )
    qtable = torch.ones((8, 8))

    reconstructed = decode_component(qcoeff, qtable)

    loss = reconstructed.square().mean()
    loss.backward()

    assert qcoeff.grad is not None
    assert torch.isfinite(qcoeff.grad).all()
    assert torch.count_nonzero(qcoeff.grad) > 0

    grad = qcoeff.grad

    print(
        f"\nCoefficient gradients: "
        f"mean_abs={grad.abs().mean().item():.6e}, "
        f"max_abs={grad.abs().max().item():.6e}, "
        f"nonzero={torch.count_nonzero(grad).item()}/{grad.numel()}"
    )


def test_decoder_qtable_gradient():
    """Verify gradients propagate to quantization tables."""
    torch.manual_seed(42)

    qcoeff = torch.randn((2, 2, 8, 8))

    qtable = torch.ones(
        (8, 8),
        requires_grad=True,
    )

    reconstructed = decode_component(qcoeff, qtable)

    loss = reconstructed.square().mean()
    loss.backward()

    assert qtable.grad is not None
    assert torch.isfinite(qtable.grad).all()
    assert torch.count_nonzero(qtable.grad) > 0

    grad = qtable.grad

    print(
        f"\nQuantization table gradients: "
        f"mean_abs={grad.abs().mean().item():.6e}, "
        f"max_abs={grad.abs().max().item():.6e}, "
        f"nonzero={torch.count_nonzero(grad).item()}/{grad.numel()}"
    )


def test_decoder_rejects_subsampling(tmp_path, source_image):
    """Reject 4:2:0 JPEGs until chroma upsampling is supported."""
    path = tmp_path / "subsampled.jpg"

    Image.fromarray(source_image).save(
        path,
        format="JPEG",
        quality=90,
        subsampling=2,  # 4:2:0
    )

    jpeg = jio.read(str(path))

    sampling = [(comp.h_samp_factor, comp.v_samp_factor) for comp in jpeg.comp_info]

    print(f"\nUnsupported sampling factors: {sampling}")

    with pytest.raises(ValueError, match="4:4:4"):
        decode_jpeg_rgb(jpeg)
