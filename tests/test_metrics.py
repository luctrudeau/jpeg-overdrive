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

"""Tests for image quality metrics and JPEG evaluation."""

import numpy as np
import pytest
import torch
from PIL import Image

from jpegoverdrive.metrics import (
    ImageReference,
    create_lpips_metric,
    load_rgb,
    measure_jpeg,
    rgb_psnr,
    rgb_to_lpips,
    rgb_to_tensor,
)


class MockLPIPS(torch.nn.Module):
    """Lightweight replacement for LPIPS in evaluation tests."""

    def forward(self, reference, reconstructed):
        """Return normalized RGB mean squared error."""
        return (
            (reference - reconstructed)
            .square()
            .mean(
                dim=(1, 2, 3),
                keepdim=True,
            )
        )


@pytest.fixture
def source_image():
    """Create a deterministic RGB test image."""
    rng = np.random.default_rng(42)

    return rng.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )


@pytest.fixture
def jpeg_path(tmp_path, source_image):
    """Create a small JPEG test image."""
    path = tmp_path / "test.jpg"

    Image.fromarray(source_image).save(
        path,
        format="JPEG",
        quality=90,
        subsampling=0,
    )

    return path


def test_load_rgb(jpeg_path):
    """Verify RGB loading and dtype."""
    rgb = load_rgb(jpeg_path)

    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.float32
    assert np.isfinite(rgb).all()
    assert rgb.min() >= 0
    assert rgb.max() <= 255

    print(
        f"\nLoaded RGB: shape={rgb.shape}, "
        f"dtype={rgb.dtype}, "
        f"range=[{rgb.min():.0f}, {rgb.max():.0f}]"
    )


def test_rgb_to_tensor():
    """Verify RGB to NCHW tensor conversion."""
    image = np.zeros((8, 16, 3), dtype=np.float32)

    image[:, :, 0] = 255.0
    image[:, :, 1] = 128.0
    image[:, :, 2] = 64.0

    tensor = rgb_to_tensor(image, device="cpu")

    assert tensor.shape == (1, 3, 8, 16)
    assert tensor.dtype == torch.float32

    assert torch.all(tensor[0, 0] == 255)
    assert torch.all(tensor[0, 1] == 128)
    assert torch.all(tensor[0, 2] == 64)

    print(f"\nRGB tensor: {image.shape} -> {tuple(tensor.shape)}")


def test_rgb_to_lpips():
    """Verify LPIPS input normalization."""
    image = np.zeros((8, 8, 3), dtype=np.float32)

    image[0, 0] = [0.0, 127.5, 255.0]

    tensor = rgb_to_lpips(image, device="cpu")

    assert tensor.shape == (1, 3, 8, 8)

    torch.testing.assert_close(
        tensor[0, :, 0, 0],
        torch.tensor([-1.0, 0.0, 1.0]),
    )

    print(
        f"\nLPIPS normalization: "
        f"RGB={image[0, 0].tolist()} -> "
        f"{tensor[0, :, 0, 0].tolist()}"
    )


def test_rgb_psnr_identical():
    """Identical images should have infinite PSNR."""
    image = np.full((8, 8, 3), 128.0, dtype=np.float32)

    psnr = rgb_psnr(image, image)

    assert psnr == float("inf")

    print("\nIdentical RGB images: PSNR=inf")


def test_rgb_psnr_single_channel_error():
    """Verify MSE is averaged across RGB channels."""
    reference = np.zeros((8, 8, 3), dtype=np.float32)
    reconstructed = reference.copy()

    reconstructed[:, :, 0] = 30.0

    # One channel has error 30; other channels have zero error.
    # MSE = 30² / 3 = 300.
    expected = 10.0 * np.log10(255.0**2 / 300.0)

    assert rgb_psnr(reference, reconstructed) == pytest.approx(expected)


def test_rgb_psnr_known_error():
    """Verify PSNR against an analytically known MSE."""
    reference = np.zeros((8, 8, 3), dtype=np.float32)
    reconstructed = np.full((8, 8, 3), 10.0, dtype=np.float32)

    # Every sample differs by 10, so MSE = 100.
    expected = 10.0 * np.log10(255.0**2 / 100.0)

    actual = rgb_psnr(reference, reconstructed)

    assert actual == pytest.approx(expected)

    print(f"\nKnown-error PSNR: expected={expected:.4f} dB, actual={actual:.4f} dB")


def test_rgb_psnr_rejects_shape_mismatch():
    """Reject images with incompatible dimensions."""
    reference = np.zeros((8, 8, 3), dtype=np.float32)
    reconstructed = np.zeros((8, 16, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="Image shapes must match"):
        rgb_psnr(reference, reconstructed)


def test_image_reference(tmp_path, source_image):
    """Verify source image preparation and caching."""
    path = tmp_path / "source.png"

    Image.fromarray(source_image).save(path)

    reference = ImageReference.from_file(path, device="cpu")

    assert reference.width == 64
    assert reference.height == 64
    assert reference.rgb.shape == (64, 64, 3)
    assert reference.lpips.shape == (1, 3, 64, 64)
    assert reference.lpips.device.type == "cpu"

    print(
        f"\nImage reference: "
        f"{reference.width}x{reference.height}, "
        f"LPIPS={tuple(reference.lpips.shape)}"
    )


def test_measure_jpeg(jpeg_path):
    """Verify JPEG rate-distortion evaluation."""
    reference = ImageReference.from_file(jpeg_path, device="cpu")

    metric = MockLPIPS()

    result = measure_jpeg(jpeg_path, reference, metric)

    expected_bytes = jpeg_path.stat().st_size
    expected_bpp = 8.0 * expected_bytes / (64 * 64)

    assert result["path"] == jpeg_path
    assert result["bytes"] == expected_bytes
    assert result["bpp"] == pytest.approx(expected_bpp)
    assert result["psnr"] == float("inf")
    assert result["lpips"] == pytest.approx(0.0)

    print(
        f"\nJPEG evaluation: "
        f"bytes={result['bytes']}, "
        f"bpp={result['bpp']:.4f}, "
        f"PSNR={result['psnr']}, "
        f"mock_lpips={result['lpips']:.6f}"
    )


def test_measure_jpeg_with_distortion(
    tmp_path,
    source_image,
    jpeg_path,
):
    """Verify evaluation against a different source image."""
    source_path = tmp_path / "source.png"

    Image.fromarray(source_image).save(source_path)

    reference = ImageReference.from_file(source_path, device="cpu")

    metric = MockLPIPS()

    result = measure_jpeg(jpeg_path, reference, metric)

    assert np.isfinite(result["psnr"])
    assert result["psnr"] > 0
    assert result["lpips"] > 0

    print(
        f"\nDistorted JPEG: "
        f"PSNR={result['psnr']:.4f} dB, "
        f"mock_lpips={result['lpips']:.6f}, "
        f"bpp={result['bpp']:.4f}"
    )


def test_measure_jpeg_dimension_mismatch(
    tmp_path,
    jpeg_path,
):
    """Reject JPEGs whose dimensions differ from the reference."""
    source = np.zeros((32, 32, 3), dtype=np.uint8)

    source_path = tmp_path / "source.png"
    Image.fromarray(source).save(source_path)

    reference = ImageReference.from_file(source_path, device="cpu")

    metric = MockLPIPS()

    with pytest.raises(ValueError, match="do not match"):
        measure_jpeg(jpeg_path, reference, metric)


def test_create_lpips_metric():
    """Verify LPIPS initialization and inference."""
    metric = create_lpips_metric("cpu", net="alex")

    assert not metric.training
    assert all(not p.requires_grad for p in metric.parameters())

    reference = torch.zeros(1, 3, 64, 64)
    reconstructed = torch.ones(1, 3, 64, 64)

    with torch.inference_mode():
        result = metric(reference, reconstructed)

    assert result.numel() == 1
    assert torch.isfinite(result).all()
