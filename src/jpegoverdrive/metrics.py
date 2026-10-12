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

"""Image quality metrics and JPEG rate-distortion evaluation."""

import warnings
from dataclasses import dataclass
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image


def create_lpips_metric(
    device: torch.device | str,
    net: str = "alex",
) -> lpips.LPIPS:
    """Create a frozen LPIPS metric.

    Args:
        device: Device on which to evaluate LPIPS.
        net: LPIPS backbone network, such as "alex" or "vgg".

    Returns:
        LPIPS metric in evaluation mode.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The parameter 'pretrained' is deprecated.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Arguments other than a weight enum or `None`.*",
            category=UserWarning,
        )

        metric = lpips.LPIPS(net=net).to(device)

    metric.eval()

    for parameter in metric.parameters():
        parameter.requires_grad_(False)

    return metric


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as an HxWx3 float32 RGB array in [0, 255]."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def rgb_to_tensor(
    image: np.ndarray,
    device: torch.device | str,
) -> torch.Tensor:
    """Convert an HxWx3 RGB array to an NCHW float32 tensor."""
    tensor = torch.as_tensor(
        image,
        dtype=torch.float32,
        device=device,
    )

    return tensor.permute(2, 0, 1).unsqueeze(0)


def rgb_to_lpips(
    image: np.ndarray,
    device: torch.device | str,
) -> torch.Tensor:
    """Convert RGB [0, 255] to LPIPS NCHW [-1, 1]."""
    tensor = rgb_to_tensor(image, device)

    return 2.0 * tensor / 255.0 - 1.0


def rgb_psnr(
    reference: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """Compute PSNR over RGB samples with a peak value of 255.

    Args:
        reference: Reference RGB image in [0, 255].
        reconstructed: Reconstructed RGB image in [0, 255].

    Returns:
        RGB PSNR in dB, or infinity for identical images.

    Raises:
        ValueError: If the images have different shapes.
    """
    if reference.shape != reconstructed.shape:
        raise ValueError(
            f"Image shapes must match: {reference.shape} != {reconstructed.shape}."
        )

    # Use float64 to avoid overflow and improve numerical precision.
    error = reference.astype(np.float64) - reconstructed.astype(np.float64)

    mse = np.mean(error**2)

    if mse == 0:
        return float("inf")

    return float(10.0 * np.log10(255.0**2 / mse))


@dataclass(eq=False)
class ImageReference:
    """Cache an original RGB image and its normalized LPIPS input.

    Attributes:
        rgb: HxWx3 float32 array with values in [0, 255].
        lpips: NCHW float32 tensor normalized to [-1, 1] on the
            requested compute device.

    The LPIPS input tensor is cached, but backbone feature maps
    are not. Feature caching is planned as a separate optimization.
    """

    rgb: np.ndarray
    lpips: torch.Tensor

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        device: torch.device | str,
    ) -> "ImageReference":
        """Load an image and prepare its RGB and LPIPS representations."""
        rgb = load_rgb(path)
        lpips_tensor = rgb_to_lpips(rgb, device)

        return cls(rgb=rgb, lpips=lpips_tensor)

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.rgb.shape[1]


def measure_jpeg(
    path: str | Path,
    reference: ImageReference,
    metric: torch.nn.Module | dict[str, torch.nn.Module],
) -> dict:
    """Measure actual JPEG bitrate and perceptual distortion.

    Decode the JPEG with Pillow and evaluate PSNR and LPIPS against
    the original reference image.

    A single LPIPS model stores its score in the historical ``lpips``
    field, regardless of the selected backbone. Joint evaluation
    stores Alex in ``lpips`` and VGG in ``lpips_vgg``.

    Args:
        path: Path to the encoded JPEG.
        reference: Prepared reference image.
        metric: Single LPIPS model or dictionary containing
            ``alex`` and ``vgg`` models.

    Returns:
        Dictionary containing the JPEG path, size in bytes, bitrate
        in bits per pixel, RGB PSNR, and LPIPS measurements.

    Raises:
        ValueError: If the JPEG dimensions differ from the reference.
    """

    if isinstance(metric, dict) and set(metric) != {"alex", "vgg"}:
        raise ValueError("Joint evaluation requires 'alex' and 'vgg' metrics.")

    path = Path(path)

    decoded_rgb = load_rgb(path)

    if decoded_rgb.shape != reference.rgb.shape:
        raise ValueError(
            f"JPEG dimensions {decoded_rgb.shape} do not match "
            f"reference dimensions {reference.rgb.shape}."
        )

    decoded_lpips = rgb_to_lpips(
        decoded_rgb,
        reference.lpips.device,
    )

    with torch.inference_mode():
        if isinstance(metric, dict):
            lpips_values = {
                name: model(reference.lpips, decoded_lpips).mean().item()
                for name, model in metric.items()
            }
        else:
            lpips_values = {
                "alex": metric(reference.lpips, decoded_lpips).mean().item()
            }

    size_bytes = path.stat().st_size
    n_pixels = reference.width * reference.height

    result = {
        "path": path,
        "bytes": size_bytes,
        "bpp": 8.0 * size_bytes / n_pixels,
        "psnr": rgb_psnr(reference.rgb, decoded_rgb),
        "lpips": lpips_values["alex"],
    }

    if "vgg" in lpips_values:
        result["lpips_vgg"] = lpips_values["vgg"]

    return result
