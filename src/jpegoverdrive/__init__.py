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

"""JPEG Overdrive: JPEG-compliant learned perceptual compression."""

__version__ = "0.1.0"

from .jpeg import quantize_component, write_jpeg_state
from .metrics import (
    ImageReference,
    create_lpips_metric,
    load_rgb,
    measure_jpeg,
    rgb_psnr,
    rgb_to_lpips,
    rgb_to_tensor,
)
from .optimize import (
    compute_objectives,
    decode_jpeg_state,
    extract_jpeg_state,
    jpeg_rate_proxy,
    make_trainable_state,
    make_trainable_state_from_state,
    optimize_jpeg,
    rate_proxy,
    select_best_candidate,
    tensor_rgb_to_lpips,
    trainable_parameters,
)

__all__ = [
    "__version__",
    "ImageReference",
    "compute_objectives",
    "create_lpips_metric",
    "decode_jpeg_state",
    "extract_jpeg_state",
    "jpeg_rate_proxy",
    "load_rgb",
    "make_trainable_state",
    "make_trainable_state_from_state",
    "measure_jpeg",
    "optimize_jpeg",
    "quantize_component",
    "rate_proxy",
    "rgb_psnr",
    "rgb_to_lpips",
    "rgb_to_tensor",
    "select_best_candidate",
    "tensor_rgb_to_lpips",
    "trainable_parameters",
    "write_jpeg_state",
]
