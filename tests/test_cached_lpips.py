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

"""Tests for LPIPS reference-feature caching."""

import warnings

import lpips
import pytest
import torch

from jpegoverdrive.metrics import (
    CachedLPIPS,
    create_lpips_metric,
    maybe_cache_lpips,
)


@pytest.mark.parametrize("net", ["alex", "vgg"])
def test_cached_lpips_equivalence(net: str) -> None:
    """Verify cached LPIPS preserves distortion values and image gradients."""
    torch.manual_seed(42)

    metric = create_lpips_metric("cpu", net=net)

    reference = torch.rand(1, 3, 64, 64) * 2 - 1
    reconstructed = torch.rand(1, 3, 64, 64) * 2 - 1

    cached = CachedLPIPS(metric, reference)

    original_image = reconstructed.clone().requires_grad_(True)
    cached_image = reconstructed.clone().requires_grad_(True)

    original_loss = metric(reference, original_image).mean()
    cached_loss = cached(reference, cached_image).mean()

    original_grad = torch.autograd.grad(original_loss, original_image)[0]
    cached_grad = torch.autograd.grad(cached_loss, cached_image)[0]

    torch.testing.assert_close(
        cached_loss,
        original_loss,
        rtol=1e-5,
        atol=1e-7,
    )

    torch.testing.assert_close(
        cached_grad,
        original_grad,
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lpips": False},
        {"version": "0.0"},
        {"spatial": True},
    ],
)
def test_unsupported_lpips_configuration(kwargs: dict) -> None:
    """Verify unsupported LPIPS configurations bypass feature caching."""
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

        metric = lpips.LPIPS(net="alex", verbose=False, **kwargs).eval()

    reference = torch.rand(1, 3, 64, 64) * 2 - 1

    result = maybe_cache_lpips(metric, reference)

    assert result is metric


@pytest.mark.parametrize("net", ["alex", "vgg"])
def test_supported_lpips_configuration(net: str) -> None:
    """Verify standard LPIPS networks use cached reference features."""
    metric = create_lpips_metric("cpu", net=net)
    reference = torch.rand(1, 3, 64, 64) * 2 - 1

    result = maybe_cache_lpips(metric, reference)

    assert isinstance(result, CachedLPIPS)


def test_lpips_caching_disabled() -> None:
    """Verify disabling caching preserves the original LPIPS evaluator."""
    metric = create_lpips_metric("cpu")
    reference = torch.rand(1, 3, 64, 64) * 2 - 1

    result = maybe_cache_lpips(metric, reference, enabled=False)

    assert result is metric


def test_lpips_subclass_not_cached() -> None:
    """Verify custom LPIPS forward implementations bypass caching."""

    class CustomLPIPS(lpips.LPIPS):
        """LPIPS variant with a modified distortion function."""

        def forward(self, in0, in1, *args, **kwargs):
            return 2.0 * super().forward(in0, in1, *args, **kwargs)

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
        metric = CustomLPIPS(net="alex", verbose=False).eval()
    reference = torch.rand(1, 3, 64, 64) * 2 - 1

    result = maybe_cache_lpips(metric, reference)

    assert result is metric


def test_cached_lpips_buffers() -> None:
    """Verify cached features are registered as non-persistent buffers."""
    metric = create_lpips_metric("cpu", net="alex")
    reference = torch.rand(1, 3, 64, 64) * 2 - 1

    cached = CachedLPIPS(metric, reference)

    buffers = {
        name: tensor
        for name, tensor in cached.named_buffers()
        if name.startswith("reference_feature_")
    }

    assert len(buffers) == len(metric.lins)

    for name, tensor in buffers.items():
        assert tensor.device == reference.device
        assert name not in cached.state_dict()
