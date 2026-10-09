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

"""Tests for gradient-based JPEG optimization."""

import jpegio as jio
import numpy as np
import pytest
import torch
from PIL import Image

from jpegoverdrive.metrics import ImageReference
from jpegoverdrive.optimize import (
    DISTORTION_FLOOR,
    RATE_FLOOR,
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


class MockLPIPS(torch.nn.Module):
    """Differentiable RGB distortion metric for lightweight tests."""

    def forward(self, reference, reconstructed):
        """Return mean squared error in normalized RGB space."""
        return (
            (reference - reconstructed)
            .square()
            .mean(
                dim=(1, 2, 3),
                keepdim=True,
            )
        )


class ZeroLPIPS(torch.nn.Module):
    """Differentiable metric that always returns zero."""

    def forward(self, reference, reconstructed):
        """Return zero while retaining a connection to the input."""
        return reconstructed.square().mean() * 0.0


@pytest.fixture
def source_image():
    """Create a deterministic RGB source image."""
    rng = np.random.default_rng(42)

    return rng.integers(
        0,
        256,
        size=(32, 32, 3),
        dtype=np.uint8,
    )


@pytest.fixture
def jpeg_template(tmp_path, source_image):
    """Create a small 4:4:4 JPEG template."""
    path = tmp_path / "template.jpg"

    Image.fromarray(source_image).save(
        path,
        format="JPEG",
        quality=90,
        subsampling=0,
    )

    return path


@pytest.fixture
def initial_state(jpeg_template):
    """Extract optimizer input from a conventional JPEG."""
    jpeg = jio.read(str(jpeg_template))

    return extract_jpeg_state(jpeg, device="cpu")


@pytest.fixture
def reference(tmp_path, source_image):
    """Prepare the original uncompressed reference image."""
    path = tmp_path / "source.png"

    Image.fromarray(source_image).save(path)

    return ImageReference.from_file(path, device="cpu")


def test_extract_jpeg_state(jpeg_template):
    """Verify coefficient and quantization-table extraction."""
    jpeg = jio.read(str(jpeg_template))

    state = extract_jpeg_state(jpeg, device="cpu")

    assert len(state) == 3

    for i, component in enumerate(state):
        qindex = jpeg.comp_info[i].quant_tbl_no

        assert component["name"] == ("Y", "Cb", "Cr")[i]
        assert component["qcoeff"].shape[-2:] == (8, 8)
        assert component["qtable"].shape == (8, 8)

        assert component["qcoeff"].dtype == torch.float32
        assert component["qtable"].dtype == torch.float32

        expected_qcoeff = np.asarray(
            jpeg.coef_arrays[i],
            dtype=np.float32,
        )

        expected_qtable = np.asarray(
            jpeg.quant_tables[qindex],
            dtype=np.float32,
        )

        # Check that all JPEG coefficient values are preserved.
        actual_qcoeff = (
            component["qcoeff"]
            .permute(0, 2, 1, 3)
            .reshape(expected_qcoeff.shape)
            .numpy()
        )

        np.testing.assert_array_equal(
            actual_qcoeff,
            expected_qcoeff,
        )

        np.testing.assert_array_equal(
            component["qtable"].numpy(),
            expected_qtable,
        )

    print("\nJPEG state extraction: coefficients and tables preserved")


def test_trainable_state_reproducible(jpeg_template):
    """Verify deterministic initialization from the same seed."""
    jpeg = jio.read(str(jpeg_template))

    state_a = make_trainable_state(jpeg, device="cpu", seed=42)
    state_b = make_trainable_state(jpeg, device="cpu", seed=42)

    for a, b in zip(state_a, state_b):
        torch.testing.assert_close(
            a["coeff"],
            b["coeff"],
        )

        torch.testing.assert_close(
            a["qtable"],
            b["qtable"],
        )

    print("\nTrainable initialization: reproducible with seed=42")


def test_trainable_state_preserves_quantized_values(jpeg_template):
    """Verify dithering preserves the initial quantized JPEG state."""
    from jpegoverdrive.jpeg import quantize_component

    jpeg = jio.read(str(jpeg_template))

    state = make_trainable_state(jpeg, device="cpu", seed=42)

    for component in state:
        qcoeff, qtable = quantize_component(
            component["coeff"],
            component["qtable"],
        )

        torch.testing.assert_close(
            qcoeff,
            component["qcoeff_reference"],
        )

        torch.testing.assert_close(
            qtable,
            component["qtable_reference"],
        )

    print("\nInitialization: quantized coefficients and tables preserved")


def test_shared_quantization_tables(jpeg_template):
    """Verify chroma components share their quantization parameter."""
    jpeg = jio.read(str(jpeg_template))

    state = make_trainable_state(jpeg, device="cpu")

    assert jpeg.comp_info[1].quant_tbl_no == jpeg.comp_info[2].quant_tbl_no

    assert state[1]["qtable"] is state[2]["qtable"]

    parameters = trainable_parameters(state)

    # Three coefficient tensors and two distinct quantization tables.
    assert len(parameters) == 5

    print(
        f"\nShared tables: "
        f"{len(state)} components, "
        f"{len(parameters)} unique parameter tensors"
    )


def test_shared_quantization_table_conflict(
    jpeg_template,
    initial_state,
):
    """Reject conflicting values for a shared JPEG quantization table."""
    jpeg = jio.read(str(jpeg_template))

    modified = [
        {
            **component,
            "qtable": component["qtable"].clone(),
        }
        for component in initial_state
    ]

    # Cb and Cr share a table in our 4:4:4 test JPEG.
    modified[2]["qtable"][0, 0] += 1

    with pytest.raises(
        ValueError,
        match="Conflicting initial quantization table",
    ):
        make_trainable_state_from_state(
            modified,
            jpeg,
            device="cpu",
        )


def test_rate_proxy_zero():
    """Zero-valued coefficients should have zero proxy rate."""
    qcoeff = torch.zeros((2, 2, 8, 8))

    rate = rate_proxy(qcoeff)

    assert rate.item() == pytest.approx(0.0)

    print("\nZero coefficients: rate proxy=0")


def test_rate_proxy_monotonic():
    """Larger coefficient magnitudes should increase proxy rate."""
    small = torch.ones((1, 1, 8, 8))
    large = torch.full((1, 1, 8, 8), 10.0)

    small_rate = rate_proxy(small)
    large_rate = rate_proxy(large)

    assert large_rate > small_rate

    print(
        f"\nRate proxy: "
        f"|coeff|=1 -> {small_rate.item():.3f}, "
        f"|coeff|=10 -> {large_rate.item():.3f}"
    )


def test_rate_proxy_gradient():
    """Verify that the rate proxy propagates gradients."""
    qcoeff = torch.tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    rate = rate_proxy(qcoeff)
    rate.backward()

    assert qcoeff.grad is not None
    assert torch.isfinite(qcoeff.grad).all()
    assert torch.count_nonzero(qcoeff.grad) > 0

    print(f"\nRate gradients: {qcoeff.grad.tolist()}")


def test_decode_jpeg_state(jpeg_template):
    """Verify trainable-state reconstruction has the expected shape."""
    jpeg = jio.read(str(jpeg_template))

    state = make_trainable_state(jpeg, device="cpu")

    reconstructed = decode_jpeg_state(
        state,
        jpeg.image_height,
        jpeg.image_width,
    )

    assert reconstructed.shape == (32, 32, 3)
    assert torch.isfinite(reconstructed).all()
    assert reconstructed.min() >= 0
    assert reconstructed.max() <= 255

    print(
        f"\nState reconstruction: "
        f"shape={tuple(reconstructed.shape)}, "
        f"range=[{reconstructed.min().item():.0f}, "
        f"{reconstructed.max().item():.0f}]"
    )


def test_tensor_rgb_to_lpips():
    """Verify LPIPS input normalization."""
    rgb = torch.tensor(
        [[[0.0, 127.5, 255.0]]],
        requires_grad=True,
    )

    normalized = tensor_rgb_to_lpips(rgb)

    assert normalized.shape == (1, 3, 1, 1)

    torch.testing.assert_close(
        normalized[0, :, 0, 0],
        torch.tensor([-1.0, 0.0, 1.0]),
    )

    normalized.sum().backward()

    assert rgb.grad is not None
    assert torch.isfinite(rgb.grad).all()

    print(
        f"\nRGB normalization: "
        f"{rgb.detach().flatten().tolist()} -> "
        f"{normalized.detach().flatten().tolist()}"
    )


def test_objective_gradients(jpeg_template, reference):
    """Verify gradients propagate through the complete RD objective."""
    jpeg = jio.read(str(jpeg_template))

    state = make_trainable_state(jpeg, device="cpu")

    metric = MockLPIPS()

    with torch.no_grad():
        initial_rgb = decode_jpeg_state(
            state,
            jpeg.image_height,
            jpeg.image_width,
        )

        initial_distortion = (
            metric(
                reference.lpips,
                tensor_rgb_to_lpips(initial_rgb),
            )
            .mean()
            .clamp_min(DISTORTION_FLOOR)
        )

        initial_rate = jpeg_rate_proxy(state).clamp_min(RATE_FLOOR)

    d_norm, r_norm = compute_objectives(
        state,
        reference,
        metric,
        initial_distortion,
        initial_rate,
        jpeg.image_height,
        jpeg.image_width,
    )

    loss = d_norm + 0.001 * r_norm
    loss.backward()

    parameters = trainable_parameters(state)

    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    assert any(torch.count_nonzero(parameter.grad) > 0 for parameter in parameters)

    print(
        f"\nRD objective: "
        f"D/scale={d_norm.item():.6f}, "
        f"R/scale={r_norm.item():.6f}, "
        f"loss={loss.item():.6f}"
    )


def test_select_best_candidate():
    """Select lowest LPIPS without exceeding the baseline size."""
    baseline = {
        "path": "baseline.jpg",
        "bytes": 1000,
        "lpips": 0.020,
    }

    history = [
        {
            "step": 0,
            "path": "step_000000.jpg",
            "bytes": 1000,
            "lpips": 0.019,
        },
        {
            "step": 10,
            "path": "step_000010.jpg",
            "bytes": 900,
            "lpips": 0.018,
        },
        {
            "step": 20,
            "path": "step_000020.jpg",
            "bytes": 1100,
            "lpips": 0.010,
        },
        {
            "step": 30,
            "path": "step_000030.jpg",
            "bytes": 800,
            "lpips": 0.021,
        },
    ]

    best = select_best_candidate(history, baseline)

    assert best["step"] == 10
    assert best["is_baseline"] is False
    assert best["bytes"] == 900
    assert best["lpips"] == pytest.approx(0.018)

    print("\nCandidate selection: step 10")


def test_select_best_candidate_baseline_fallback():
    """Keep the baseline if no candidate offers an eligible improvement."""
    baseline = {
        "path": "baseline.jpg",
        "bytes": 1000,
        "lpips": 0.020,
    }

    history = [
        {
            "step": 0,
            "bytes": 1050,
            "lpips": 0.019,
        },
        {
            "step": 10,
            "bytes": 900,
            "lpips": 0.025,
        },
        {
            "step": 20,
            "bytes": 1100,
            "lpips": 0.018,
        },
    ]

    best = select_best_candidate(history, baseline)

    assert best["is_baseline"] is True
    assert best["step"] is None
    assert best["bytes"] == baseline["bytes"]

    print("\nCandidate selection: baseline retained")


def test_select_best_candidate_zero_distortion():
    """Keep a perfect baseline if candidates introduce distortion."""
    baseline = {
        "path": "baseline.jpg",
        "bytes": 1000,
        "lpips": 0.0,
    }

    history = [
        {
            "step": 10,
            "bytes": 900,
            "lpips": 0.001,
        },
        {
            "step": 20,
            "bytes": 950,
            "lpips": 0.002,
        },
    ]

    best = select_best_candidate(history, baseline)

    assert best["is_baseline"] is True
    assert best["step"] is None

    print("\nZero-distortion baseline: retained")


def test_select_best_candidate_equal_quality():
    """Prefer the smaller JPEG when LPIPS is identical."""
    baseline = {
        "path": "baseline.jpg",
        "bytes": 1000,
        "lpips": 0.020,
    }

    history = [
        {
            "step": 10,
            "bytes": 900,
            "lpips": 0.020,
        },
    ]

    best = select_best_candidate(history, baseline)

    assert best["step"] == 10
    assert best["bytes"] == 900

    print("\nEqual quality: smaller JPEG selected")


def test_select_best_candidate_nonfinite():
    """Ignore snapshots with invalid LPIPS measurements."""
    baseline = {
        "path": "baseline.jpg",
        "bytes": 1000,
        "lpips": 0.020,
    }

    history = [
        {
            "step": 10,
            "bytes": 800,
            "lpips": float("nan"),
        },
        {
            "step": 20,
            "bytes": 900,
            "lpips": float("inf"),
        },
        {
            "step": 30,
            "bytes": 950,
            "lpips": 0.019,
        },
    ]

    best = select_best_candidate(history, baseline)

    assert best["step"] == 30
    assert best["lpips"] == pytest.approx(0.019)

    print("\nCandidate selection: nonfinite measurements ignored")


def test_optimize_jpeg(
    jpeg_template,
    initial_state,
    reference,
    tmp_path,
):
    """Run a short optimization and validate exported snapshots."""
    output_dir = tmp_path / "optimization"

    state, history = optimize_jpeg(
        initial_state=initial_state,
        template_path=jpeg_template,
        output_dir=output_dir,
        reference=reference,
        metric=MockLPIPS(),
        lambda_rd=0.001,
        steps=2,
        lr=0.001,
        snapshot_every=1,
        seed=42,
        verbose=True,
    )

    assert len(state) == 3
    assert len(history) == 3
    assert [row["step"] for row in history] == [0, 1, 2]

    for row in history:
        assert row["path"].exists()
        assert row["bytes"] > 0
        assert row["bpp"] > 0
        assert np.isfinite(row["loss"])
        assert np.isfinite(row["lpips"])
        assert np.isfinite(row["distortion_norm"])
        assert np.isfinite(row["rate_norm"])

        with Image.open(row["path"]) as image:
            image.load()
            assert image.size == (32, 32)

    print(f"\nOptimization integration: {len(history)} valid JPEG snapshots")


def test_optimize_final_snapshot(
    jpeg_template,
    initial_state,
    reference,
    tmp_path,
):
    """Always export the final step, even off the snapshot interval."""
    _, history = optimize_jpeg(
        initial_state=initial_state,
        template_path=jpeg_template,
        output_dir=tmp_path / "snapshots",
        reference=reference,
        metric=MockLPIPS(),
        lambda_rd=0.001,
        steps=3,
        lr=0.001,
        snapshot_every=2,
        seed=42,
        verbose=False,
    )

    assert [row["step"] for row in history] == [0, 2, 3]

    for row in history:
        assert row["path"].exists()

    print("\nFinal snapshot: step 3 exported")


def test_optimize_zero_distortion(
    jpeg_template,
    initial_state,
    reference,
    tmp_path,
):
    """Verify finite optimization with an exactly zero distortion."""
    _, history = optimize_jpeg(
        initial_state=initial_state,
        template_path=jpeg_template,
        output_dir=tmp_path / "zero",
        reference=reference,
        metric=ZeroLPIPS(),
        lambda_rd=0.001,
        steps=2,
        lr=0.001,
        snapshot_every=1,
        seed=42,
        verbose=False,
    )

    assert len(history) == 3

    for row in history:
        assert np.isfinite(row["loss"])
        assert np.isfinite(row["rate_norm"])
        assert row["distortion_norm"] == pytest.approx(0.0)
        assert row["lpips"] == pytest.approx(0.0)

    print("\nZero-distortion optimization: finite objectives")


def test_optimize_near_zero_distortion(
    jpeg_template,
    initial_state,
    reference,
    tmp_path,
    capsys,
):
    """Verify the distortion normalization floor is used."""
    optimize_jpeg(
        initial_state=initial_state,
        template_path=jpeg_template,
        output_dir=tmp_path / "near_zero",
        reference=reference,
        metric=ZeroLPIPS(),
        lambda_rd=0.001,
        steps=0,
        snapshot_every=1,
        verbose=True,
    )

    output = capsys.readouterr().out

    assert f"Distortion scale: {DISTORTION_FLOOR:g}" in output

    print(f"\nDistortion normalization floor: {DISTORTION_FLOOR:g}")


@pytest.mark.parametrize(
    "steps,lr,snapshot_every,lambda_rd",
    [
        (-1, 0.001, 1, 0.001),
        (1, 0.0, 1, 0.001),
        (1, float("nan"), 1, 0.001),
        (1, 0.001, 0, 0.001),
        (1, 0.001, 1, -1.0),
        (1, 0.001, 1, float("inf")),
    ],
)
def test_optimize_rejects_invalid_config(
    jpeg_template,
    initial_state,
    reference,
    tmp_path,
    steps,
    lr,
    snapshot_every,
    lambda_rd,
):
    """Reject invalid optimizer configuration."""
    with pytest.raises(ValueError):
        optimize_jpeg(
            initial_state=initial_state,
            template_path=jpeg_template,
            output_dir=tmp_path / "invalid",
            reference=reference,
            metric=MockLPIPS(),
            lambda_rd=lambda_rd,
            steps=steps,
            lr=lr,
            snapshot_every=snapshot_every,
            verbose=False,
        )
