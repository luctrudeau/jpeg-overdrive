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

"""Tests for JPEG bitstream export."""

from pathlib import Path

import jpegio as jio
import numpy as np
import pytest
import torch
from PIL import Image

from jpegoverdrive.jpeg import quantize_component, write_jpeg_state
from jpegoverdrive.utils import array_to_blocks

LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "logo.jpg"


@pytest.fixture
def jpeg_template(tmp_path, request):
    """Create a JPEG using the project logo."""
    subsampling = getattr(request, "param", 0)

    with Image.open(LOGO_PATH) as image:
        source = image.convert("RGB").resize((128, 128))

        path = tmp_path / "template.jpg"
        source.save(path, quality=90, subsampling=subsampling)

    return path


def make_state(jpeg):
    """Construct a continuous optimizer state from a JPEG."""
    state = []

    for i, component in enumerate(jpeg.comp_info):
        qcoeff = array_to_blocks(
            torch.as_tensor(
                jpeg.coef_arrays[i],
                dtype=torch.float32,
            )
        )

        qtable = torch.as_tensor(
            jpeg.quant_tables[component.quant_tbl_no],
            dtype=torch.float32,
        ).clone()

        state.append(
            {
                "coeff": qcoeff * qtable,
                "qtable": qtable.clone(),
            }
        )

    return state


def test_quantize_component():
    """Verify coefficient and quantization-table rounding."""
    coeff = torch.zeros((1, 1, 8, 8))
    coeff[0, 0, 0, 0] = 12.0
    coeff[0, 0, 0, 1] = -13.0

    qtable = torch.full((8, 8), 5.0)

    qcoeff, quantized_table = quantize_component(coeff, qtable)

    assert qcoeff[0, 0, 0, 0].item() == 2
    assert qcoeff[0, 0, 0, 1].item() == -3
    assert torch.all(quantized_table == 5)

    print(
        f"\nQuantization: "
        f"coefficients={coeff[0, 0, 0, :2].tolist()}, "
        f"quantized={qcoeff[0, 0, 0, :2].tolist()}"
    )


def test_quantize_component_gradient():
    """Verify gradients propagate through quantization."""
    coeff = torch.zeros((1, 1, 8, 8))
    coeff[0, 0, 0, 0] = 12.0
    coeff.requires_grad_()
    qtable = torch.full((8, 8), 5.0, requires_grad=True)

    qcoeff, _ = quantize_component(coeff, qtable)

    qcoeff.sum().backward()

    assert coeff.grad is not None
    assert qtable.grad is not None
    assert torch.isfinite(coeff.grad).all()
    assert torch.isfinite(qtable.grad).all()

    print(
        "\nQuantization gradients:"
        f"\n  Coefficients: "
        f"mean_abs={coeff.grad.abs().mean().item():.6e}, "
        f"max_abs={coeff.grad.abs().max().item():.6e}, "
        f"nonzero={torch.count_nonzero(coeff.grad).item()}/{coeff.grad.numel()}"
        f"\n  Quantization table: "
        f"mean_abs={qtable.grad.abs().mean().item():.6e}, "
        f"max_abs={qtable.grad.abs().max().item():.6e}, "
        f"nonzero={torch.count_nonzero(qtable.grad).item()}/{qtable.grad.numel()}"
    )


@pytest.mark.parametrize(
    "jpeg_template",
    [0, 1, 2],
    indirect=True,
    ids=["444", "422", "420"],
)
def test_jpeg_round_trip(jpeg_template, tmp_path):
    """Verify lossless JPEG round-tripping for different sampling modes."""
    original = jio.read(str(jpeg_template))
    state = make_state(original)

    output_path = tmp_path / "roundtrip.jpg"

    write_jpeg_state(state, jpeg_template, output_path)

    rewritten = jio.read(str(output_path))

    for i, component in enumerate(original.comp_info):
        np.testing.assert_array_equal(
            rewritten.coef_arrays[i],
            original.coef_arrays[i],
        )

        index = component.quant_tbl_no

        np.testing.assert_array_equal(
            rewritten.quant_tables[index],
            original.quant_tables[index],
        )

    with Image.open(jpeg_template) as image:
        reference = np.asarray(image.convert("RGB"))

    with Image.open(output_path) as image:
        actual = np.asarray(image.convert("RGB"))

    np.testing.assert_array_equal(actual, reference)

    original_bytes = jpeg_template.stat().st_size
    output_bytes = output_path.stat().st_size

    sampling = [(c.h_samp_factor, c.v_samp_factor) for c in original.comp_info]

    print(
        f"\nJPEG round-trip: "
        f"sampling={sampling}, "
        f"{original_bytes} -> {output_bytes} bytes "
        f"({100 * output_bytes / original_bytes:.1f}%)"
    )


def test_jpeg_coefficient_modification(jpeg_template, tmp_path):
    """Verify that modifying a DCT coefficient changes the JPEG."""
    original = jio.read(str(jpeg_template))
    state = make_state(original)

    # Modify one luma AC coefficient.
    state[0]["coeff"][0, 0, 0, 1] += 10 * state[0]["qtable"][0, 1]

    output_path = tmp_path / "modified_coeff.jpg"

    write_jpeg_state(state, jpeg_template, output_path)

    modified = jio.read(str(output_path))

    expected = original.coef_arrays[0][0, 1] + 10

    assert modified.coef_arrays[0][0, 1] == expected

    # Verify a conventional decoder can read the result.
    with Image.open(output_path) as image:
        image.load()

    print(
        f"\nCoefficient modification: "
        f"{original.coef_arrays[0][0, 1]} -> "
        f"{modified.coef_arrays[0][0, 1]}"
    )


def test_jpeg_quantization_modification(jpeg_template, tmp_path):
    """Verify that modified quantization tables survive export."""
    original = jio.read(str(jpeg_template))
    state = make_state(original)

    # Change the DC quantization value for luma.
    table_index = original.comp_info[0].quant_tbl_no
    old_value = int(original.quant_tables[table_index][0, 0])
    new_value = old_value + 1 if old_value < 255 else old_value - 1

    state[0]["qtable"][0, 0] = float(new_value)

    output_path = tmp_path / "modified_qtable.jpg"

    write_jpeg_state(state, jpeg_template, output_path)

    modified = jio.read(str(output_path))

    assert modified.quant_tables[table_index][0, 0] == new_value

    with Image.open(output_path) as image:
        image.load()

    print(f"\nQuantization modification: Q[0,0]={old_value} -> {new_value}")


def test_jpeg_rejects_shared_qtable_mismatch(jpeg_template, tmp_path):
    """Reject conflicting quantization tables shared by components."""
    jpeg = jio.read(str(jpeg_template))
    state = make_state(jpeg)

    # Pillow normally assigns the same quantization table to Cb and Cr.
    assert jpeg.comp_info[1].quant_tbl_no == jpeg.comp_info[2].quant_tbl_no

    state[2]["qtable"][0, 0] += 1.0

    output_path = tmp_path / "invalid.jpg"

    with pytest.raises(ValueError, match="sharing quantization table"):
        write_jpeg_state(state, jpeg_template, output_path)

    assert not output_path.exists()

    print("\nShared quantization table mismatch correctly rejected.")


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_jpeg_rejects_nonfinite_coefficients(jpeg_template, tmp_path, value):
    """Reject non-finite DCT coefficients."""
    jpeg = jio.read(str(jpeg_template))
    state = make_state(jpeg)

    state[0]["coeff"][0, 0, 0, 0] = value

    output = tmp_path / "invalid.jpg"

    with pytest.raises(ValueError, match="must be finite"):
        write_jpeg_state(state, jpeg_template, output)

    assert not output.exists()


@pytest.mark.parametrize("value", [-1025, 1024])
def test_jpeg_rejects_coefficient_overflow(jpeg_template, tmp_path, value):
    """Reject coefficients outside the supported JPEG range."""
    jpeg = jio.read(str(jpeg_template))
    state = make_state(jpeg)

    qvalue = state[0]["qtable"][0, 0]

    state[0]["coeff"][0, 0, 0, 0] = value * qvalue

    output = tmp_path / "invalid.jpg"

    with pytest.raises(ValueError, match="between -1024 and 1023"):
        write_jpeg_state(state, jpeg_template, output)

    assert not output.exists()


def test_jpeg_rejects_invalid_dimensions(jpeg_template, tmp_path):
    """Reject coefficient arrays that do not match the template."""
    jpeg = jio.read(str(jpeg_template))
    state = make_state(jpeg)

    state[0]["coeff"] = state[0]["coeff"][:-1]

    output = tmp_path / "invalid.jpg"

    with pytest.raises(ValueError, match="expected coefficient shape"):
        write_jpeg_state(state, jpeg_template, output)

    assert not output.exists()
