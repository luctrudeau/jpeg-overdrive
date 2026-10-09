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

"""JPEG bitstream export utilities."""

from pathlib import Path

import jpegio as jio
import numpy as np
import torch

from jpegoverdrive.utils import blocks_to_array, ste_round

# Legal coefficient range for 8-bit JPEG (libjpeg-turbo, jchuff.h).
MIN_DCT_COEFF = -1024
MAX_DCT_COEFF = 1023


def quantize_component(
    coeff: torch.Tensor,
    qtable: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize DCT coefficients and quantization table using STE.

    Args:
        coeff: Dequantized DCT coefficients with shape (..., 8, 8).
        qtable: Quantization table with shape (8, 8).

    Returns:
        Tuple containing quantized DCT coefficients and the
        integer-valued quantization table.
    """
    qtable = ste_round(torch.clamp(qtable, 1.0, 255.0))
    qcoeff = ste_round(coeff / qtable)

    return qcoeff, qtable


def write_jpeg_state(
    state: list[dict[str, torch.Tensor]],
    template_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Export an optimizer state as a JPEG using an existing template.

    The template provides the JPEG structure, component information,
    sampling factors, and entropy coding configuration.

    Args:
        state: List of component dictionaries containing dequantized
            DCT coefficients and quantization tables.
        template_path: Path to the original JPEG template.
        output_path: Destination path for the encoded JPEG.

    Returns:
        Path to the written JPEG file.

    Raises:
        ValueError: If the state contains invalid dimensions, non-finite
            values, out-of-range coefficients, or conflicting tables.
    """
    jpeg = jio.read(str(template_path))

    if len(state) != len(jpeg.comp_info):
        raise ValueError(
            f"Expected {len(jpeg.comp_info)} components, got {len(state)}."
        )

    # Quantization tables may be shared between JPEG components.
    used_qtables = {}

    for i, component in enumerate(state):
        coeff = component["coeff"]
        qtable = component["qtable"]

        # Validate tensor dimensions.
        expected_shape = jpeg.coef_arrays[i].shape
        expected_blocks = (
            expected_shape[0] // 8,
            expected_shape[1] // 8,
            8,
            8,
        )

        if tuple(coeff.shape) != expected_blocks:
            raise ValueError(
                f"Component {i}: expected coefficient shape "
                f"{expected_blocks}, got {tuple(coeff.shape)}."
            )

        if tuple(qtable.shape) != (8, 8):
            raise ValueError(
                f"Component {i}: expected quantization table shape "
                f"(8, 8), got {tuple(qtable.shape)}."
            )

        # Validate continuous inputs before quantization.
        if not bool(torch.isfinite(coeff).all()):
            raise ValueError(f"Component {i}: DCT coefficients must be finite.")

        if not bool(torch.isfinite(qtable).all()):
            raise ValueError(f"Component {i}: quantization table must be finite.")

        qcoeff, qtable = quantize_component(coeff, qtable)

        # Validate the quantized values before integer conversion.
        if not bool(torch.isfinite(qcoeff).all()):
            raise ValueError(
                f"Component {i}: quantized DCT coefficients must be finite."
            )

        if not bool(torch.isfinite(qtable).all()):
            raise ValueError(f"Component {i}: quantized table must be finite.")

        # Reject coefficients outside the supported 8-bit JPEG range.
        if bool(((qcoeff < MIN_DCT_COEFF) | (qcoeff > MAX_DCT_COEFF)).any()):
            raise ValueError(
                f"Component {i}: quantized DCT coefficients must be "
                f"between {MIN_DCT_COEFF} and {MAX_DCT_COEFF}."
            )

        qtable_index = jpeg.comp_info[i].quant_tbl_no

        qtable_array = qtable.detach().cpu().numpy().astype(np.int32)

        # Ensure shared quantization tables remain consistent.
        if qtable_index in used_qtables:
            if not np.array_equal(
                used_qtables[qtable_index],
                qtable_array,
            ):
                raise ValueError(
                    f"Components sharing quantization table "
                    f"{qtable_index} have different values."
                )
        else:
            used_qtables[qtable_index] = qtable_array

        coef_array = blocks_to_array(qcoeff)

        jpeg.coef_arrays[i][:] = coef_array.detach().cpu().numpy().astype(np.int32)

    # Update quantization tables.
    for index, table in used_qtables.items():
        jpeg.quant_tables[index][:] = table

    # Optimize Huffman coding for the modified coefficients.
    jpeg.optimize_coding = True

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jio.write(jpeg, str(output_path))

    return output_path
