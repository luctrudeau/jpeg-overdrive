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

"""Differentiable JPEG inverse discrete cosine transform (IDCT).

Based on the ISLOW integer IDCT implementation in libjpeg-turbo:
https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/src/jidctint.c

The forward pass approximates libjpeg's fixed-point arithmetic,
while straight-through estimators enable gradient propagation.
"""

import torch

# Fixed-point precision used by libjpeg's ISLOW IDCT.
ISLOW_CONST_BITS = 13
ISLOW_PASS1_BITS = 2

# Fixed-point constants scaled by 2**ISLOW_CONST_BITS.
FIX_0_298631336 = 2446
FIX_0_390180644 = 3196
FIX_0_541196100 = 4433
FIX_0_765366865 = 6270
FIX_0_899976223 = 7373
FIX_1_175875602 = 9633
FIX_1_501321110 = 12299
FIX_1_847759065 = 15137
FIX_1_961570560 = 16069
FIX_2_053119869 = 16819
FIX_2_562915447 = 20995
FIX_3_072711026 = 25172


def descale_ste(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Apply libjpeg-style fixed-point descaling with surrogate gradients.

    The forward pass computes:

        floor((x + 2**(bits - 1)) / 2**bits)

    The backward pass uses the derivative of x / 2**bits.

    Args:
        x: Input tensor.
        bits: Number of fractional bits to remove.

    Returns:
        Descaled tensor with straight-through gradients.
    """
    scale = float(1 << bits)
    bias = float(1 << (bits - 1))

    continuous = x / scale
    rounded = torch.floor((x + bias) / scale)

    return continuous + (rounded - continuous).detach()


def idct_islow_1d(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Apply one pass of the differentiable ISLOW inverse DCT.

    Implements the fixed-point butterfly structure used by libjpeg.

    The transform operates on the last dimension of the input,
    which must have length 8.

    Args:
        x: Input tensor with shape (..., 8).
        shift: Number of bits to remove after the transform.

    Returns:
        Transformed tensor with the same shape as the input.
    """
    x0, x1, x2, x3, x4, x5, x6, x7 = x.unbind(dim=-1)

    # Even part.
    z2 = x2
    z3 = x6
    z1 = (z2 + z3) * FIX_0_541196100

    tmp2 = z1 - z3 * FIX_1_847759065
    tmp3 = z1 + z2 * FIX_0_765366865

    tmp0 = (x0 + x4) * (1 << ISLOW_CONST_BITS)
    tmp1 = (x0 - x4) * (1 << ISLOW_CONST_BITS)

    tmp10 = tmp0 + tmp3
    tmp13 = tmp0 - tmp3
    tmp11 = tmp1 + tmp2
    tmp12 = tmp1 - tmp2

    # Odd part.
    t0 = x7
    t1 = x5
    t2 = x3
    t3 = x1

    z1 = t0 + t3
    z2 = t1 + t2
    z3 = t0 + t2
    z4 = t1 + t3

    z5 = (z3 + z4) * FIX_1_175875602

    t0 = t0 * FIX_0_298631336
    t1 = t1 * FIX_2_053119869
    t2 = t2 * FIX_3_072711026
    t3 = t3 * FIX_1_501321110

    z1 = -z1 * FIX_0_899976223
    z2 = -z2 * FIX_2_562915447
    z3 = -z3 * FIX_1_961570560
    z4 = -z4 * FIX_0_390180644

    z3 = z3 + z5
    z4 = z4 + z5

    t0 = t0 + z1 + z3
    t1 = t1 + z2 + z4
    t2 = t2 + z2 + z3
    t3 = t3 + z1 + z4

    return torch.stack(
        [
            descale_ste(tmp10 + t3, shift),
            descale_ste(tmp11 + t2, shift),
            descale_ste(tmp12 + t1, shift),
            descale_ste(tmp13 + t0, shift),
            descale_ste(tmp13 - t0, shift),
            descale_ste(tmp12 - t1, shift),
            descale_ste(tmp11 - t2, shift),
            descale_ste(tmp10 - t3, shift),
        ],
        dim=-1,
    )
