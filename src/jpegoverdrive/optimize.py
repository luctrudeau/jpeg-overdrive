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

"""Gradient-based rate-distortion optimization for JPEG."""

import math
import time
from pathlib import Path

import jpegio as jio
import torch

from jpegoverdrive.decoder import decode_component, ycbcr_to_rgb
from jpegoverdrive.jpeg import quantize_component, write_jpeg_state
from jpegoverdrive.metrics import ImageReference, measure_jpeg
from jpegoverdrive.utils import array_to_blocks, ste_round

DISTORTION_FLOOR = 1e-3
RATE_FLOOR = 1e-8


def extract_jpeg_state(
    jpeg: jio.DecompressedJpeg,
    device: torch.device | str,
) -> list[dict]:
    """Extract quantized DCT coefficients and quantization tables."""
    names = ("Y", "Cb", "Cr")
    state = []

    for i, component in enumerate(jpeg.comp_info):
        qcoeff = array_to_blocks(
            torch.as_tensor(
                jpeg.coef_arrays[i],
                dtype=torch.float32,
                device=device,
            )
        )

        qtable = torch.as_tensor(
            jpeg.quant_tables[component.quant_tbl_no],
            dtype=torch.float32,
            device=device,
        ).clone()

        state.append(
            {
                "name": names[i] if i < len(names) else str(i),
                "qcoeff": qcoeff,
                "qtable": qtable,
            }
        )

    return state


def prepare_metrics(
    metric: torch.nn.Module | dict[str, torch.nn.Module],
) -> None:
    """Validate and freeze perceptual metrics."""
    if isinstance(metric, dict):
        if set(metric) != {"alex", "vgg"}:
            raise ValueError("Joint optimization requires 'alex' and 'vgg' metrics.")

        metrics = metric.values()
    else:
        metrics = (metric,)

    for model in metrics:
        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad_(False)


def validate_initial_state(
    initial_state: list[dict],
    jpeg: jio.DecompressedJpeg,
) -> None:
    """Verify that the initial state matches the JPEG template."""

    if len(initial_state) != len(jpeg.comp_info):
        raise ValueError("Initial state component count does not match JPEG template.")

    template_state = extract_jpeg_state(jpeg, device="cpu")

    for i, (component, expected) in enumerate(zip(initial_state, template_state)):
        for key in ("qcoeff", "qtable"):
            actual_tensor = component[key]

            if actual_tensor.shape != expected[key].shape:
                raise ValueError(
                    f"Initial state component {i} has an incorrect {key} shape."
                )

            actual = actual_tensor.detach().to("cpu")

            if not torch.equal(actual, expected[key]):
                raise ValueError(
                    f"Initial state component {i} has "
                    f"{key} values that do not match "
                    "the JPEG template."
                )


def make_trainable_state(
    jpeg: jio.DecompressedJpeg,
    device: torch.device | str,
    seed: int = 0,
) -> list[dict]:
    """Create a reproducible trainable state from a JPEG."""
    initial_state = extract_jpeg_state(jpeg, device)

    return make_trainable_state_from_state(
        initial_state,
        jpeg,
        device,
        seed,
    )


def make_trainable_state_from_state(
    initial_state: list[dict],
    jpeg: jio.DecompressedJpeg,
    device: torch.device | str,
    seed: int = 0,
) -> list[dict]:
    """Initialize dequantized coefficients and shared quantization tables.

    Random dithering moves parameters within their quantization bins.
    Components referencing the same JPEG quantization table share one
    trainable parameter.
    """
    device = torch.device(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if len(initial_state) != len(jpeg.comp_info):
        raise ValueError("State component count does not match JPEG.")

    state = []
    shared_qtables = {}
    shared_references = {}

    for i, component in enumerate(initial_state):
        qindex = jpeg.comp_info[i].quant_tbl_no

        qcoeff = (
            component["qcoeff"]
            .to(
                device=device,
                dtype=torch.float32,
            )
            .detach()
            .clone()
        )

        qtable_reference = (
            component["qtable"]
            .to(
                device=device,
                dtype=torch.float32,
            )
            .detach()
            .clone()
        )

        if qindex not in shared_qtables:
            q_dither = torch.empty_like(qtable_reference).uniform_(
                -0.49,
                0.49,
                generator=generator,
            )

            shared_qtables[qindex] = torch.nn.Parameter(qtable_reference + q_dither)

            shared_references[qindex] = qtable_reference

        elif not torch.equal(
            shared_references[qindex],
            qtable_reference,
        ):
            raise ValueError(f"Conflicting initial quantization table {qindex}.")

        coeff_dither = torch.empty_like(qcoeff).uniform_(
            -0.49,
            0.49,
            generator=generator,
        )

        coeff = torch.nn.Parameter((qcoeff + coeff_dither) * qtable_reference)

        state.append(
            {
                "name": component.get("name", str(i)),
                "qtable_index": qindex,
                "qtable": shared_qtables[qindex],
                "coeff": coeff,
                "qcoeff_reference": qcoeff,
                "qtable_reference": shared_references[qindex],
            }
        )

    return state


def trainable_parameters(
    state: list[dict],
) -> list[torch.nn.Parameter]:
    """Return unique trainable parameters from a JPEG state."""
    parameters = []
    seen = set()

    for component in state:
        for key in ("qtable", "coeff"):
            parameter = component[key]

            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))

    return parameters


def rate_proxy(qcoeff: torch.Tensor) -> torch.Tensor:
    """Compute a differentiable proxy for JPEG coefficient rate.

    This heuristic rate model is not an estimate of the actual
    Huffman-coded bitstream length.
    """
    magnitude = torch.abs(qcoeff)

    return (torch.log2(1.0 + magnitude) + 1.0 - torch.exp(-magnitude)).sum()


def jpeg_rate_proxy(state: list[dict]) -> torch.Tensor:
    """Accumulate the coefficient rate proxy over JPEG components."""
    rate = state[0]["coeff"].new_zeros(())

    for component in state:
        qcoeff, _ = quantize_component(
            component["coeff"],
            component["qtable"],
        )

        rate = rate + rate_proxy(qcoeff)

    return rate


def decode_jpeg_state(
    state: list[dict],
    height: int,
    width: int,
) -> torch.Tensor:
    """Decode a trainable JPEG state into an RGB tensor.

    Currently supports three-component 4:4:4 YCbCr images.

    Returns:
        RGB tensor with shape (H, W, 3) and values in [0, 255].
    """
    if len(state) != 3:
        raise ValueError("Expected three JPEG components.")

    planes = []

    for component in state:
        qcoeff, qtable = quantize_component(
            component["coeff"],
            component["qtable"],
        )

        plane = decode_component(qcoeff, qtable)
        planes.append(plane[:height, :width])

    if any(plane.shape != (height, width) for plane in planes):
        raise ValueError("JPEG state must contain full-resolution 4:4:4 components.")

    rgb = ycbcr_to_rgb(*planes)

    return ste_round(rgb).clamp(0.0, 255.0)


def tensor_rgb_to_lpips(image: torch.Tensor) -> torch.Tensor:
    """Convert HWC RGB [0, 255] to LPIPS NCHW [-1, 1]."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB tensor, got {tuple(image.shape)}.")

    image = image.permute(2, 0, 1).unsqueeze(0)

    return 2.0 * image / 255.0 - 1.0


def compute_objectives(
    state: list[dict],
    reference: ImageReference,
    metric: torch.nn.Module | dict[str, torch.nn.Module],
    initial_distortion: torch.Tensor | dict[str, float],
    initial_rate: torch.Tensor,
    height: int,
    width: int,
    *,
    return_details: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
):
    """Compute differentiable normalized distortion and rate objectives.

    For joint LPIPS optimization, distortion is the maximum of the
    independently normalized Alex and VGG scores.

    Args:
        state: Trainable JPEG representation.
        reference: Cached source image representations.
        metric: Single LPIPS network or an Alex/VGG metric dictionary.
        initial_distortion: Normalization scales for perceptual distortion.
        initial_rate: Normalization scale for the differentiable rate proxy.
        height: Image height in pixels.
        width: Image width in pixels.
        return_details: Include individual normalized LPIPS scores.

    Returns:
        Normalized distortion and rate. When return_details is True,
        also returns a dictionary containing individual normalized
        distortions for joint optimization.
    """
    rgb = decode_jpeg_state(state, height, width)

    rgb_lpips = tensor_rgb_to_lpips(rgb)

    if isinstance(metric, dict):
        d_alex = (
            metric["alex"](reference.lpips, rgb_lpips).mean()
            / initial_distortion["alex"]
        )

        d_vgg = (
            metric["vgg"](reference.lpips, rgb_lpips).mean() / initial_distortion["vgg"]
        )

        distortion = torch.maximum(d_alex, d_vgg)
    else:
        distortion = metric(reference.lpips, rgb_lpips).mean() / initial_distortion

    rate = jpeg_rate_proxy(state) / initial_rate

    if return_details:
        details = {}

        if isinstance(metric, dict):
            details = {
                "alex_norm": d_alex,
                "vgg_norm": d_vgg,
            }

        return distortion, rate, details

    return distortion, rate


def select_best_candidate(
    history: list[dict],
    baseline: dict,
    nets: tuple[str, ...] = ("alex",),
) -> dict:
    """Select the best JPEG without exceeding the baseline file size.

    Single-network selection minimizes LPIPS. Joint selection minimizes
    the maximum independently normalized Alex and VGG scores, using
    actual JPEG measurements.

    The baseline remains eligible, and smaller files win ties.

        Raises:
            ValueError: If joint selection is requested but the baseline
                or any candidate is missing its VGG LPIPS measurement.
    """

    if len(nets) > 1:
        if "lpips_vgg" not in baseline:
            raise ValueError("Joint selection requires 'lpips_vgg' in the baseline.")

        for i, candidate in enumerate(history):
            if "lpips_vgg" not in candidate:
                raise ValueError(
                    f"Joint selection requires 'lpips_vgg' in candidate {i}."
                )

    def score(row: dict) -> float:
        """Score a JPEG using single or worst-case normalized LPIPS."""
        if len(nets) == 1:
            return row["lpips"]

        return max(
            row["lpips"] / max(baseline["lpips"], DISTORTION_FLOOR),
            row["lpips_vgg"] / max(baseline["lpips_vgg"], DISTORTION_FLOOR),
        )

    best = {
        **baseline,
        "step": None,
        "is_baseline": True,
    }

    for candidate in history:
        distortion = score(candidate)

        if not math.isfinite(distortion):
            continue

        if candidate["bytes"] > baseline["bytes"]:
            continue

        if (
            distortion,
            candidate["bytes"],
        ) < (
            score(best),
            best["bytes"],
        ):
            best = {
                **candidate,
                "is_baseline": False,
            }

    return best


def optimize_jpeg(
    initial_state: list[dict],
    template_path: str | Path,
    output_dir: str | Path,
    reference: ImageReference,
    metric: torch.nn.Module | dict[str, torch.nn.Module],
    lambda_rd: float,
    distortion_scale: float | dict[str, float] | None = None,
    steps: int = 1000,
    lr: float = 0.001,
    snapshot_every: int = 50,
    seed: int = 0,
    max_grad_norm: float | None = None,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Optimize JPEG coefficients and quantization tables using Adam.

    Minimizes a rate-distortion objective using LPIPS and a differentiable
    rate proxy. With multiple LPIPS networks, distortion is the maximum
    of their independently normalized scores.

        Args:
            initial_state: Component dictionaries containing quantized DCT
                coefficients and quantization tables.
            template_path: JPEG used as the bitstream template.
            output_dir: Directory for snapshot JPEGs.
            reference: Cached source image representations.
            metric: Single LPIPS model or dictionary containing Alex and VGG
                models. Networks are frozen automatically.
            lambda_rd: Weight applied to normalized rate.
            steps: Number of Adam optimization steps.
            lr: Adam learning rate.
            snapshot_every: Interval for exporting and evaluating JPEGs.
            seed: Random seed used for parameter dithering.
            max_grad_norm: Optional gradient clipping threshold.
            verbose: Print optimization progress.
            distortion_scale: Optional normalization scale for LPIPS.
                Use a positive scalar for single-network optimization or a
                dictionary with separate Alex and VGG scales for joint mode.
                Defaults to the initial distortion, floored at DISTORTION_FLOOR.
                Explicitly provided scales are not clamped.

        Returns:
            Tuple containing the final trainable state and snapshot history.
    """
    if steps < 0:
        raise ValueError("steps must be nonnegative.")

    if snapshot_every < 1:
        raise ValueError("snapshot_every must be positive.")

    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite.")

    if not math.isfinite(lambda_rd) or lambda_rd < 0:
        raise ValueError("lambda_rd must be finite and nonnegative.")

    if max_grad_norm is not None:
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive and finite.")

    prepare_metrics(metric)

    template_path = Path(template_path)
    output_dir = Path(output_dir)

    jpeg = jio.read(str(template_path))

    if reference.height != jpeg.image_height:
        raise ValueError("Reference and JPEG heights do not match.")

    if reference.width != jpeg.image_width:
        raise ValueError("Reference and JPEG widths do not match.")

    validate_initial_state(initial_state, jpeg)

    if len(jpeg.comp_info) != 3 or any(
        component.h_samp_factor != 1 or component.v_samp_factor != 1
        for component in jpeg.comp_info
    ):
        raise ValueError("Optimization currently requires 4:4:4 JPEG.")

    device = reference.lpips.device

    # Initialize parameters from the supplied baseline state.
    state = make_trainable_state_from_state(
        initial_state,
        jpeg,
        device,
        seed,
    )

    parameters = trainable_parameters(state)

    optimizer = torch.optim.Adam(parameters, lr=lr)

    height = reference.height
    width = reference.width

    # Measure objectives at the initialized state.
    with torch.no_grad():
        initial_rgb = decode_jpeg_state(state, height, width)
        initial_lpips = tensor_rgb_to_lpips(initial_rgb)

        if isinstance(metric, dict):
            initial_distortion = {
                name: model(reference.lpips, initial_lpips).mean().detach()
                for name, model in metric.items()
            }
        else:
            initial_distortion = metric(reference.lpips, initial_lpips).mean().detach()

        initial_rate = jpeg_rate_proxy(state).detach()

    distortions = (
        initial_distortion.values()
        if isinstance(initial_distortion, dict)
        else (initial_distortion,)
    )

    for distortion in distortions:
        if not bool(torch.isfinite(distortion)):
            raise ValueError("Initial distortion is not finite.")

        if bool(distortion < 0):
            raise ValueError("Initial distortion must be nonnegative.")

    if not bool(torch.isfinite(initial_rate)):
        raise ValueError("Initial rate is not finite.")

    if bool(initial_rate < 0):
        raise ValueError("Initial rate must be nonnegative.")

    if isinstance(metric, dict):
        if distortion_scale is not None:
            if not isinstance(distortion_scale, dict):
                raise ValueError(
                    "Joint optimization requires separate distortion scales."
                )

            if set(distortion_scale) != {"alex", "vgg"}:
                raise ValueError("Distortion scales must contain 'alex' and 'vgg'.")

            for name, value in distortion_scale.items():
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"{name} distortion scale must be positive and finite."
                    )

    elif distortion_scale is not None:
        if isinstance(distortion_scale, dict):
            raise ValueError(
                "Single-metric optimization requires a scalar distortion scale."
            )

        distortion_scale = float(distortion_scale)

        if not math.isfinite(distortion_scale) or distortion_scale <= 0:
            raise ValueError("distortion_scale must be positive and finite.")

    if distortion_scale is None:
        if isinstance(initial_distortion, dict):
            distortion_scale = {
                name: max(value.item(), DISTORTION_FLOOR)
                for name, value in initial_distortion.items()
            }
        else:
            distortion_scale = max(
                initial_distortion.item(),
                DISTORTION_FLOOR,
            )

    rate_scale = initial_rate.clamp_min(RATE_FLOOR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []
    start_time = time.perf_counter()

    if verbose:
        n_parameters = sum(parameter.numel() for parameter in parameters)

        print(f"JPEG Overdrive | Adam | {device.type.upper()}")

        print(
            f"Image: {width}x{height} | "
            f"Components: {len(state)} | "
            f"Parameters: {n_parameters:,}"
        )

        print(f"Steps: {steps} | LR: {lr:g} | Lambda: {lambda_rd:g} | Seed: {seed}")

        if isinstance(distortion_scale, dict):
            scale_text = " | ".join(
                f"{name}: {value:.6g}" for name, value in distortion_scale.items()
            )
        else:
            scale_text = f"{distortion_scale:.6g}"

        print(f"Distortion scale: {scale_text} | Rate scale: {rate_scale.item():.6g}")

        print()

        if isinstance(metric, dict):
            print(
                f"{'Step':>6} "
                f"{'Loss':>9} "
                f"{'D/max':>9} "
                f"{'D/Alex':>9} "
                f"{'D/VGG':>9} "
                f"{'R/scale':>9} "
                f"{'bpp':>9} "
                f"{'PSNR':>9} "
                f"{'Alex':>12} "
                f"{'VGG':>12} "
                f"{'Time':>9}"
            )
            print("-" * 115)
        else:
            print(
                f"{'Step':>6} "
                f"{'Loss':>9} "
                f"{'D/scale':>9} "
                f"{'R/scale':>9} "
                f"{'bpp':>9} "
                f"{'PSNR':>9} "
                f"{'LPIPS':>12} "
                f"{'Time':>9}"
            )
            print("-" * 83)

    for step in range(steps + 1):
        if step > 0:
            optimizer.zero_grad(set_to_none=True)

            d_norm, r_norm = compute_objectives(
                state,
                reference,
                metric,
                distortion_scale,
                rate_scale,
                height,
                width,
            )

            loss = d_norm + lambda_rd * r_norm

            if not bool(torch.isfinite(loss).detach()):
                raise FloatingPointError(f"Non-finite loss at step {step}.")

            loss.backward()

            # Check gradients before modifying optimizer parameters.
            for parameter in parameters:
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError(f"Non-finite gradient at step {step}.")

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    max_grad_norm,
                    error_if_nonfinite=True,
                )

            optimizer.step()

            # Enforce valid JPEG quantization-table bounds.
            with torch.no_grad():
                for component in state:
                    component["qtable"].clamp_(
                        1.0,
                        255.0,
                    )

        if step % snapshot_every != 0 and step != steps:
            continue

        # Recompute objectives after the optimizer update.
        with torch.no_grad():
            d_norm, r_norm, details = compute_objectives(
                state,
                reference,
                metric,
                distortion_scale,
                rate_scale,
                height,
                width,
                return_details=True,
            )

            loss = d_norm + lambda_rd * r_norm
            rate = r_norm * rate_scale

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite snapshot loss at step {step}.")

        path = output_dir / f"step_{step:06d}.jpg"

        write_jpeg_state(
            state,
            template_path,
            path,
        )

        actual = measure_jpeg(
            path,
            reference,
            metric,
        )

        elapsed = time.perf_counter() - start_time

        row = {
            "step": step,
            "path": path,
            "loss": float(loss.item()),
            "distortion_norm": float(d_norm.item()),
            "rate_norm": float(r_norm.item()),
            "rate_proxy": float(rate.item()),
            "bytes": actual["bytes"],
            "bpp": actual["bpp"],
            "psnr": actual["psnr"],
            "lpips": actual["lpips"],
            "elapsed_seconds": elapsed,
        }

        if isinstance(metric, dict):
            row["lpips_vgg"] = actual["lpips_vgg"]
            row["alex_norm"] = float(details["alex_norm"].item())
            row["vgg_norm"] = float(details["vgg_norm"].item())

        for component in state:
            qcoeff, qtable = quantize_component(
                component["coeff"],
                component["qtable"],
            )

            name = component["name"]

            row[f"{name}_qtable_changes"] = int(
                torch.count_nonzero(qtable - component["qtable_reference"]).item()
            )

            row[f"{name}_coefficient_changes"] = int(
                torch.count_nonzero(qcoeff - component["qcoeff_reference"]).item()
            )

        history.append(row)

        if verbose:
            if isinstance(metric, dict):
                print(
                    f"{step:6d} "
                    f"{row['loss']:9.4f} "
                    f"{row['distortion_norm']:9.4f} "
                    f"{row['alex_norm']:9.4f} "
                    f"{row['vgg_norm']:9.4f} "
                    f"{row['rate_norm']:9.4f} "
                    f"{row['bpp']:9.4f} "
                    f"{row['psnr']:9.2f} "
                    f"{row['lpips']:12.6f} "
                    f"{row['lpips_vgg']:12.6f} "
                    f"{elapsed:8.1f}s"
                )
            else:
                print(
                    f"{step:6d} "
                    f"{row['loss']:9.4f} "
                    f"{row['distortion_norm']:9.4f} "
                    f"{row['rate_norm']:9.4f} "
                    f"{row['bpp']:9.4f} "
                    f"{row['psnr']:9.2f} "
                    f"{row['lpips']:12.6f} "
                    f"{elapsed:8.1f}s"
                )

    if verbose:
        print("-" * (115 if isinstance(metric, dict) else 83))

        print(
            f"Completed {steps} steps | "
            f"{len(history)} snapshots | "
            f"{history[-1]['elapsed_seconds']:.1f}s elapsed"
        )

    return state, history
