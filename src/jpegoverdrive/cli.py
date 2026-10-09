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

"""Command-line interface for JPEG Overdrive."""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

from . import __version__


def get_device(name: str):
    """Resolve the requested PyTorch device."""
    import torch

    if name != "auto":
        device = torch.device(name)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available.")

        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available.")

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    """Create the JPEG Overdrive command-line parser."""
    parser = argparse.ArgumentParser(
        description="JPEG-compliant learned perceptual image compression."
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Optimize an existing JPEG bitstream.",
    )

    optimize_parser.add_argument(
        "source",
        type=Path,
        help="Original reference image.",
    )

    optimize_parser.add_argument(
        "baseline",
        type=Path,
        help="JPEG used to initialize optimization.",
    )

    optimize_parser.add_argument(
        "output",
        type=Path,
        help="Destination for the optimized JPEG.",
    )

    optimize_parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of optimization steps (default: 1000).",
    )

    optimize_parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Adam learning rate (default: 0.001).",
    )

    optimize_parser.add_argument(
        "--lambda-rd",
        type=float,
        default=0.001,
        help="Rate-distortion weight (default: 0.001).",
    )

    optimize_parser.add_argument(
        "--distortion-scale",
        type=float,
        default=None,
        metavar="VALUE",
        help=("LPIPS normalization scale. Default: max(initial LPIPS, 1e-3)."),
    )

    optimize_parser.add_argument(
        "--snapshot-every",
        type=int,
        default=50,
        help="Save a JPEG every N steps (default: 50).",
    )

    optimize_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )

    optimize_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device (default: auto).",
    )

    optimize_parser.add_argument(
        "--net",
        choices=["alex", "vgg"],
        default="alex",
        help="LPIPS backbone (default: alex).",
    )

    return parser


def compare_jpeg_states(
    baseline_path: str | Path,
    snapshot_path: str | Path,
) -> dict:
    """Compare the baseline JPEG against an exported snapshot.

    Compare quantization tables and quantized DCT coefficients directly,
    then compare conventional RGB decoder output.

    A representation-preserving rewrite may have a different file size
    because entropy coding and JPEG metadata can differ.
    """
    import jpegio as jio
    import numpy as np
    from PIL import Image

    baseline_path = Path(baseline_path)
    snapshot_path = Path(snapshot_path)

    baseline = jio.read(str(baseline_path))
    snapshot = jio.read(str(snapshot_path))

    if (
        baseline.image_height != snapshot.image_height
        or baseline.image_width != snapshot.image_width
    ):
        raise ValueError("JPEG dimensions do not match.")

    if len(baseline.comp_info) != len(snapshot.comp_info):
        raise ValueError("JPEG component counts do not match.")

    components = []

    for i, (base_comp, snap_comp) in enumerate(
        zip(baseline.comp_info, snapshot.comp_info)
    ):
        base_qtable = np.asarray(
            baseline.quant_tables[base_comp.quant_tbl_no],
        )
        snap_qtable = np.asarray(
            snapshot.quant_tables[snap_comp.quant_tbl_no],
        )

        base_coeff = np.asarray(baseline.coef_arrays[i])
        snap_coeff = np.asarray(snapshot.coef_arrays[i])

        if base_qtable.shape != snap_qtable.shape:
            raise ValueError(f"Quantization table shapes differ for component {i}.")

        if base_coeff.shape != snap_coeff.shape:
            raise ValueError(f"Coefficient array shapes differ for component {i}.")

        components.append(
            {
                "name": (("Y", "Cb", "Cr")[i] if i < 3 else str(i)),
                "qtable_changes": int(np.count_nonzero(base_qtable != snap_qtable)),
                "coefficient_changes": int(np.count_nonzero(base_coeff != snap_coeff)),
            }
        )

    with Image.open(baseline_path) as image:
        baseline_rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )

    with Image.open(snapshot_path) as image:
        snapshot_rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )

    if baseline_rgb.shape != snapshot_rgb.shape:
        raise ValueError("Decoded RGB dimensions do not match.")

    difference = np.abs(baseline_rgb.astype(np.int16) - snapshot_rgb.astype(np.int16))

    baseline_bytes = baseline_path.stat().st_size
    snapshot_bytes = snapshot_path.stat().st_size

    qtable_changes = sum(row["qtable_changes"] for row in components)

    coefficient_changes = sum(row["coefficient_changes"] for row in components)

    return {
        "baseline_path": baseline_path,
        "snapshot_path": snapshot_path,
        "components": components,
        "qtable_changes": qtable_changes,
        "coefficient_changes": coefficient_changes,
        "rgb_sample_changes": int(np.count_nonzero(difference)),
        "max_rgb_error": int(difference.max()),
        "baseline_bytes": baseline_bytes,
        "snapshot_bytes": snapshot_bytes,
        "size_difference_bytes": snapshot_bytes - baseline_bytes,
        "size_difference_percent": 100.0 * (snapshot_bytes / baseline_bytes - 1.0),
    }


def print_initialization_validation(result: dict) -> None:
    """Print the JPEG representation round-trip diagnostics."""
    print()
    print("=" * 64)
    print(" JPEG Overdrive — Initialization Validation")
    print("=" * 64)

    print(f"{'Component':<12}{'Q-table changes':>20}{'Coefficient changes':>22}")
    print("-" * 54)

    for component in result["components"]:
        print(
            f"{component['name']:<12}"
            f"{component['qtable_changes']:>20,}"
            f"{component['coefficient_changes']:>22,}"
        )

    print("-" * 54)

    print(f"Different RGB samples: {result['rgb_sample_changes']:,}")
    print(f"Maximum RGB error:     {result['max_rgb_error']}")

    print()
    print(f"Baseline size:         {result['baseline_bytes']:,} bytes")
    print(f"Step 0 size:           {result['snapshot_bytes']:,} bytes")
    print(
        f"Size difference:       "
        f"{result['size_difference_bytes']:+,} bytes "
        f"({result['size_difference_percent']:+.2f}%)"
    )

    print()

    if (
        result["qtable_changes"] == 0
        and result["coefficient_changes"] == 0
        and result["rgb_sample_changes"] == 0
    ):
        print("Result: Exact JPEG representation round-trip.")
    elif result["qtable_changes"] == 0 and result["coefficient_changes"] == 0:
        print(
            "Result: JPEG coefficients and tables preserved, but decoded RGB differs."
        )
    else:
        print("Result: JPEG representation changed at initialization.")

    print("=" * 64)


def save_history(
    history: list[dict],
    path: Path,
) -> None:
    """Save optimization snapshot measurements as CSV."""
    if not history:
        raise ValueError("Optimization history is empty.")

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(history[0])

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in history:
            writer.writerow(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in row.items()
                }
            )


def json_safe(value):
    """Convert experiment results to JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    return value


def save_summary(
    summary: dict,
    path: Path,
) -> None:
    """Save the optimization summary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            json_safe(summary),
            file,
            indent=2,
            allow_nan=False,
        )

        file.write("\n")


def print_optimization_summary(
    baseline: dict,
    best: dict,
    output: Path,
) -> None:
    """Print the actual rate-distortion comparison."""
    rate_savings = 100.0 * (1.0 - best["bytes"] / baseline["bytes"])

    if baseline["lpips"] > 0:
        lpips_improvement = 100.0 * (1.0 - best["lpips"] / baseline["lpips"])

        lpips_change = f"{lpips_improvement:+.2f}%"
    else:
        lpips_change = "N/A (zero baseline)"

    if math.isfinite(baseline["psnr"]) and math.isfinite(best["psnr"]):
        psnr_change = f"{best['psnr'] - baseline['psnr']:+.3f} dB"
    else:
        psnr_change = "N/A"

    print()
    print("=" * 64)
    print(" JPEG Overdrive — Optimization Summary")
    print("=" * 64)

    if best["is_baseline"]:
        print("Selection: Original baseline retained")
    else:
        print(f"Selection: Best eligible snapshot (step {best['step']})")

    print()
    print(f"{'Metric':<19}{'Baseline':>17}{'Selected':>17}")
    print("-" * 53)

    print(f"{'Size (bytes)':<19}{baseline['bytes']:>17,}{best['bytes']:>17,}")

    print(f"{'BPP':<19}{baseline['bpp']:>17.4f}{best['bpp']:>17.4f}")

    print(f"{'PSNR (dB)':<19}{baseline['psnr']:>17.3f}{best['psnr']:>17.3f}")

    print(f"{'LPIPS':<19}{baseline['lpips']:>17.8f}{best['lpips']:>17.8f}")

    print("-" * 53)

    print(f"Rate savings:       {rate_savings:+.2f}%")
    print(f"LPIPS improvement:  {lpips_change}")
    print(f"PSNR improvement:   {psnr_change}")

    print()
    print(f"Output: {output}")
    print("=" * 64)


def run_optimize(args: argparse.Namespace) -> None:
    """Optimize a JPEG and export the best measured candidate."""
    import jpegio as jio

    from jpegoverdrive.metrics import (
        ImageReference,
        create_lpips_metric,
        measure_jpeg,
    )
    from jpegoverdrive.optimize import (
        extract_jpeg_state,
        optimize_jpeg,
        select_best_candidate,
    )

    if args.output.resolve() == args.baseline.resolve():
        raise ValueError("Output must be different from the baseline JPEG.")

    device = get_device(args.device)

    reference = ImageReference.from_file(
        args.source,
        device=device,
    )

    metric = create_lpips_metric(
        device=device,
        net=args.net,
    )

    jpeg = jio.read(str(args.baseline))

    initial_state = extract_jpeg_state(
        jpeg,
        device=device,
    )

    baseline = measure_jpeg(
        args.baseline,
        reference,
        metric,
    )

    snapshot_dir = args.output.parent / f"{args.output.stem}_snapshots"

    _, history = optimize_jpeg(
        initial_state=initial_state,
        template_path=args.baseline,
        output_dir=snapshot_dir,
        reference=reference,
        metric=metric,
        lambda_rd=args.lambda_rd,
        distortion_scale=args.distortion_scale,
        steps=args.steps,
        lr=args.lr,
        snapshot_every=args.snapshot_every,
        seed=args.seed,
    )

    # Validate the actual step-0 bitstream against the baseline.
    initialization = compare_jpeg_states(
        args.baseline,
        history[0]["path"],
    )

    print_initialization_validation(initialization)

    # Select using actual JPEG measurements, not the rate proxy.
    best = select_best_candidate(
        history,
        baseline,
    )

    selected_path = args.baseline if best["is_baseline"] else best["path"]

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copyfile(
        selected_path,
        args.output,
    )

    history_path = snapshot_dir / "history.csv"
    summary_path = snapshot_dir / "summary.json"

    save_history(
        history,
        history_path,
    )

    rate_savings = 100.0 * (1.0 - best["bytes"] / baseline["bytes"])

    if baseline["lpips"] > 0:
        lpips_improvement = 100.0 * (1.0 - best["lpips"] / baseline["lpips"])
    else:
        lpips_improvement = None

    if math.isfinite(baseline["psnr"]) and math.isfinite(best["psnr"]):
        psnr_improvement = best["psnr"] - baseline["psnr"]
    else:
        psnr_improvement = None

    summary = {
        "version": __version__,
        "source": args.source,
        "baseline_path": args.baseline,
        "output": args.output,
        "configuration": {
            "steps": args.steps,
            "lr": args.lr,
            "lambda_rd": args.lambda_rd,
            "distortion_scale": args.distortion_scale,
            "snapshot_every": args.snapshot_every,
            "seed": args.seed,
            "device": str(device),
            "net": args.net,
        },
        "selection": "best_lpips_at_or_below_baseline_size",
        "baseline": baseline,
        "initialization": initialization,
        "selected": best,
        "rate_savings_percent": rate_savings,
        "lpips_improvement_percent": lpips_improvement,
        "psnr_improvement_db": psnr_improvement,
    }

    save_summary(
        summary,
        summary_path,
    )

    print_optimization_summary(
        baseline,
        best,
        args.output,
    )

    print(f"History: {history_path}")
    print(f"Summary: {summary_path}")


def main(argv: list[str] | None = None) -> None:
    """Run the JPEG Overdrive CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "optimize":
        run_optimize(args)


if __name__ == "__main__":
    main()
