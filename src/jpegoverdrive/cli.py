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

    # Measure the original JPEG using the conventional decoder.
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
        steps=args.steps,
        lr=args.lr,
        snapshot_every=args.snapshot_every,
        seed=args.seed,
    )

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

    # Preserve all measured snapshots for later analysis.
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
            "snapshot_every": args.snapshot_every,
            "seed": args.seed,
            "device": str(device),
            "net": args.net,
        },
        "selection": "best_lpips_at_or_below_baseline_size",
        "baseline": baseline,
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
