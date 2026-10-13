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

"""Benchmark full joint versus adaptive LPIPS optimization and compare JPEG RD."""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse benchmark settings shared by both optimizer runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original reference image.")
    parser.add_argument("baseline", type=Path, help="Initial JPEG (4:4:4).")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/adaptive_lpips_benchmark")
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda-rd", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--metric-check-every", type=int, default=10)
    parser.add_argument("--metric-switch-threshold", type=float, default=0.02)
    parser.add_argument(
        "--order", choices=("joint-first", "adaptive-first"), default="joint-first"
    )
    args = parser.parse_args()
    if args.steps < 1 or args.snapshot_every < 1 or args.metric_check_every < 1:
        parser.error("steps, snapshot-every and metric-check-every must be positive")
    if not 0 <= args.metric_switch_threshold < 1:
        parser.error("metric-switch-threshold must be in [0, 1)")
    for path in (args.source, args.baseline):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    return args


def run(mode: str, args: argparse.Namespace) -> tuple[float, list[dict[str, float]]]:
    """Run one optimizer configuration and load its exported snapshots."""
    output = args.output_dir / f"{mode}.jpg"
    command = [
        sys.executable,
        "-m",
        "jpegoverdrive.cli",
        "optimize",
        str(args.source.resolve()),
        str(args.baseline.resolve()),
        str(output.resolve()),
        "--steps",
        str(args.steps),
        "--snapshot-every",
        str(args.snapshot_every),
        "--lr",
        str(args.lr),
        "--lambda-rd",
        str(args.lambda_rd),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--net",
        "alex",
        "vgg",
    ]
    if mode == "adaptive":
        command += [
            "--adaptive-lpips",
            "--metric-check-every",
            str(args.metric_check_every),
            "--metric-switch-threshold",
            str(args.metric_switch_threshold),
        ]

    print(f"\nRunning {mode} ...", flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    wall_time = time.perf_counter() - started
    history_path = output.parent / f"{output.stem}_snapshots" / "history.csv"
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No history rows in {history_path}")
    required = ("step", "bytes", "bpp", "psnr", "lpips", "lpips_vgg", "elapsed_seconds")
    missing = [field for field in required if field not in rows[0]]
    if missing:
        raise RuntimeError(f"Missing history columns in {history_path}: {missing}")
    history = [{key: float(row[key]) for key in required} for row in rows]
    return wall_time, history


def summarize(
    results: dict[str, tuple[float, list[dict[str, float]]]],
    args: argparse.Namespace,
) -> None:
    """Show wall-time speedup and snapshot-by-snapshot RD differences."""
    joint_wall, joint = results["joint"]
    adaptive_wall, adaptive = results["adaptive"]
    joint_by_step = {int(row["step"]): row for row in joint}
    adaptive_by_step = {int(row["step"]): row for row in adaptive}
    if (
        joint_by_step.keys() != adaptive_by_step.keys()
        or len(joint_by_step) != len(joint)
        or len(adaptive_by_step) != len(adaptive)
    ):
        raise RuntimeError("Snapshot steps differ between the two runs")

    baseline_alex = max(joint[0]["lpips"], 1e-3)
    baseline_vgg = max(joint[0]["lpips_vgg"], 1e-3)

    def worst(row: dict[str, float]) -> float:
        return max(row["lpips"] / baseline_alex, row["lpips_vgg"] / baseline_vgg)

    print("\nJPEG Overdrive — adaptive LPIPS benchmark")
    print(f"Wall time: joint={joint_wall:.2f}s, adaptive={adaptive_wall:.2f}s")
    print(f"Wall-time speedup: {joint_wall / adaptive_wall:.2f}x")
    print(
        f"Optimizer/snapshot elapsed: joint={joint[-1]['elapsed_seconds']:.2f}s, "
        f"adaptive={adaptive[-1]['elapsed_seconds']:.2f}s"
    )
    print("\nSnapshot differences (adaptive minus joint):")
    print(
        f"{'Step':>6} {'Bytes':>9} {'PSNR dB':>11} {'Alex':>12} {'VGG':>12} "
        f"{'Worst Δ':>12}"
    )
    for step in sorted(joint_by_step):
        j, a = joint_by_step[step], adaptive_by_step[step]
        print(
            f"{step:6d} {int(a['bytes'] - j['bytes']):+9d} "
            f"{a['psnr'] - j['psnr']:+11.5f} "
            f"{a['lpips'] - j['lpips']:+12.6g} "
            f"{a['lpips_vgg'] - j['lpips_vgg']:+12.6g} "
            f"{worst(a) - worst(j):+12.6g}"
        )
    final_step = max(joint_by_step)
    j, a = joint_by_step[final_step], adaptive_by_step[final_step]
    print(
        f"\nFinal: joint {j['bytes']:.0f} B / worst={worst(j):.6f}; "
        f"adaptive {a['bytes']:.0f} B / worst={worst(a):.6f}"
    )
    print(f"Histories: {args.output_dir / 'joint_snapshots/history.csv'}")
    print(f"           {args.output_dir / 'adaptive_snapshots/history.csv'}")
    print(
        "Note: adaptive scheduling changes optimization; RD equality is not required."
    )


def main() -> None:
    """Execute both optimizer variants using identical encoding settings."""
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order = (
        ("joint", "adaptive") if args.order == "joint-first" else ("adaptive", "joint")
    )
    results = {mode: run(mode, args) for mode in order}
    summarize(results, args)


if __name__ == "__main__":
    main()
