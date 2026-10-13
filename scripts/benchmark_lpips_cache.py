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

"""Benchmark LPIPS reference caching against uncached JPEG optimization."""

import argparse
import csv
import math
import subprocess
import sys
import time
from pathlib import Path

FLOAT_FIELDS = (
    "bpp",
    "psnr",
    "lpips",
    "lpips_vgg",
    "loss",
    "distortion_norm",
    "rate_norm",
    "rate_proxy",
    "alex_norm",
    "vgg_norm",
)


def read_history(path: Path) -> list[dict[str, str]]:
    """Read snapshot measurements exported by the JPEG Overdrive CLI."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_case(
    args: argparse.Namespace, label: str, use_cache: bool
) -> tuple[list[dict[str, str]], float]:
    """Run one CLI configuration and return its snapshots and wall time."""
    output = args.output_dir / f"{label}.jpg"
    command = [
        sys.executable,
        "-m",
        "jpegoverdrive.cli",
        "optimize",
        str(args.source),
        str(args.baseline),
        str(output),
        "--steps",
        str(args.steps),
        "--lr",
        str(args.lr),
        "--lambda-rd",
        str(args.lambda_rd),
        "--snapshot-every",
        str(args.snapshot_every),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--net",
        *args.net,
    ]
    if not use_cache:
        command.append("--no-cache-lpips")

    print(f"\nRunning {label} ...", flush=True)
    start = time.perf_counter()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    duration = time.perf_counter() - start
    history_path = output.parent / f"{output.stem}_snapshots" / "history.csv"
    return read_history(history_path), duration


def equivalent(a: float, b: float, atol: float, rtol: float) -> bool:
    """Compare finite values with tolerance, including matching infinities."""
    if a == b:
        return True
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return math.isclose(a, b, abs_tol=atol, rel_tol=rtol)


def compare_histories(
    cached: list[dict[str, str]],
    uncached: list[dict[str, str]],
    atol: float,
    rtol: float,
) -> bool:
    """Compare all matching snapshots and print worst metric differences."""
    if [row["step"] for row in cached] != [row["step"] for row in uncached]:
        print("FAIL: snapshot step lists differ")
        return False

    fields = ("bytes",) + tuple(
        key for key in FLOAT_FIELDS if key in cached[0] and key in uncached[0]
    )
    passed = True
    print("\nRD equivalence (all snapshots):")
    for field in fields:
        differences = []
        ok = True
        for c, u in zip(cached, uncached):
            x, y = float(c[field]), float(u[field])
            if not equivalent(x, y, atol, rtol):
                ok = False
            differences.append(
                abs(x - y) if math.isfinite(x) and math.isfinite(y) else 0.0
            )
        if field == "bytes":
            ok = all(c[field] == u[field] for c, u in zip(cached, uncached))
        passed &= ok
        print(
            f"  {field:<18} max Δ = {max(differences):.6g}  {'PASS' if ok else 'FAIL'}"
        )
    return passed


def main() -> int:
    """Run paired optimizations and exit nonzero if RD results diverge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/lpips_cache_benchmark")
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda-rd", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--net", nargs="+", choices=("alex", "vgg"), default=["alex", "vgg"]
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    uncached, t_uncached = run_case(args, "uncached", False)
    cached, t_cached = run_case(args, "cached", True)
    print(f"\nWall time: uncached={t_uncached:.2f}s, cached={t_cached:.2f}s")
    print(f"Speedup: {t_uncached / t_cached:.2f}x")
    passed = compare_histories(cached, uncached, args.atol, args.rtol)
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    print(f"Histories: {args.output_dir / 'cached_snapshots/history.csv'}")
    print(f"           {args.output_dir / 'uncached_snapshots/history.csv'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
