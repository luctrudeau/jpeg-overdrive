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

"""Tests for the JPEG Overdrive command-line interface."""

import argparse
import csv
import json
from pathlib import Path

import jpegio as jio
import numpy as np
import pytest
import torch
from PIL import Image

from jpegoverdrive import __version__
from jpegoverdrive.cli import (
    build_parser,
    compare_jpeg_states,
    main,
    print_initialization_validation,
    print_optimization_summary,
    run_optimize,
    save_history,
    save_summary,
)


@pytest.fixture
def jpeg_template(tmp_path):
    """Create a small deterministic 4:4:4 JPEG."""
    rng = np.random.default_rng(42)

    image = rng.integers(
        0,
        256,
        size=(32, 32, 3),
        dtype=np.uint8,
    )

    path = tmp_path / "baseline.jpg"

    Image.fromarray(image).save(
        path,
        format="JPEG",
        quality=90,
        subsampling=0,
    )

    return path


def test_help(capsys):
    """Verify top-level help."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0

    output = capsys.readouterr().out

    assert "JPEG-compliant" in output
    assert "optimize" in output


def test_version(capsys):
    """Verify the CLI reports the package version."""
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_optimize_help(capsys):
    """Verify optimize subcommand help."""
    with pytest.raises(SystemExit) as exc:
        main(["optimize", "--help"])

    assert exc.value.code == 0

    output = capsys.readouterr().out

    assert "--steps" in output
    assert "--lambda-rd" in output
    assert "--snapshot-every" in output
    assert "--device" in output


def test_optimize_arguments():
    """Verify argument parsing."""
    parser = build_parser()

    args = parser.parse_args(
        [
            "optimize",
            "source.png",
            "baseline.jpg",
            "output.jpg",
            "--steps",
            "100",
            "--lr",
            "0.002",
            "--lambda-rd",
            "0.01",
        ]
    )

    assert args.command == "optimize"
    assert args.source == Path("source.png")
    assert args.baseline == Path("baseline.jpg")
    assert args.output == Path("output.jpg")
    assert args.steps == 100
    assert args.lr == pytest.approx(0.002)
    assert args.lambda_rd == pytest.approx(0.01)


def test_compare_jpeg_states_identical(jpeg_template):
    """An identical JPEG must have zero representation differences."""
    result = compare_jpeg_states(
        jpeg_template,
        jpeg_template,
    )

    assert len(result["components"]) == 3
    assert result["qtable_changes"] == 0
    assert result["coefficient_changes"] == 0
    assert result["rgb_sample_changes"] == 0
    assert result["max_rgb_error"] == 0
    assert result["size_difference_bytes"] == 0
    assert result["size_difference_percent"] == pytest.approx(0.0)

    print("\nIdentical JPEG: zero differences")


def test_compare_jpeg_states_rewrite(jpeg_template, tmp_path):
    """Check whether a jpegio rewrite preserves the representation."""
    output = tmp_path / "rewritten.jpg"

    jpeg = jio.read(str(jpeg_template))
    jpeg.optimize_coding = True

    jio.write(jpeg, str(output))

    result = compare_jpeg_states(
        jpeg_template,
        output,
    )

    assert result["qtable_changes"] == 0
    assert result["coefficient_changes"] == 0
    assert result["rgb_sample_changes"] == 0
    assert result["max_rgb_error"] == 0

    # File size may change even when the representation is identical.
    assert result["size_difference_bytes"] == (
        output.stat().st_size - jpeg_template.stat().st_size
    )

    print(
        f"\nJPEG rewrite: "
        f"coefficients unchanged, "
        f"size delta={result['size_difference_bytes']:+,} bytes"
    )


def test_compare_jpeg_states_changed_coefficient(
    jpeg_template,
    tmp_path,
):
    """Detect a modified quantized DCT coefficient."""
    output = tmp_path / "modified.jpg"

    jpeg = jio.read(str(jpeg_template))

    # Modify the first luminance DC coefficient by one.
    jpeg.coef_arrays[0][0, 0] += 1

    jio.write(jpeg, str(output))

    result = compare_jpeg_states(
        jpeg_template,
        output,
    )

    assert result["coefficient_changes"] == 1
    assert result["qtable_changes"] == 0

    print(f"\nModified JPEG: {result['coefficient_changes']} coefficient changed")


def test_compare_jpeg_states_changed_qtable(
    jpeg_template,
    tmp_path,
):
    """Detect a modified quantization-table entry."""
    output = tmp_path / "modified_qtable.jpg"

    jpeg = jio.read(str(jpeg_template))

    qindex = jpeg.comp_info[0].quant_tbl_no

    original = int(jpeg.quant_tables[qindex][0, 0])
    replacement = original + 1 if original < 255 else original - 1

    jpeg.quant_tables[qindex][0, 0] = replacement

    jio.write(jpeg, str(output))

    result = compare_jpeg_states(
        jpeg_template,
        output,
    )

    assert result["qtable_changes"] >= 1
    assert result["coefficient_changes"] == 0

    print(f"\nModified quantization table: {result['qtable_changes']} entries changed")


def test_print_initialization_validation(
    jpeg_template,
    capsys,
):
    """Verify the initialization validation report."""
    result = compare_jpeg_states(
        jpeg_template,
        jpeg_template,
    )

    print_initialization_validation(result)

    output = capsys.readouterr().out

    assert "Initialization Validation" in output
    assert "Q-table changes" in output
    assert "Coefficient changes" in output
    assert "Different RGB samples" in output
    assert "Maximum RGB error" in output
    assert "Exact JPEG representation round-trip" in output


def test_save_history(tmp_path):
    """Verify optimization history is saved as CSV."""
    history = [
        {
            "step": 0,
            "path": tmp_path / "step_000000.jpg",
            "bpp": 2.5,
            "lpips": 0.02,
        },
        {
            "step": 10,
            "path": tmp_path / "step_000010.jpg",
            "bpp": 2.4,
            "lpips": 0.019,
        },
    ]

    output = tmp_path / "history.csv"

    save_history(history, output)

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["step"] == "0"
    assert rows[1]["step"] == "10"
    assert float(rows[1]["bpp"]) == pytest.approx(2.4)

    print(f"\nHistory CSV: {len(rows)} snapshots")


def test_save_summary(tmp_path):
    """Verify JSON serialization of paths and infinite PSNR."""
    summary = {
        "output": tmp_path / "optimized.jpg",
        "baseline": {
            "bytes": 1000,
            "psnr": float("inf"),
            "lpips": 0.0,
        },
        "initialization": {
            "coefficient_changes": 0,
            "size_difference_bytes": 123,
        },
    }

    output = tmp_path / "summary.json"

    save_summary(summary, output)

    with output.open(encoding="utf-8") as file:
        result = json.load(file)

    assert result["output"] == str(tmp_path / "optimized.jpg")
    assert result["baseline"]["psnr"] is None
    assert result["initialization"]["coefficient_changes"] == 0
    assert result["initialization"]["size_difference_bytes"] == 123


def test_print_optimization_summary(capsys, tmp_path):
    """Verify actual rate-distortion improvements are reported."""
    baseline = {
        "bytes": 1000,
        "bpp": 2.0,
        "psnr": 35.0,
        "lpips": 0.02,
    }

    best = {
        "step": 50,
        "is_baseline": False,
        "bytes": 900,
        "bpp": 1.8,
        "psnr": 35.5,
        "lpips": 0.018,
    }

    print_optimization_summary(
        baseline,
        best,
        tmp_path / "optimized.jpg",
    )

    output = capsys.readouterr().out

    assert "Optimization Summary" in output
    assert "step 50" in output
    assert "+10.00%" in output
    assert "+0.500 dB" in output


def test_print_zero_distortion_summary(capsys, tmp_path):
    """Verify zero baseline distortion is handled."""
    baseline = {
        "bytes": 1000,
        "bpp": 2.0,
        "psnr": float("inf"),
        "lpips": 0.0,
    }

    best = {
        **baseline,
        "step": None,
        "is_baseline": True,
    }

    print_optimization_summary(
        baseline,
        best,
        tmp_path / "output.jpg",
    )

    output = capsys.readouterr().out

    assert "Original baseline retained" in output
    assert "N/A (zero baseline)" in output
    assert "PSNR improvement:   N/A" in output


def test_run_optimize_rejects_baseline_overwrite(jpeg_template):
    """Reject output paths that would overwrite the input JPEG."""
    args = argparse.Namespace(
        baseline=jpeg_template,
        output=jpeg_template,
    )

    with pytest.raises(
        ValueError,
        match="Output must be different",
    ):
        run_optimize(args)


def test_run_optimize_baseline_fallback(
    jpeg_template,
    tmp_path,
    monkeypatch,
):
    """Verify CLI integration without loading pretrained LPIPS."""
    from jpegoverdrive import metrics, optimize

    source = tmp_path / "source.png"

    with Image.open(jpeg_template) as image:
        image.save(source)

    output = tmp_path / "optimized.jpg"

    args = argparse.Namespace(
        source=source,
        baseline=jpeg_template,
        output=output,
        steps=0,
        lr=0.001,
        lambda_rd=0.01,
        distortion_scale=0.00073351,
        snapshot_every=1,
        seed=0,
        device="cpu",
        net="alex",
    )

    class MockMetric(torch.nn.Module):
        """Minimal metric used by the CLI integration test."""

        def forward(self, reference, reconstructed):
            return (
                (reference - reconstructed)
                .square()
                .mean(
                    dim=(1, 2, 3),
                    keepdim=True,
                )
            )

    monkeypatch.setattr(
        metrics,
        "create_lpips_metric",
        lambda device, net: MockMetric(),
    )

    def fake_optimize_jpeg(
        initial_state,
        template_path,
        output_dir,
        reference,
        metric,
        lambda_rd,
        **kwargs,
    ):
        assert kwargs["distortion_scale"] == pytest.approx(0.00073351)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        snapshot = output_dir / "step_000000.jpg"

        shutil = __import__("shutil")
        shutil.copyfile(template_path, snapshot)

        actual = metrics.measure_jpeg(
            snapshot,
            reference,
            metric,
        )

        return [], [
            {
                "step": 0,
                "path": snapshot,
                "bytes": actual["bytes"],
                "bpp": actual["bpp"],
                "psnr": actual["psnr"],
                "lpips": actual["lpips"] + 0.01,
                "loss": 1.0,
                "distortion_norm": 1.0,
                "rate_norm": 1.0,
            }
        ]

    monkeypatch.setattr(
        optimize,
        "optimize_jpeg",
        fake_optimize_jpeg,
    )

    run_optimize(args)

    assert output.exists()
    assert output.read_bytes() == jpeg_template.read_bytes()

    snapshot_dir = tmp_path / "optimized_snapshots"

    history_path = snapshot_dir / "history.csv"
    summary_path = snapshot_dir / "summary.json"

    assert history_path.exists()
    assert summary_path.exists()

    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)

    assert summary["selected"]["is_baseline"] is True
    assert summary["initialization"]["coefficient_changes"] == 0
    assert summary["initialization"]["qtable_changes"] == 0
    assert summary["initialization"]["rgb_sample_changes"] == 0
    assert summary["configuration"]["distortion_scale"] == pytest.approx(0.00073351)

    print("\nCLI integration: baseline fallback and diagnostics verified")


def test_run_optimize_selects_snapshot(
    jpeg_template,
    tmp_path,
    monkeypatch,
):
    """Verify the CLI selects and copies an improved JPEG snapshot."""
    import json
    import shutil

    from jpegoverdrive import metrics, optimize

    source = tmp_path / "source.png"

    with Image.open(jpeg_template) as image:
        image.save(source)

    output = tmp_path / "optimized.jpg"

    args = argparse.Namespace(
        source=source,
        baseline=jpeg_template,
        output=output,
        steps=0,
        lr=0.001,
        lambda_rd=0.01,
        distortion_scale=None,
        snapshot_every=1,
        seed=0,
        device="cpu",
        net="alex",
    )

    class MockMetric(torch.nn.Module):
        def forward(self, reference, reconstructed):
            return (
                (reference - reconstructed).square().mean(dim=(1, 2, 3), keepdim=True)
            )

    monkeypatch.setattr(
        metrics,
        "create_lpips_metric",
        lambda device, net: MockMetric(),
    )

    def fake_optimize_jpeg(
        initial_state,
        template_path,
        output_dir,
        reference,
        metric,
        lambda_rd,
        **kwargs,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        snapshot = output_dir / "step_000000.jpg"

        shutil.copyfile(template_path, snapshot)

        actual = metrics.measure_jpeg(
            snapshot,
            reference,
            metric,
        )

        return [], [
            {
                "step": 0,
                "path": snapshot,
                "bytes": actual["bytes"],
                "bpp": actual["bpp"],
                "psnr": actual["psnr"],
                # Force this snapshot to win selection.
                "lpips": actual["lpips"] - 0.01,
                "loss": 1.0,
                "distortion_norm": 1.0,
                "rate_norm": 1.0,
            }
        ]

    monkeypatch.setattr(
        optimize,
        "optimize_jpeg",
        fake_optimize_jpeg,
    )

    run_optimize(args)

    snapshot = tmp_path / "optimized_snapshots" / "step_000000.jpg"

    assert output.exists()
    assert output.read_bytes() == snapshot.read_bytes()

    summary_path = tmp_path / "optimized_snapshots" / "summary.json"

    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)

    assert summary["selected"]["is_baseline"] is False
    assert summary["selected"]["step"] == 0
