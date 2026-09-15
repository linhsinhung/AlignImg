from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.performance_validation import sha256
from tools.stage5_validation import (
    BASELINE_SOURCE_SHA256,
    validate_baseline_report,
    validate_raw_gpu_accumulation_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def cases():
    baseline = {
        "performance": {
            "counters": {},
            "gpu_memory": {"tracked_pool_live_bytes_peak": 900_000},
            "stages": {
                "workflow/final_raw_average": {
                    "wall_seconds": 0.02,
                    "counters": {
                        "h2d_calls": 5,
                        "h2d_bytes": 4096,
                        "d2h_calls": 1,
                        "d2h_bytes": 4096,
                    },
                }
            },
        }
    }
    current = {
        "particle_count": 4,
        "image_shape": [16, 16],
        "class_average_metadata": {
            "accumulation": "cupy_fp64_device",
            "transform_backend": "cuda",
            "component_count": 2,
            "gpu_memory_plan": {"batch_size": 4},
            "gpu_memory_events": [],
        },
        "performance": {
            "counters": {},
            "gpu_memory": {"tracked_pool_live_bytes_peak": 700_000},
            "stages": {
                "workflow/final_raw_average": {
                    "wall_seconds": 0.005,
                    "counters": {
                        "h2d_calls": 10,
                        "h2d_bytes": 5000,
                        "d2h_calls": 1,
                        "d2h_bytes": 2048,
                        "final_raw_gpu_accumulation_batches": 1,
                        "final_raw_gpu_accumulation_particles": 4,
                        "final_raw_aligned_d2h_bytes_avoided": 4096,
                        "final_raw_output_d2h_bytes": 2048,
                    },
                }
            },
        },
    }
    return baseline, current


def test_stage5_contract_accepts_gpu_raw_accumulation_and_k_image_download():
    baseline, current = cases()
    report = validate_raw_gpu_accumulation_contract("case", baseline, current)

    assert report["final_raw_aligned_d2h_bytes_avoided"] == 4096
    assert report["final_raw_output_d2h_bytes"] == 2048
    assert report["final_raw_transfer_bytes_saved"] == 1144
    assert report["tracked_pool_live_bytes_saved"] == 200_000
    assert report["d2h_bytes_saved"] == 2048


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda old, new: new["class_average_metadata"].update(
                accumulation=None
            ),
            "GPU accumulation",
        ),
        (
            lambda old, new: new["class_average_metadata"].update(
                transform_backend="cupy"
            ),
            "native CUDA",
        ),
        (
            lambda old, new: new["class_average_metadata"].update(
                gpu_memory_plan=None
            ),
            "VRAM plan",
        ),
        (
            lambda old, new: new["class_average_metadata"][
                "gpu_memory_events"
            ].append(
                {"event": "oom_retry"}
            ),
            "unexpected final raw OOM",
        ),
        (
            lambda old, new: new["performance"]["stages"][
                "workflow/final_raw_average"
            ]["counters"].update(
                final_raw_gpu_accumulation_particles=3
            ),
            "particle accounting",
        ),
        (
            lambda old, new: new["performance"]["stages"][
                "workflow/final_raw_average"
            ]["counters"].update(
                final_raw_aligned_d2h_bytes_avoided=1
            ),
            "aligned-stack bytes",
        ),
        (
            lambda old, new: new["performance"]["stages"][
                "workflow/final_raw_average"
            ]["counters"].update(
                d2h_calls=2
            ),
            "reduced to K images",
        ),
        (
            lambda old, new: new["performance"]["stages"][
                "workflow/final_raw_average"
            ]["counters"].update(
                h2d_bytes=7000
            ),
            "total final raw transfer",
        ),
    ],
)
def test_stage5_contract_rejects_failed_acceptance_conditions(mutation, message):
    baseline, current = cases()
    mutation(baseline, current)
    with pytest.raises(AssertionError, match=message):
        validate_raw_gpu_accumulation_contract("case", baseline, current)


def test_stage5_baseline_report_requires_matching_result(tmp_path: Path):
    report_path = tmp_path / "baseline.json"
    result_path = tmp_path / "baseline.global.result.npz"
    np.savez_compressed(result_path, value=np.arange(3))
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "source": {"source_sha256": BASELINE_SOURCE_SHA256},
                "environment": {"alignimg_module_version": "2.1.0.dev9"},
                "native_build": {"version": "2.1.0.dev9"},
                "cases": {
                    "global": {
                        "status": "passed",
                        "result_sha256": sha256(result_path),
                    }
                },
            }
        )
    )

    parsed = validate_baseline_report(report_path, ("global",))
    assert parsed["source"]["source_sha256"] == BASELINE_SOURCE_SHA256


def test_stage5_script_can_import_tools_when_run_directly(tmp_path: Path):
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/stage5_validation.py"),
            "--baseline-report",
            str(tmp_path / "missing.json"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "ModuleNotFoundError" not in completed.stderr
    assert json.loads(output.read_text())["status"] == "failed"
