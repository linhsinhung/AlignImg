from __future__ import annotations

import json
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from tools import stage7_validation as stage7


def write_result_bundle(base: Path, seed: int) -> dict[str, str]:
    result = base / f"seed-{seed}.result.npz"
    references = base / f"seed-{seed}.references.mrcs"
    gui_report = base / f"seed-{seed}.report.json"
    responsibilities = np.full((5000, 10), 0.1, dtype=np.float32)
    np.savez_compressed(
        result,
        alignimg_version=np.asarray(stage7.EXPECTED_VERSION),
        angle_deg=np.zeros(5000, dtype=np.float32),
        shift_y_px=np.zeros(5000, dtype=np.float32),
        shift_x_px=np.zeros(5000, dtype=np.float32),
        assignments=np.zeros(5000, dtype=np.int32),
        responsibilities=responsibilities,
        inlier_weights=np.ones(5000, dtype=np.float32),
    )
    with mrcfile.new(references, overwrite=True) as mrc:
        mrc.set_data(np.zeros((10, 4, 4), dtype=np.float32))
    gui_report.write_text("{}\n", encoding="utf-8")
    return {
        "result": str(result),
        "references": str(references),
        "gui_report": str(gui_report),
    }


def formal_report(tmp_path: Path) -> dict:
    config = {
        "max_iterations": 20,
        "temperature_anneal_iterations": 10,
        "angle_samples": 128,
        "translation_range": 4.0,
        "candidate_scoring": "fourier",
        "score_model": "fourier_ncc",
        "reference_update": "fourier",
        "batch_size": 512,
        "memory_fraction": 0.8,
    }
    runs = {}
    for seed in stage7.FORMAL_SPEC["seeds"]:
        runs[str(seed)] = {
            "artifacts": write_result_bundle(tmp_path, seed),
            "metadata": {
                "backend": "cuda",
                "engine": "alignimg-soft-fourier-cuda",
                "candidate_scoring": "fourier",
                "score_model": "fourier_ncc",
                "reference_update": "fourier",
                "gpu_memory_plans": [{"batch_size": 512}],
            },
        }
    return {
        "status": "completed",
        "parameters": {
            "backend": "cuda",
            "component_count": 10,
            "seeds": [0, 1, 2],
            "config": config,
        },
        "runs": runs,
    }


def test_dev10_compute_sources_remain_frozen():
    result = stage7.verify_dev10_compute_freeze()
    assert result["baseline_version"] == "2.1.0.dev10"
    assert len(result["files"]) == len(stage7.FROZEN_COMPUTE_FILES)
    assert all(item["matches_dev10"] for item in result["files"].values())


def test_dev10_compute_freeze_rejects_changed_hash(tmp_path: Path):
    original = json.loads(stage7.DEFAULT_DEV10_MANIFEST.read_text(encoding="utf-8"))
    original["files"][stage7.FROZEN_COMPUTE_FILES[0]] = "0" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(RuntimeError, match="frozen dev10 compute source changed"):
        stage7.verify_dev10_compute_freeze(manifest)


def test_formal_report_contract_and_artifacts(tmp_path: Path):
    checks = stage7.validate_formal_report(
        formal_report(tmp_path),
        backend="cuda",
        batch_size=512,
        memory_fraction=0.8,
    )
    assert set(checks) == {"0", "1", "2"}
    assert checks["0"]["planned_batch_sizes"] == [512]
    assert checks["0"]["responsibility_sum_max_error"] < 1e-6


def test_formal_report_rejects_reduced_batch(tmp_path: Path):
    report = formal_report(tmp_path)
    report["runs"]["1"]["metadata"]["gpu_memory_plans"][0]["batch_size"] = 256

    with pytest.raises(RuntimeError, match="did not retain batch 512"):
        stage7.validate_formal_report(
            report,
            backend="cuda",
            batch_size=512,
            memory_fraction=0.8,
        )


def test_stage7_cli_defaults_are_the_frozen_release_spec():
    args = stage7.parse_args([])
    assert args.backend == "cuda"
    assert args.batch_size == 512
    assert args.memory_fraction == 0.8
    assert stage7.FORMAL_SPEC["seeds"] == [0, 1, 2]
    assert stage7.FORMAL_SPEC["iterations"] == 20
