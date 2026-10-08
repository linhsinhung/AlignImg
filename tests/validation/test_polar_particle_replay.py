from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from alignimg._fourier import prepare_stack
from tools import polar_particle_replay as replay


def config():
    return replace(
        replay.mirror.ai.AlignmentConfig.preset("fast3"), angle_samples=64
    ).normalized(workflow="global")


def test_safe_boundary_drops_a_column_after_tiny_center_increase():
    cfg = config()
    inside = replay.grid_summary(
        np.array([[4, np.nextafter(np.float32(4), np.float32(0))]]), 100, cfg
    )
    outside = replay.grid_summary(
        np.array([[4, np.nextafter(np.float32(4), np.float32(5))]]), 100, cfg
    )
    assert inside["radius"] == 41
    assert inside["legal_count"] == 81 and inside["boundary_rejected_count"] == 0
    assert outside["legal_count"] == 72 and outside["boundary_rejected_count"] == 9
    assert all(x > 8 for _, x in outside["excluded_centers"])
    assert inside["incoming_center_fp32_bits"] != outside["incoming_center_fp32_bits"]


def test_peak_summary_uses_score_then_original_flat_id():
    curves = np.array([[0, 2, 0, 0], [0, 0, 2, 0], [0, 1, 0, 0]], dtype=float)
    peaks = replay.cpu_peaks(curves)
    summary = replay.peak_summary(peaks, [(0, 0), (1, 0), (2, 0)], 0.08, 4)
    assert [v["center_index"] for v in summary["top_two"]] == [0, 1]
    assert summary["objective_margin"] == 0
    assert summary["top_two"][0]["score"] == 2


def test_peak_score_preserves_cpu_python_float_arithmetic():
    curve = np.array([[0, 0.73, 1.02, 0.88, 0.1]], np.float32)
    _, offsets, scores, _ = replay.cpu_peaks(curve)
    expected = float(curve[0, 2]) - 0.25 * float(curve[0, 1] - curve[0, 3]) * offsets[0]
    assert scores.dtype == np.float64 and scores[0] == expected


def test_cpu_capture_leaves_candidate_computation_unchanged():
    images, refs, _ = replay.mirror.mirror_fixture()
    cfg = replay.mirror.mirror_config(False)
    particles, references = prepare_stack(images[:1], cfg), prepare_stack(refs[:1], cfg)
    center = np.zeros((1, 2))
    expected = replay.cpu.infer_polar_hard_candidates_cpu(
        particles,
        references,
        cfg,
        np.ones((1, 1), np.float32),
        0.08,
        None,
        translation_centers=center,
    )
    original = replay.cpu.angular_correlation
    result, curves = replay.cpu_candidates(particles, references, cfg, 0.08, center)
    assert replay.cpu.angular_correlation is original
    for key in expected:
        np.testing.assert_array_equal(expected[key], result[key])
    grid = replay.grid_summary(center, 32, cfg)["legal_centers"]
    peak = replay.peak_summary(replay.cpu_peaks(curves), grid, 0.08, cfg.angle_samples)[
        "top_two"
    ][0]
    assert abs(peak["angle_deg"] - float(result["angle_deg"][0, 0])) < 1e-3
    assert abs(peak["score"] - float(result["score"][0, 0])) < 2e-5


def test_gpu_observers_restore_on_failure(monkeypatch):
    from alignimg_gpu import backend

    names = ("_polar_sample_cuda", "_normalize_polar_rings_gpu", "_polar_peak_cuda")
    originals = {name: getattr(backend, name) for name in names}
    monkeypatch.setattr(backend, "_cupy", lambda: np)

    def fail(*args, **kwargs):
        assert all(
            getattr(backend, name) is not value for name, value in originals.items()
        )
        raise RuntimeError("injected GPU failure")

    monkeypatch.setattr(backend, "_gpu_candidate_inference", fail)
    with pytest.raises(RuntimeError, match="injected"):
        replay.gpu_candidates(
            None, None, config(), 0.08, np.zeros((1, 2)), None, "cuda", {}
        )
    assert all(getattr(backend, name) is value for name, value in originals.items())


def saved_endpoint():
    values = {
        name: np.zeros((1, 1), np.float32)
        for name in (
            "angle_deg",
            "shift_y_px",
            "shift_x_px",
            "score",
            "reference_index",
            "mirror",
            "_polar_raw_shift_y_px",
            "_polar_raw_shift_x_px",
        )
    }
    saved = {
        "candidate_" + name: np.zeros((3050, 1), np.float32)
        for name in (
            "angle_deg",
            "shift_y_px",
            "shift_x_px",
            "score",
            "reference_index",
            "mirror",
        )
    }
    saved.update(
        {"polar_raw_shift_y_px": np.zeros(3050), "polar_raw_shift_x_px": np.zeros(3050)}
    )
    return saved, values


@pytest.mark.parametrize(
    "field,error,passes",
    [
        ("angle_deg", 0.0005, True),
        ("angle_deg", 0.002, False),
        ("shift_x_px", 2e-5, False),
        ("score", 1e-5, True),
    ],
)
def test_endpoint_uses_existing_candidate_thresholds(field, error, passes):
    saved, values = saved_endpoint()
    values[field][0, 0] = error
    assert (
        replay.endpoint(saved, values)["anchored_within_existing_tolerance"] is passes
    )


@pytest.mark.parametrize(
    "status,code", [("completed", 0), ("inconclusive", 2), ("failed", 1)]
)
def test_diagnostic_status_does_not_pass_t6(tmp_path, monkeypatch, status, code):
    monkeypatch.setattr(
        replay,
        "run",
        lambda engine, source, output, report: report.update(status=status),
    )
    output = tmp_path / "replay.json"
    assert replay.main(["--output", str(output)]) == code
    report = json.loads(output.read_text())
    assert report["diagnostic_only"] and report["status"] == status


def test_existing_report_is_not_overwritten(tmp_path):
    output = tmp_path / "replay.json"
    output.write_text("frozen")
    with pytest.raises(FileExistsError):
        replay.main(["--output", str(output)])
    assert output.read_text() == "frozen"


def test_missing_gpu_records_failure(tmp_path, monkeypatch):
    def fail(engine):
        raise RuntimeError("native CUDA missing")

    monkeypatch.setattr(replay.integration, "runtime_info", fail)
    output = tmp_path / "replay.json"
    assert replay.main(["--output", str(output)]) == 1
    assert "native CUDA missing" in json.loads(output.read_text())["error"]


@pytest.mark.parametrize(
    "sample_errors,endpoint_error,status",
    [(0, 0, "completed"), (1, 0, "failed"), (0, 0.01, "inconclusive")],
)
def test_gpu_driver_and_independent_cpu_keep_separate_centers(
    tmp_path, monkeypatch, sample_errors, endpoint_error, status
):
    cfg = config()
    saved, values = saved_endpoint()
    saved["candidate_angle_deg"][replay.INDEX, 0] = endpoint_error
    # These mocked candidates isolate controller propagation from GPU numerics.
    values.update(
        {
            "_polar_center_y_px": np.array([[4]], np.float32),
            "_polar_center_x_px": np.array([[4]], np.float32),
        }
    )
    saved["reference_history"] = np.zeros((4, 1, 100, 100), np.float32)
    monkeypatch.setattr(replay.integration, "runtime_info", lambda engine: {})
    monkeypatch.setattr(
        replay,
        "load_inputs",
        lambda path: (
            np.zeros((1, 100, 100), np.float32),
            cfg,
            {v: saved for v in ("current", "legacy")},
            {},
        ),
    )
    monkeypatch.setattr(
        replay,
        "cpu_candidates",
        lambda *args: (
            deepcopy(values),
            np.tile(
                np.arange(64)[None],
                (len(replay.grid_summary(args[-1], 100, cfg)["legal_centers"]), 1),
            ),
        ),
    )
    calls = []

    def gpu(*args):
        calls.append(args[4].copy())
        args[-1]["samplers"] = [{"different_values": sample_errors}]
        return deepcopy(values), replay.cpu_peaks(args[5])

    monkeypatch.setattr(replay, "gpu_candidates", gpu)
    monkeypatch.setattr(replay, "comparison", lambda *args: {"passed": True})
    report = {}
    replay.run("cuda", tmp_path / "unused", tmp_path / "replay.json", report)
    assert len(calls) == 6
    # Current retains the input candidate center; legacy reconstructs (0,0) from pose.
    np.testing.assert_array_equal(calls[1], [[4, 4]])
    np.testing.assert_array_equal(calls[4], [[0, 0]])
    assert report["status"] == status
