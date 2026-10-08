import json
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from tools import polar_resource_validation as validation


def test_original_baseline_remains_strict():
    report = {}
    with pytest.raises(AssertionError):
        validation.numeric_baseline_checks(
            {"angle_deg": np.array([0.0])}, {"angle_deg": np.array([1.0])}, None, report
        )
    assert not report["legacy_baseline_comparison"]["passed"]
    assert not report["legacy_baseline_comparison"]["accepted_exception"]


def test_corrected_baseline_records_old_failure_without_waiving_new_gate():
    original = {"angle_deg": np.array([0.0])}
    corrected = {"angle_deg": np.array([1.0])}
    report = {}
    validation.numeric_baseline_checks(original, corrected, corrected, report)
    assert report["legacy_baseline_comparison"]["passed"] is False
    assert report["legacy_baseline_comparison"]["accepted_exception"] is True
    assert "corrected_baseline_errors" in report
    with pytest.raises(AssertionError):
        validation.numeric_baseline_checks(
            original, {"angle_deg": np.array([1.002])}, corrected, {}
        )


@pytest.mark.parametrize(
    "tamper", [None, "manifest", "result", "acceptance", "missing"]
)
def test_corrected_baseline_is_pinned_and_requires_acceptance(
    tmp_path, monkeypatch, tamper
):
    manifest_path = tmp_path / (validation.CORRECTED_BASELINE + ".json")
    manifest_path.parent.mkdir(parents=True)
    result_path = manifest_path.with_suffix(".npz")
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "compatibility_exception": {
                    "id": "zero-shift-noop-local3050-legacy-baseline"
                }
            }
        )
    )
    np.savez(result_path, angle_deg=np.array([1.0]), extra=np.zeros(1))
    result_hash = validation.mirror.sha256(result_path)
    manifest_path.write_text(
        json.dumps(
            {
                "config": {},
                "inputs": {},
                "comparison_fields": ["angle_deg"],
                "result_npz_sha256": result_hash,
                "acceptance_path": "acceptance.json",
                "acceptance_sha256": validation.mirror.sha256(acceptance_path),
            }
        )
    )
    monkeypatch.setattr(
        validation, "CORRECTED_MANIFEST_SHA256", validation.mirror.sha256(manifest_path)
    )
    monkeypatch.setattr(validation, "CORRECTED_RESULT_SHA256", result_hash)
    if tamper:
        path = {
            "manifest": manifest_path,
            "result": result_path,
            "acceptance": acceptance_path,
            "missing": manifest_path,
        }[tamper]
        if tamper == "missing":
            path.unlink()
        else:
            path.write_bytes(b"changed")
        with pytest.raises((RuntimeError, FileNotFoundError)):
            validation.corrected_baseline(tmp_path, {}, {}, ["angle_deg"])
    else:
        values, _ = validation.corrected_baseline(tmp_path, {}, {}, ["angle_deg"])
        assert set(values) == {"angle_deg"}
        np.testing.assert_array_equal(values["angle_deg"], [1.0])


@pytest.mark.parametrize("suite", ["conformance", "resources"])
def test_corrected_baseline_cannot_waive_other_suites(tmp_path, suite):
    with pytest.raises(SystemExit):
        validation.main(
            [
                "--suite",
                suite,
                "--backend",
                "cuda",
                "--numeric-baseline",
                "corrected",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_cli_passes_explicit_baseline_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "runtime_info", lambda *args: {})
    monkeypatch.setattr(validation.mirror, "environment_info", lambda *args: {})

    def representative(*args, report, numeric_baseline):
        assert numeric_baseline == "corrected"
        report["selected"] = numeric_baseline

    monkeypatch.setattr(validation, "validate_representative", representative)
    output = tmp_path / "report.json"
    assert (
        validation.main(
            [
                "--suite",
                "representative",
                "--backend",
                "cuda",
                "--numeric-baseline",
                "corrected",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["checks"]["selected"] == "corrected"


def test_corrected_numerical_pass_does_not_waive_speed_limit():
    values = {"angle_deg": np.array([1.0])}
    validation.numeric_baseline_checks(
        {"angle_deg": np.array([0.0])}, values, values, {}
    )
    old = {"median_seconds": 10.0, "environment": {"hostname": "same", "gpu": "3090"}}
    assert not validation.speed_gate(
        old, 10.5001, "cuda", {"hostname": "same", "gpu": {"name": "3090"}}
    )["passed"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--suite", "representative", "--trace-memory"],
        ["--suite", "conformance", "--trace-memory"],
        ["--suite", "resources", "--no-timing-events"],
    ],
)
def test_memory_diagnostic_flags_cannot_change_other_suites(tmp_path, arguments):
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as error:
        validation.main(arguments + ["--backend", "cuda", "--output", str(output)])
    assert error.value.code != 0 and not output.exists()


def test_events_disabled_is_a_separate_diagnostic_not_normal_acceptance(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(validation, "runtime_info", lambda *args: {})
    monkeypatch.setattr(validation.mirror, "environment_info", lambda *args: {})
    monkeypatch.setattr(validation, "planning_checks", lambda: {})
    monkeypatch.setattr(
        validation.streaming,
        "device_context_snapshot",
        lambda: {"status": "unavailable", "error": "injected"},
    )

    def run(engine, *, trace_memory, timing_events, report, **kwargs):
        assert trace_memory and not timing_events
        report["checked"] = True

    monkeypatch.setattr(validation.streaming, "validate_streaming", run)
    output = tmp_path / "diagnostic.json"
    assert (
        validation.main(
            [
                "--suite",
                "resources",
                "--backend",
                "cuda",
                "--trace-memory",
                "--no-timing-events",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["status"] == "passed" and report["diagnostic_only"]
    assert report["checks"]["storage"]["checked"]
    assert not report["memory_diagnostics"]["timing_events_enabled"]
    assert report["memory_diagnostics"]["after"]["status"] == "unavailable"


@pytest.mark.parametrize("suite", ["conformance", "resources", "representative"])
def test_existing_report_is_not_overwritten(tmp_path, suite):
    path = tmp_path / "frozen.json"
    path.write_text('{"frozen":true}\n')
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validation.main(["--suite", suite, "--backend", "cuda", "--output", str(path)])
    assert json.loads(path.read_text()) == {"frozen": True}


def test_missing_gpu_is_failed_report_not_skip(tmp_path, monkeypatch):
    def missing(*args):
        raise RuntimeError("native GPU unavailable")

    monkeypatch.setattr(validation, "runtime_info", missing)
    path = tmp_path / "missing.json"
    assert (
        validation.main(
            ["--suite", "conformance", "--backend", "cuda", "--output", str(path)]
        )
        == 1
    )
    report = json.loads(path.read_text())
    assert report["status"] == "failed" and "GPU unavailable" in report["error"]


@pytest.mark.parametrize("budget", ["0", "-1", "nan", "inf"])
def test_budget_must_be_positive_and_finite(tmp_path, budget):
    with pytest.raises(SystemExit) as error:
        validation.main(
            [
                "--suite",
                "resources",
                "--backend",
                "cuda",
                "--polar-budget-mib",
                budget,
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
    assert error.value.code != 0


def test_representative_requires_frozen_batch(tmp_path):
    with pytest.raises(SystemExit) as error:
        validation.main(
            [
                "--suite",
                "representative",
                "--backend",
                "cuda",
                "--batch-size",
                "256",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
    assert error.value.code != 0


def test_budget_override_only_applies_to_resource_suite(tmp_path):
    with pytest.raises(SystemExit):
        validation.main(
            [
                "--suite",
                "representative",
                "--backend",
                "cuda",
                "--polar-budget-mib",
                "3",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_large_n_planning_uses_sizes_only_and_streams(monkeypatch):
    from alignimg_gpu import backend

    monkeypatch.setattr(
        backend, "_asdevice", lambda *args, **kwargs: pytest.fail("unexpected upload")
    )
    plans = validation.planning_checks()
    assert set(plans) == {"1000", "105000", "140000"}
    for plan in plans.values():
        assert plan["fits_minimum"] and plan["particle_storage_policy"] == "streaming"
        assert plan["estimated_peak_bytes"] <= plan["budget_bytes"]
        assert "resident_particles_fp32" not in plan["fixed_components"]
    assert len({plan["batch_size"] for plan in plans.values()}) == 1
    assert plans["1000"]["fixed_components"] == plans["140000"]["fixed_components"]
    assert plans["1000"]["item_components"] == plans["140000"]["item_components"]


def test_missing_frozen_input_does_not_generate_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(
        validation.baseline, "FROZEN_INPUTS", {"synthetic": ("missing.npz", "frozen")}
    )
    monkeypatch.setattr(
        validation.fast,
        "freeze_inputs",
        lambda *args: pytest.fail("resampling frozen input"),
    )
    with pytest.raises(FileNotFoundError):
        validation.preflight(tmp_path)
    assert not (tmp_path / "missing.npz").exists()


def test_modified_frozen_input_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "input.npz").write_bytes(b"changed")
    monkeypatch.setattr(
        validation.baseline, "FROZEN_INPUTS", {"synthetic": ("input.npz", "frozen")}
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validation.preflight(tmp_path)


@pytest.mark.parametrize("ratio,passed", [(1.0, True), (1.05, True), (1.05001, False)])
def test_cuda_speed_gate_keeps_five_percent_limit(ratio, passed):
    old = {"median_seconds": 10.0, "environment": {"hostname": "same", "gpu": "3090"}}
    current_env = {"hostname": "same", "gpu": {"name": "3090"}}
    gate = validation.speed_gate(old, 10.0 * ratio, "cuda", current_env)
    assert gate["applicable"] and gate["passed"] is passed
    assert gate["limit_seconds"] == 10.5


def test_different_server_is_not_a_valid_performance_comparison():
    old = {"median_seconds": 10.0, "environment": {"hostname": "old", "gpu": "3090"}}
    gate = validation.speed_gate(
        old, 1.0, "cuda", {"hostname": "new", "gpu": {"name": "3090"}}
    )
    assert gate["applicable"] and not gate["passed"] and not gate["same_server"]


def test_cupy_is_not_compared_as_cuda_speed():
    old = {"median_seconds": 10.0, "environment": {"hostname": "same", "gpu": "3090"}}
    gate = validation.speed_gate(
        old, 100.0, "cupy", {"hostname": "same", "gpu": {"name": "3090"}}
    )
    assert not gate["applicable"] and gate["passed"] is None


def resource_entry():
    entry = {
        "backend_metadata": {
            "backend": "cuda",
            "polar_sampler_backend": "native_cuda",
            "polar_peak_backend": "native_cuda",
            "polar_full_correlation_map_d2h": False,
        },
        "deterministic_exact_match": True,
        "gpu_workspace": {"closed": True, "budget_bytes": 1024},
        "gpu_memory_plans": [
            {
                "stage": "polar_hard_candidate_inference",
                "particle_storage_policy": "streaming",
                "budget_bytes": 1024,
                "fixed_bytes": 64,
                "bytes_per_item": 64,
                "batch_size": 7,
                "fits_minimum": True,
                "estimated_peak_bytes": 512,
            }
        ],
        "performance": {
            "counters": {"native_polar_sampling_calls": 1},
            "gpu_memory": {
                "tracked_pool_live_bytes_peak": 512,
                "entry": {"device_used_bytes": 1024},
                "sampled_device_used_bytes_peak": 1536,
            },
        },
    }
    execution = {
        "phase": "profiled",
        "index": 0,
        "result_hashes": {"combined": "same"},
        "gpu_workspace": deepcopy(entry["gpu_workspace"]),
        "gpu_memory_plans": deepcopy(entry["gpu_memory_plans"]),
    }
    entry["execution_records"] = [execution]
    entry["execution_records"].extend(
        {**deepcopy(execution), "phase": "unprofiled", "index": index}
        for index in range(3)
    )
    return entry


@pytest.mark.parametrize(
    "change,same",
    [
        ("budget", True),
        ("polar_batch", False),
        ("storage", False),
        ("mstep_batch", False),
        ("cache_policy", False),
        ("raw_average_batch", False),
    ],
)
def test_repeat_schedule_uses_actual_allocation_not_requested_batch(change, same):
    first = {
        "phase": "unprofiled",
        "index": 0,
        "class_average_batch_size": 512,
        "gpu_memory_plans": [
            {
                "stage": "polar_hard_candidate_inference",
                "batch_size": 90,
                "requested_particle_batch": 512,
                "particle_storage_policy": "streaming",
                "budget_bytes": 1000,
                "free_bytes": 2000,
            },
            {"stage": "fourier_reference_update", "batch_size": 512},
            {"stage": "particle_fourier_cache", "policy": "device"},
        ],
    }
    second = deepcopy(first)
    second["index"] = 1
    if change == "budget":
        second["gpu_memory_plans"][0].update(budget_bytes=1100, free_bytes=2500)
    elif change == "polar_batch":
        second["gpu_memory_plans"][0]["batch_size"] = 89
    elif change == "storage":
        second["gpu_memory_plans"][0]["particle_storage_policy"] = "resident"
    elif change == "mstep_batch":
        second["gpu_memory_plans"][1]["batch_size"] = 256
    elif change == "cache_policy":
        second["gpu_memory_plans"][2]["policy"] = "host_streamed"
    else:
        second["class_average_batch_size"] = 256
    observed = validation.repeat_execution_diagnostics([first, second])
    assert observed["same_recorded_allocation_schedule"] is same


@pytest.mark.parametrize(
    "records", [[], [{"phase": "unprofiled", "gpu_memory_plans": []}]]
)
def test_missing_repeat_plans_are_not_same_schedule_evidence(records):
    assert (
        validation.repeat_execution_diagnostics(records)[
            "same_recorded_allocation_schedule"
        ]
        is None
    )


def test_profile_memory_is_checked_against_its_own_budget():
    entry = resource_entry()
    entry["execution_records"][0]["gpu_workspace"]["budget_bytes"] = 400
    with pytest.raises(AssertionError, match="measured pool peak"):
        validation.resource_checks(entry, "cuda")
    entry["gpu_workspace"]["budget_bytes"] = 400
    entry["execution_records"][0]["gpu_workspace"]["budget_bytes"] = 1024
    assert validation.resource_checks(entry, "cuda")["budget_bytes"] == 1024


def test_cross_schedule_nonexact_result_still_fails_gate():
    entry = resource_entry()
    entry["deterministic_exact_match"] = False
    for index, batch_size in enumerate((90, 89, 88)):
        entry["execution_records"].append(
            {
                "phase": "unprofiled",
                "index": index,
                "gpu_memory_plans": [
                    {
                        "stage": "polar_hard_candidate_inference",
                        "batch_size": batch_size,
                    }
                ],
            }
        )
    with pytest.raises(AssertionError, match="same-backend repeats are not exact"):
        validation.resource_checks(entry, "cuda")
    assert (
        entry["repeat_execution_diagnostics"]["same_recorded_allocation_schedule"]
        is False
    )


@pytest.mark.parametrize("insufficient", [False, True])
def test_controlled_batch_respects_real_budget_and_restores_planner(
    monkeypatch, insufficient
):
    from alignimg_gpu import backend
    from alignimg_gpu._polar_memory import plan_polar_storage

    config = replace(
        validation.mirror.ai.AlignmentConfig.preset("fast3"), batch_size=512
    )
    args = (config, 32, 515, 3, 64, 8, 9, 2)
    calls = []

    def original(*args, **kwargs):
        calls.append(kwargs)
        return plan_polar_storage(
            *args,
            free_bytes=8 * 1024**3,
            total_bytes=8 * 1024**3,
            workflow_budget_bytes=1 if insufficient else None,
            **kwargs,
        )

    monkeypatch.setattr(backend, "_polar_memory_plan", original)
    records = []
    with validation.controlled_polar_batch(7, records):
        if insufficient:
            with pytest.raises(AssertionError, match="controlled polar batch"):
                backend._polar_memory_plan(*args)
        else:
            plan = backend._polar_memory_plan(*args)
            assert plan.fits_minimum and plan.batch_size == 7
            assert plan.requested_batch_size == 512
            assert plan.particle_storage_policy == "streaming"
            with pytest.raises(AssertionError, match="controlled polar batch"):
                backend._polar_memory_plan(*args, particle_limit=3)
    assert backend._polar_memory_plan is original
    assert calls[0] == {"force_streaming": True, "particle_limit": 7}
    assert records


@pytest.mark.parametrize(
    "defect", [None, "nonexact", "schedule", "array_error", "polar_error"]
)
def test_cross_schedule_requires_positive_controlled_repeat_evidence(
    tmp_path, monkeypatch, defect
):
    from contextlib import nullcontext

    entry = resource_entry()
    entry["deterministic_exact_match"] = False
    entry["result"] = str(tmp_path / "normal.npz")
    np.savez(entry["result"], angle_deg=np.zeros(2, np.float32))
    for record in entry["execution_records"]:
        if record["phase"] == "unprofiled":
            record["gpu_memory_plans"][0]["batch_size"] = 7 - record["index"]
    controlled = resource_entry()
    controlled["execution_records"] = [
        r for r in controlled["execution_records"] if r["phase"] == "unprofiled"
    ]
    for record in controlled["execution_records"]:
        record["gpu_memory_plans"][0]["batch_size"] = 5
    controlled["result"] = str(tmp_path / "controlled.npz")
    np.savez(
        controlled["result"],
        angle_deg=np.full(2, 1.0 if defect == "array_error" else 0.0, np.float32),
    )
    if defect == "nonexact":
        controlled["deterministic_exact_match"] = False
    if defect == "schedule":
        controlled["execution_records"][1]["class_average_batch_size"] = 256
    config = replace(
        validation.mirror.ai.AlignmentConfig.preset("fast3"), batch_size=512
    )
    monkeypatch.setattr(
        validation, "controlled_polar_batch", lambda *args: nullcontext()
    )

    def run(name, case, **kwargs):
        assert kwargs["config"] is config  # original requested batch/config is retained
        assert kwargs["measured_repeats"] == 3 and not kwargs["profile_execution"]
        return controlled

    monkeypatch.setattr(validation.fast, "run_case", run)
    monkeypatch.setattr(
        validation.fast,
        "compare_polar_backend_parity",
        lambda *args: {"within_tolerance": defect != "polar_error"},
    )
    if defect:
        with pytest.raises(AssertionError):
            validation.ensure_repeatability(
                entry, "fixture", {}, config, "cuda", tmp_path / "report.json"
            )
    else:
        validation.ensure_repeatability(
            entry, "fixture", {}, config, "cuda", tmp_path / "report.json"
        )
        assert entry["controlled_repeatability"]["passed"]
        assert entry["controlled_repeatability"]["target_particle_batch"] == 5
        assert not entry[
            "deterministic_exact_match"
        ]  # ordinary timings/results are retained


def test_same_schedule_nonexact_repeats_do_not_trigger_a_probe(monkeypatch, tmp_path):
    entry = resource_entry()
    entry["deterministic_exact_match"] = False
    monkeypatch.setattr(
        validation.fast, "run_case", lambda *a, **k: pytest.fail("unexpected probe")
    )
    with pytest.raises(AssertionError, match="same-backend repeats are not exact"):
        validation.ensure_repeatability(
            entry, "fixture", {}, None, "cuda", tmp_path / "report.json"
        )


@pytest.mark.parametrize("matching", [False, True])
def test_mixed_schedules_cannot_hide_same_schedule_mismatch(
    monkeypatch, tmp_path, matching
):
    entry = resource_entry()
    entry["deterministic_exact_match"] = False
    repeats = [r for r in entry["execution_records"] if r["phase"] == "unprofiled"]
    repeats[1]["result_hashes"]["combined"] = "same" if matching else "different"
    repeats[2]["gpu_memory_plans"][0]["batch_size"] = 5
    repeats[2]["result_hashes"]["combined"] = "third"

    def probe(*args, **kwargs):
        raise RuntimeError("probe reached")

    monkeypatch.setattr(validation.fast, "run_case", probe)
    error = RuntimeError if matching else AssertionError
    message = "probe reached" if matching else "same-schedule repeats are not exact"
    with pytest.raises(error, match=message):
        validation.ensure_repeatability(
            entry, "fixture", {}, None, "cuda", tmp_path / "report.json"
        )
    diagnostic = entry["repeat_execution_diagnostics"]
    assert diagnostic["same_schedule_hashes_exact"] is matching
    if not matching:
        assert diagnostic["mismatched_repeat_indices"] == [0, 1]
        assert "controlled_repeatability" not in entry


def test_resource_gate_distinguishes_measured_and_estimated_bytes():
    result = validation.resource_checks(resource_entry(), "cuda")
    assert result["tracked_pool_live_bytes_peak"] == 512
    assert result["sampled_device_increase_bytes"] == 512
    assert result["budget_bytes"] == 1024


@pytest.mark.parametrize(
    "defect",
    [
        "infeasible",
        "over_budget",
        "pool_peak",
        "device_peak",
        "fallback",
        "maps_downloaded",
        "nondeterministic",
        "not_closed",
    ],
)
def test_resource_gate_rejects_broken_contract(defect):
    entry = resource_entry()
    if defect == "infeasible":
        entry["gpu_memory_plans"][0]["fits_minimum"] = False
    elif defect == "over_budget":
        entry["gpu_memory_plans"][0]["estimated_peak_bytes"] = 2048
    elif defect == "pool_peak":
        entry["performance"]["gpu_memory"]["tracked_pool_live_bytes_peak"] = 2048
    elif defect == "device_peak":
        entry["performance"]["gpu_memory"]["sampled_device_used_bytes_peak"] = 4096
    elif defect == "fallback":
        entry["backend_metadata"]["backend"] = "cpu"
    elif defect == "maps_downloaded":
        entry["backend_metadata"]["polar_full_correlation_map_d2h"] = True
    elif defect == "nondeterministic":
        entry["deterministic_exact_match"] = False
    elif defect == "not_closed":
        entry["gpu_workspace"]["closed"] = False
    with pytest.raises(AssertionError):
        validation.resource_checks(entry, "cuda")


def test_failed_case_keeps_partial_report_and_records(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "runtime_info", lambda *args: {})
    monkeypatch.setattr(validation.mirror, "environment_info", lambda *args: {})

    def fail(*args, report, **kwargs):
        report["active_case"] = "local3050"
        report["speed_gate"] = {"passed": False}
        error = MemoryError("injected")
        error.polar_memory_records = [{"event": "oom_failure"}]
        raise error

    monkeypatch.setattr(validation, "validate_representative", fail)
    path = tmp_path / "failed.json"
    assert (
        validation.main(
            ["--suite", "representative", "--backend", "cuda", "--output", str(path)]
        )
        == 1
    )
    report = json.loads(path.read_text())
    assert report["checks"]["active_case"] == "local3050"
    assert report["checks"]["speed_gate"]["passed"] is False
    assert report["failure_memory_records"] == [{"event": "oom_failure"}]


def test_representative_reuses_case_runner_and_preserves_baseline_config(
    tmp_path, monkeypatch
):
    config = asdict(
        replace(validation.mirror.ai.AlignmentConfig.preset("fast3"), batch_size=512)
    )
    old = {
        "config": config,
        "median_seconds": 10.0,
        "environment": {"hostname": "same", "gpu": "3090"},
    }
    case = validation.fast._case_values(
        np.ones((2, 8, 8), np.float32), np.ones((1, 8, 8), np.float32)
    )
    cases = {
        name: deepcopy(case)
        for name in (
            "k1_class_0",
            "k1_class_1",
            "k1_class_2",
            "fixed_k3",
            "open_k10",
            "local3050",
        )
    }
    prior_values = {"angle_deg": np.zeros(2, np.float32)}
    monkeypatch.setattr(
        validation, "load_representative", lambda root: (cases, old, prior_values, {})
    )
    seen = []

    def run(name, values, **kwargs):
        seen.append((name, kwargs))
        path = tmp_path / (name + ".npz")
        np.savez(path, **prior_values)
        return {"result": str(path), "unprofiled_median_seconds": 10.0}

    monkeypatch.setattr(validation.fast, "run_case", run)
    monkeypatch.setattr(validation, "resource_checks", lambda *args: {})
    monkeypatch.setattr(
        validation, "ensure_repeatability", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        validation.fast,
        "compare_homogeneous_fixed_k3_to_k1",
        lambda *args: {
            "pose_within_tolerance": True,
            "assignment_contract_exact": True,
            "per_class": [{"reference_correlation": 1.0}],
        },
    )
    report = {}
    validation.validate_representative(
        "cuda",
        tmp_path / "report.json",
        {"hostname": "same", "gpu": {"name": "3090"}},
        report=report,
    )
    assert len(seen) == 6
    for name, kwargs in seen:
        assert asdict(kwargs["config"]) == config
        assert kwargs["measured_repeats"] == 3 and kwargs["profile_execution"]
    assert report["speed_gate"]["passed"]
    assert report["baseline_errors"]["angle_deg"]["maximum_absolute_error"] == 0


@pytest.mark.gpu
@pytest.mark.parametrize("engine", ["cuda", "cupy"])
def test_real_gpu_t6_storage_parity(engine):
    validation.runtime_info(engine)
    report = {}
    validation.validate_storage_parity(engine, batch_size=512, report=report)
    assert set(report) == {"fixed_False", "fixed_True"}
    assert all(
        value["plans"][-1]["particle_storage_policy"] == "streaming"
        for value in report.values()
    )
