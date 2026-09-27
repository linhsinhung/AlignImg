from __future__ import annotations

from dataclasses import replace

import numpy as np

from tools.fast_to_precise_validation import (
    _execute_schedule,
    _stage_summary,
    parse_args,
    run_schedule,
    schedule_configs,
)


def _case(size: int = 16) -> dict[str, np.ndarray]:
    y, x = np.indices((size, size), dtype=np.float32)
    reference = np.exp(-((y - 5.0) ** 2 + (x - 10.0) ** 2) / 4.0)
    reference += 0.6 * np.exp(-((y - 11.0) ** 2 + (x - 6.0) ** 2) / 3.0)
    images = np.repeat(reference[None], 2, axis=0).astype(np.float32)
    return {
        "images": images,
        "references": reference[None].astype(np.float32),
        "component": np.zeros(2, dtype=np.int32),
        "priors": np.empty((0, 0), dtype=np.float32),
        "truth_angle_deg": np.empty(0, dtype=np.float32),
        "truth_shift_y_px": np.empty(0, dtype=np.float32),
        "truth_shift_x_px": np.empty(0, dtype=np.float32),
        "truth_mirror": np.empty(0, dtype=np.bool_),
    }


def test_equal_iteration_schedule_contract_and_existing_defaults_are_unchanged():
    schedules = schedule_configs(
        global_iterations=3,
        precise_iterations=2,
        batch_size=512,
        memory_fraction=0.8,
    )

    assert {
        name: sum(config.max_iterations for config in configs)
        for name, configs in schedules.items()
    } == {"balanced_soft": 5, "fast_only": 5, "fast_to_precise": 5}
    assert schedules["balanced_soft"][0].search_strategy == "proposal"
    assert schedules["balanced_soft"][0].halfset_diagnostics is True
    assert schedules["fast_only"][0].search_strategy == "polar_hard"
    assert schedules["fast_only"][0].halfset_diagnostics is False
    fast, precise = schedules["fast_to_precise"]
    assert fast.search_strategy == "polar_hard"
    assert fast.halfset_diagnostics is False
    assert precise.search_strategy == "quadratic_refine"
    assert precise.halfset_diagnostics is True


def test_corrective_schedule_keeps_twelve_total_iterations():
    schedules = schedule_configs(
        global_iterations=10,
        precise_iterations=2,
        batch_size=512,
        memory_fraction=0.8,
    )

    assert {
        name: sum(config.max_iterations for config in configs)
        for name, configs in schedules.items()
    } == {"balanced_soft": 12, "fast_only": 12, "fast_to_precise": 12}
    assert [config.max_iterations for config in schedules["fast_to_precise"]] == [
        10,
        2,
    ]


def test_fast_to_precise_executes_fixed_class_ladder_on_cpu():
    configs = schedule_configs(
        global_iterations=1,
        precise_iterations=1,
        batch_size=8,
        memory_fraction=0.8,
    )["fast_to_precise"]
    configs = tuple(
        replace(
            config,
            angle_samples=36,
            translation_range=0.0,
            local_angle_range=5.0,
            coarse_angle_step=5.0,
            local_shift_range=0.0,
            center_references=False,
        )
        for config in configs
    )

    final, stages = _execute_schedule(_case(), configs, "cpu")

    assert len(stages) == 2
    assert stages[0].metadata["search_strategy"] == "polar_hard"
    assert stages[1].metadata["search_strategy"] == "quadratic_refine"
    assert np.array_equal(stages[0].reference_assignments, final.reference_assignments)
    assert final.metadata["class_average_estimator"] == (
        "final_map_pose_inlier_weighted_raw"
    )


def test_stage_summary_reports_quality_before_and_after_precise_refinement():
    case = _case()
    case.update(
        component=np.zeros(2, dtype=np.int32),
        truth_angle_deg=np.zeros(2, dtype=np.float32),
        truth_shift_y_px=np.zeros(2, dtype=np.float32),
        truth_shift_x_px=np.zeros(2, dtype=np.float32),
        truth_mirror=np.zeros(2, dtype=np.bool_),
    )
    configs = schedule_configs(
        global_iterations=1,
        precise_iterations=1,
        batch_size=8,
        memory_fraction=0.8,
    )["fast_to_precise"]
    configs = tuple(
        replace(
            config,
            angle_samples=36,
            translation_range=0.0,
            local_angle_range=5.0,
            coarse_angle_step=5.0,
            local_shift_range=0.0,
            center_references=False,
        )
        for config in configs
    )
    _, stages = _execute_schedule(case, configs, "cpu")

    summary = _stage_summary(stages, case, "cpu")

    assert [stage["search_strategy"] for stage in summary] == [
        "polar_hard",
        "quadratic_refine",
    ]
    assert [stage["quality"]["assignment_accuracy"] for stage in summary] == [
        1.0,
        1.0,
    ]
    assert [stage["occupancy"].tolist() for stage in summary] == [[2], [2]]


def test_run_schedule_records_custom_primary_stage_analysis(tmp_path):
    case = _case()
    configs = schedule_configs(
        global_iterations=1,
        precise_iterations=1,
        batch_size=8,
        memory_fraction=0.8,
    )["fast_to_precise"]
    configs = tuple(
        replace(
            config,
            angle_samples=36,
            translation_range=0.0,
            local_angle_range=5.0,
            coarse_angle_step=5.0,
            local_shift_range=0.0,
            center_references=False,
        )
        for config in configs
    )

    report = run_schedule(
        "test:k1",
        case,
        schedule="fast_to_precise",
        configs=configs,
        backend="cpu",
        measured_repeats=1,
        profile_execution=False,
        output=tmp_path / "report.json",
        stage_analysis=lambda stages: {
            "strategies": [stage.metadata["search_strategy"] for stage in stages]
        },
    )

    assert report["unprofiled_stage_analysis"] == {
        "strategies": ["polar_hard", "quadratic_refine"]
    }


def test_representative_cli_uses_frozen_inputs_directory(tmp_path):
    args = parse_args(
        [
            "--suite",
            "representative",
            "--backend",
            "cpu",
            "--inputs-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    assert args.suite == "representative"
    assert args.inputs_dir == tmp_path
    assert args.global_iterations == 3
    assert args.precise_iterations == 2
