from __future__ import annotations

from dataclasses import asdict
from importlib.util import find_spec
import json
from pathlib import Path

import mrcfile
import numpy as np
import pytest

import alignimg as ai
from alignimg._geometry import CENTER_CONVENTION
from alignimg_gui.artifacts import load_previous_result, load_result_bundle
from alignimg_gui.runner import execute_run
from alignimg_gui.spec import RunSpec, prepare_run_directory


def small_config(*, store_history: bool = True) -> dict:
    return asdict(
        ai.AlignmentConfig(
            max_iterations=1,
            top_l=2,
            angle_samples=12,
            proposal_angles_per_reference=2,
            translation_range=1,
            halfset_diagnostics=False,
            center_references=False,
            store_history=store_history,
            batch_size=4,
        )
    )


def test_run_spec_validates_workflow_inputs(tmp_path: Path):
    with pytest.raises(ValueError, match="requires a reference"):
        RunSpec(
            particles="particles.mrcs",
            output_directory=str(tmp_path),
            run_name="missing-reference",
            mode="reference_based",
        ).normalized()
    with pytest.raises(ValueError, match="requires its global RF stage"):
        RunSpec(
            particles="particles.mrcs",
            output_directory=str(tmp_path),
            run_name="invalid-rf-resume",
            mode="reference_free",
            run_global=False,
        ).normalized()
    with pytest.raises(ValueError, match="single non-empty directory"):
        RunSpec(
            particles="particles.mrcs",
            output_directory=str(tmp_path),
            run_name="../escape",
        ).normalized()


def test_gui_runner_writes_reloadable_alignment_artifacts(tmp_path: Path):
    size = 16
    y, x = np.indices((size, size), dtype=np.float32)
    reference = np.exp(-((y - 5) ** 2 + (x - 10) ** 2) / 4.0).astype(np.float32)
    particles = np.repeat(reference[None], 3, axis=0)
    particle_path = tmp_path / "particles.mrcs"
    reference_path = tmp_path / "reference.mrc"
    with mrcfile.new(particle_path) as mrc:
        mrc.set_data(particles)
        mrc.voxel_size = 2.0
    with mrcfile.new(reference_path) as mrc:
        mrc.set_data(reference)
        mrc.voxel_size = 2.0

    spec, run_directory = prepare_run_directory(
        RunSpec(
            particles=str(particle_path),
            references=str(reference_path),
            output_directory=str(tmp_path / "runs"),
            run_name="pipeline-test",
            mode="reference_based",
            backend="cpu",
            refine_enabled=True,
            global_config=small_config(),
            refine_config=small_config(store_history=False),
        )
    )
    events: list[dict] = []
    report = execute_run(spec, emit=events.append)

    assert report["status"] == "completed"
    assert report["input"]["particle_count"] == 3
    assert report["input"]["reference_count"] == 1
    assert report["final_stage"] == "refine"
    assert set(report["stages"]) == {"global", "refine"}
    assert (run_directory / "references.mrcs").exists()
    assert (run_directory / "result.npz").exists()
    for stage in ("global", "refine"):
        assert (run_directory / stage / "report.json").exists()
        assert (run_directory / stage / "references.mrcs").exists()
        assert (run_directory / stage / "result.npz").exists()
    assert [event["event"] for event in events] == [
        "started",
        "running",
        "stage_completed",
        "running",
        "stage_completed",
        "saving",
        "completed",
    ]
    loaded_report, arrays = load_result_bundle(run_directory / "report.json")
    assert loaded_report["status"] == "completed"
    assert arrays["reference_history"].shape == (3, 1, size, size)
    assert arrays["responsibilities"].shape == (3, 1)
    assert arrays["pose_entropy"].shape == (3,)
    assert arrays["map_posterior"].shape == (3,)
    assert arrays["soft_references"].shape == (1, size, size)
    assert arrays["class_averages"].shape == (1, size, size)
    assert str(arrays["center_convention"].item()) == CENTER_CONVENTION


def test_gui_runner_runs_reference_free_without_refinement(tmp_path: Path):
    size = 16
    y, x = np.indices((size, size), dtype=np.float32)
    reference = np.exp(-((y - 5) ** 2 + (x - 10) ** 2) / 4.0).astype(np.float32)
    particle_path = tmp_path / "particles.mrcs"
    with mrcfile.new(particle_path) as mrc:
        mrc.set_data(np.repeat(reference[None], 3, axis=0))

    spec, run_directory = prepare_run_directory(
        RunSpec(
            particles=str(particle_path),
            output_directory=str(tmp_path / "runs"),
            run_name="rf-test",
            mode="reference_free",
            backend="cpu",
            n_components=1,
            refine_enabled=False,
            global_config=small_config(),
        )
    )
    report = execute_run(spec)

    assert report["status"] == "completed"
    assert report["final_stage"] == "global"
    assert set(report["stages"]) == {"global"}
    _, arrays = load_result_bundle(run_directory / "report.json")
    assert arrays["reference_history"].shape == (2, 1, size, size)


def test_gui_converts_previous_results_without_center_metadata(tmp_path: Path):
    path = tmp_path / "legacy-result.npz"
    legacy = ai.PoseSet(
        np.asarray([30.0]),
        np.asarray([2.0]),
        np.asarray([-1.0]),
        np.asarray([False]),
    )
    np.savez_compressed(
        path,
        angle_deg=legacy.angle_deg,
        shift_y_px=legacy.shift_y_px,
        shift_x_px=legacy.shift_x_px,
        mirror=legacy.mirror,
        assignments=np.asarray([0], dtype=np.int32),
    )
    with pytest.warns(UserWarning, match="pre-1.5"):
        poses, assignments, responsibilities = load_previous_result(path, 1, 16)
    expected = ai.convert_v1_4_poses_to_integer_center(legacy, 16)
    assert np.allclose(poses.shift_y_px, expected.shift_y_px)
    assert np.allclose(poses.shift_x_px, expected.shift_x_px)
    assert assignments.tolist() == [0]
    assert responsibilities is None


def test_main_window_can_be_created_offscreen(monkeypatch):
    pyqt_spec = find_spec("PyQt6")
    if pyqt_spec is None or pyqt_spec.origin is None:
        pytest.skip("PyQt6 is not installed")
    plugin_path = Path(pyqt_spec.origin).parent / "Qt6" / "plugins" / "platforms"
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    if plugin_path.is_dir():
        monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugin_path))
    from PyQt6.QtWidgets import QApplication

    from alignimg_gui.window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.mode_tabs.currentIndex() == 0
    assert window.global_iterations.value() == 15


@pytest.mark.parametrize("components", [1, 10])
def test_main_window_displays_reference_free_report_without_refinement(
    monkeypatch, tmp_path: Path, components: int
):
    pyqt_spec = find_spec("PyQt6")
    if pyqt_spec is None or pyqt_spec.origin is None:
        pytest.skip("PyQt6 is not installed")
    plugin_path = Path(pyqt_spec.origin).parent / "Qt6" / "plugins" / "platforms"
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    if plugin_path.is_dir():
        monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugin_path))
    from PyQt6.QtWidgets import QApplication

    from alignimg_gui.window import MainWindow

    size = 16
    y, x = np.indices((size, size), dtype=np.float32)
    image = np.exp(-((y - 5) ** 2 + (x - 10) ** 2) / 4.0).astype(np.float32)
    particle_path = tmp_path / "particles.mrcs"
    with mrcfile.new(particle_path) as mrc:
        mrc.set_data(
            np.stack(
                [
                    np.roll(image, shift=index, axis=1)
                    for index in range(max(3, components))
                ]
            )
        )
    spec, run_directory = prepare_run_directory(
        RunSpec(
            particles=str(particle_path),
            output_directory=str(tmp_path / "runs"),
            run_name="rf-display-test",
            mode="reference_free",
            backend="cpu",
            n_components=components,
            refine_enabled=False,
            global_config=small_config(),
        )
    )
    execute_run(spec)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._display_report(run_directory / "report.json")
    window.results.refresh()
    assert window.results.gallery_layout.count() == components


def test_main_window_displays_reference_based_mra_report(monkeypatch, tmp_path: Path):
    pyqt_spec = find_spec("PyQt6")
    if pyqt_spec is None or pyqt_spec.origin is None:
        pytest.skip("PyQt6 is not installed")
    plugin_path = Path(pyqt_spec.origin).parent / "Qt6" / "plugins" / "platforms"
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    if plugin_path.is_dir():
        monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugin_path))
    from PyQt6.QtWidgets import QApplication

    from alignimg_gui.window import MainWindow

    size = 16
    image = np.zeros((size, size), dtype=np.float32)
    particle_path = tmp_path / "particles.mrcs"
    reference_path = tmp_path / "reference.mrc"
    with mrcfile.new(particle_path) as mrc:
        mrc.set_data(np.repeat(image[None], 3, axis=0))
    with mrcfile.new(reference_path) as mrc:
        mrc.set_data(image)
    spec, run_directory = prepare_run_directory(
        RunSpec(
            particles=str(particle_path),
            references=str(reference_path),
            output_directory=str(tmp_path / "runs"),
            run_name="mra-display-test",
            mode="reference_based",
            backend="cpu",
            refine_enabled=True,
            global_config=small_config(),
            refine_config=small_config(store_history=False),
        )
    )
    execute_run(spec)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._display_report(run_directory / "report.json")
    window.results.refresh()
    assert window.results.gallery_layout.count() == 1
    assert window.refine_enabled.isChecked()
    assert window.refine_iterations.value() == 2
    assert window.candidate_scoring.currentData() == "fourier"
    assert window.score_model.currentData() == "fourier_ncc"
    assert window.reference_update.currentData() == "fourier"
    assert not window.apply_final_pose_to_raw.isChecked()
    assert window.refine_search_strategy.currentData() == "quadratic_refine"
    assert window._refine_config()["coarse_angle_step"] == 1.0
    assert window._refine_config()["local_angle_range"] == 7.0
    assert window._refine_config()["local_shift_range"] == 3.0
    window.apply_final_pose_to_raw.setChecked(True)
    assert not window._global_config()["apply_final_pose_to_raw"]
    assert window._refine_config()["apply_final_pose_to_raw"]
    window.mode_tabs.setCurrentIndex(1)
    assert window.references.isEnabled()
    assert not window.previous.isEnabled()
    assert window.global_iterations.value() == 10
    window.reference_start.setCurrentIndex(1)
    assert window.previous.isEnabled()
    assert not window.global_iterations.isEnabled()
    assert window.refine_enabled.isChecked()
    assert not window.refine_enabled.isEnabled()
    assert window.corrective_trust.isEnabled()
    assert window.search_strategy.currentData() == "proposal"
    assert window._refine_config()["search_strategy"] == "quadratic_refine"
    window.refine_search_strategy.setCurrentIndex(
        window.refine_search_strategy.findData("adaptive_posterior")
    )
    assert window._refine_config()["search_strategy"] == "adaptive_posterior"
    assert window._refine_config()["coarse_angle_step"] == 6.0
    assert window._refine_config()["local_angle_range"] == 15.0
    window.results.report = {
        "stages": {
            "global": {"diagnostics": [{}, {}]},
            "refine": {"diagnostics": [{}]},
        }
    }
    assert window.results._history_labels(4) == [
        "Initial",
        "Global 1/2",
        "Global 2/2",
        "Refine 1/1",
    ]
    window.close()
    app.processEvents()


def test_summary_exposes_execution_rescue_and_memory_plans(tmp_path: Path, monkeypatch):
    pyqt_spec = find_spec("PyQt6")
    if pyqt_spec is None or pyqt_spec.origin is None:
        pytest.skip("PyQt6 is not installed")
    plugin_path = Path(pyqt_spec.origin).parent / "Qt6" / "plugins" / "platforms"
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    if plugin_path.is_dir():
        monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugin_path))
    from PyQt6.QtWidgets import QApplication

    from alignimg_gui.widgets import SummaryWidget

    report = {
        "status": "completed",
        "metadata": {
            "engine": "alignimg-soft-fourier-cupy",
            "backend": "cuda",
            "candidate_scoring": "fourier",
            "score_model": "fourier_ncc",
            "reference_update": "fourier",
            "gpu_memory_plan_at_completion": {"batch_size": 512},
            "gpu_memory_plans": [
                {
                    "stage": "candidate_inference",
                    "batch_size": 512,
                    "budget_bytes": 1024,
                }
            ],
        },
        "diagnostics": [
            {
                "rescue_particle_count": 3,
                "rescue_accepted_count": 2,
                "rescue_rejected_count": 1,
                "rescue_scheduled_count": 4,
                "mean_rescue_score_gain": 0.1,
            }
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    app = QApplication.instance() or QApplication([])
    widget = SummaryWidget()
    widget.load_report(report_path)
    displayed = json.loads(widget.toPlainText())
    assert displayed["execution"]["candidate_scoring"] == "fourier"
    assert displayed["execution"]["gpu_memory_plan_stages"][0]["batch_size"] == 512
    assert displayed["rescue"]["accepted_total"] == 2
    widget.close()
    app.processEvents()
