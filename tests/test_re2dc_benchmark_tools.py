from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np

import alignimg as ai
from alignimg_gui.artifacts import load_result_bundle
from tools.prepare_re2dc_70s_benchmark import (
    ctf_array,
    fourier_crop,
    phase_flip,
    read_particle_loop,
)
from tools.re2dc_70s_rf_validation import (
    adjusted_rand_index,
    normalized_entropy,
    run_metrics,
    save_run,
)
from tools.re2dc_70s_final_validation import validate_result_bundle
from tools.re2dc_70s_relion_benchmark import (
    align_relion_metadata,
    comparison_metrics,
    normalized_mutual_information,
)
from tools.re2dc_70s_rf_ablation import VARIANTS
from tools.quadratic_refine_validation import (
    accuracy_gate,
    relion_angle_lattice_metrics,
)


def test_fourier_crop_preserves_constant_amplitude():
    image = np.full((130, 130), 3.5, dtype=np.float32)
    cropped = fourier_crop(image, 128)
    assert cropped.shape == (128, 128)
    assert cropped.dtype == np.float32
    assert np.allclose(cropped, 3.5, atol=1e-6)
    with np.testing.assert_raises_regex(ValueError, "even output size"):
        fourier_crop(image, 127)


def test_phase_flip_is_self_inverse_up_to_global_polarity():
    rng = np.random.default_rng(3)
    image = rng.normal(size=(16, 16)).astype(np.float32)
    fy = np.fft.fftfreq(16)[:, None]
    fx = np.fft.fftfreq(16)[None, :]
    ctf = np.where(fx * fx + fy * fy < 0.08, 1.0, -1.0).astype(np.float32)
    observed = phase_flip(image, ctf)
    recovered = phase_flip(observed, ctf)
    assert np.allclose(recovered, image, atol=1e-5)


def test_ctf_array_includes_phase_shift():
    parameters = {
        "pixel_size": 2.82,
        "defocus_u": 21580.0,
        "defocus_v": 21700.0,
        "defocus_angle_deg": 13.0,
        "voltage_kv": 200.0,
        "cs_mm": 2.0,
        "amplitude_contrast": 0.15,
    }
    without_shift = ctf_array((16, 16), phase_shift_deg=0.0, **parameters)
    with_shift = ctf_array((16, 16), phase_shift_deg=25.0, **parameters)
    assert not np.allclose(with_shift, without_shift)


def test_star_particle_loop_parser(tmp_path: Path):
    star = tmp_path / "particles.star"
    star.write_text(
        "data_images\n\nloop_\n_rlnImageName #1\n_rlnDefocusU #2\n"
        "000001@x.mrcs 20000\n000002@x.mrcs 21000\n",
        encoding="utf-8",
    )
    labels, rows = read_particle_loop(star)
    assert labels == ["_rlnImageName", "_rlnDefocusU"]
    assert rows[1] == ["000002@x.mrcs", "21000"]


def test_assignment_diagnostics_are_permutation_invariant():
    first = np.asarray([0, 0, 1, 1, 2, 2])
    second = np.asarray([2, 2, 0, 0, 1, 1])
    assert adjusted_rand_index(first, second) == 1.0
    assert np.isclose(normalized_entropy(np.asarray([2, 2, 2])), 1.0)


def write_classification_star(
    path: Path,
    rows: list[tuple[int, int, int, float, int]],
) -> None:
    lines = [
        "data_particles",
        "",
        "loop_",
        "_rlnImageId #1",
        "_rlnImageName #2",
        "_rlnClassNumber #3",
        "_rlnGroupNumber #4",
        "_rlnMaxValueProbDistribution #5",
        "_rlnNrOfSignificantSamples #6",
    ]
    lines.extend(
        f"{image_id} {image_id:06d}@source.mrcs {class_number} {group_number} "
        f"{probability} {significant_samples}"
        for image_id, class_number, group_number, probability, significant_samples in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prepared_star(path: Path, image_ids: list[int]) -> None:
    lines = [
        "data_images",
        "",
        "loop_",
        "_rlnImageId #1",
        "_rlnImageName #2",
    ]
    lines.extend(
        f"{image_id} {output_index:06d}@prepared.mrcs"
        for output_index, image_id in enumerate(image_ids, start=1)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_relion_metadata_is_mapped_by_image_id_and_uses_class_number(tmp_path: Path):
    relion_star = tmp_path / "relion.star"
    prepared_star = tmp_path / "prepared.star"
    write_classification_star(
        relion_star,
        [
            (1, 2, 7, 0.9, 1),
            (2, 1, 7, 0.8, 2),
            (3, 2, 8, 0.7, 3),
            (4, 1, 8, 0.6, 4),
        ],
    )
    write_prepared_star(prepared_star, [3, 1, 4, 2])

    metadata = align_relion_metadata(relion_star, prepared_star)

    assert np.array_equal(metadata["image_id"], [3, 1, 4, 2])
    assert np.array_equal(metadata["class_number"], [2, 2, 1, 1])
    assert np.array_equal(metadata["group_number"], [8, 7, 8, 7])


def test_relion_comparison_metrics_are_label_permutation_invariant():
    relion = np.asarray([1, 1, 2, 2, 3, 3])
    alignimg = np.asarray([2, 2, 0, 0, 1, 1])

    metrics = comparison_metrics(relion, alignimg)

    assert metrics["adjusted_rand_index"] == 1.0
    assert np.isclose(metrics["normalized_mutual_information"], 1.0)
    assert metrics["optimal_label_agreement"] == 1.0
    assert metrics["optimal_predicted_to_relion_mapping"] == {0: 2, 1: 3, 2: 1}
    assert np.array_equal(
        metrics["confusion_matrix_relion_rows_alignimg_columns"],
        [[0, 0, 2], [2, 0, 0], [0, 2, 0]],
    )


def test_normalized_mutual_information_detects_independent_partitions():
    first = np.asarray([0, 0, 1, 1] * 2)
    second = np.asarray([0, 1, 0, 1] * 2)
    assert np.isclose(normalized_mutual_information(first, second), 0.0)


def test_relion_mapping_rejects_missing_prepared_particle(tmp_path: Path):
    relion_star = tmp_path / "relion.star"
    prepared_star = tmp_path / "prepared.star"
    write_classification_star(relion_star, [(1, 1, 1, 0.9, 1)])
    write_prepared_star(prepared_star, [1, 2])

    with np.testing.assert_raises_regex(ValueError, "missing prepared particle"):
        align_relion_metadata(relion_star, prepared_star)


def test_first_rf_ablation_changes_one_factor_at_a_time():
    assert VARIANTS == {
        "baseline": {},
        "top_l_8": {"top_l": 8},
        "proposal_angles_8": {"proposal_angles_per_reference": 8},
        "temperature_end_0_05": {"temperature_end": 0.05},
    }


def test_relion_angle_lattice_and_resolution_aware_gate():
    lattice = relion_angle_lattice_metrics(
        -0.60394 + 5.625 * np.arange(8, dtype=np.float64)
    )
    assert lattice["half_step_deg"] == 2.8125
    assert lattice["maximum_absolute_residual_deg"] < 1e-10

    def metrics(angle, shift, component_angles, correlations):
        return {
            "angle_error_deg": {"median": angle, "p95": 150.0},
            "shift_error_px": {"median": shift, "p95": 15.0},
            "per_component": [
                {
                    "angle_error_deg": {"median": value},
                    "result_reference_correlation": correlation,
                }
                for value, correlation in zip(
                    component_angles, correlations, strict=True
                )
            ],
        }

    adaptive = metrics(2.8125, 2.22, [2.8125, 2.8125, 5.4375], [0.85] * 3)
    quadratic = metrics(3.37, 2.11, [3.37, 2.80, 4.08], [0.86] * 3)
    initial = metrics(5.625, 3.4, [5.625, 2.8125, 5.625], [0.0] * 3)
    gate = accuracy_gate(adaptive, quadratic, initial)

    assert gate["passed"]
    assert not gate["observations"]["raw_median_angle_strictly_improved"]
    quadratic["per_component"][1]["angle_error_deg"]["median"] = 2.8125
    assert not accuracy_gate(adaptive, quadratic, initial)["passed"]


def test_rf_seed_artifacts_are_gui_loadable(tmp_path: Path):
    size = 16
    y, x = np.indices((size, size), dtype=np.float32)
    first = np.exp(-((y - 5) ** 2 + (x - 10) ** 2) / 4.0)
    second = np.exp(-((y - 11) ** 2 + (x - 5) ** 2) / 3.0)
    images = np.stack((first, first, second, second)).astype(np.float32)
    config = ai.AlignmentConfig(
        max_iterations=1,
        top_l=2,
        angle_samples=12,
        proposal_angles_per_reference=2,
        translation_range=1.0,
        halfset_diagnostics=True,
        center_references=False,
        store_history=True,
        batch_size=4,
    )
    result = ai.reference_free_align(
        images, n_components=2, config=config, backend="cpu"
    )
    metrics = run_metrics(result, 1.0, 2, 1.0, 2.5)
    artifacts = save_run(
        tmp_path / "formal.seed-0",
        result,
        pixel_size=2.5,
        source_stack=tmp_path / "particles.mrcs",
        backend="cpu",
        config=config,
        seed=0,
        seconds=1.0,
        summary={"seconds": 1.0},
    )

    report, arrays = load_result_bundle(artifacts["gui_report"])
    assert report["status"] == "completed"
    assert report["input"]["pixel_size_angstrom"] == 2.5
    assert arrays["inlier_weights"].shape == (4,)
    assert arrays["reference_history"].shape == (2, 2, size, size)
    assert metrics["responsibility_sum_max_error"] < 1e-6
    assert metrics["reliable_median_stable_frc_resolution_angstrom"] is not None
    with mrcfile.open(artifacts["references"]) as mrc:
        assert np.isclose(float(mrc.voxel_size.x), 2.5)
    checks = validate_result_bundle(
        Path(artifacts["result"]),
        Path(artifacts["references"]),
        particle_count=4,
        component_count=2,
    )
    assert checks["all_required_arrays_finite"]
