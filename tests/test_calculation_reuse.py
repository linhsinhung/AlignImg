from dataclasses import replace

import numpy as np
import pytest

import alignimg as ai
from alignimg import _fourier
from alignimg._frequency_tables import frequency_tables
from alignimg._profiling import ExecutionProfile, _ACTIVE
from tools.performance_fixtures import fixture_config, synthetic_inputs
from tools.performance_validation import execute, result_arrays


@pytest.mark.parametrize("size", [16, 24, 32])
def test_frequency_tables_preserve_dft_indexing_and_are_bounded(size):
    tables = frequency_tables(size)
    np.testing.assert_array_equal(
        tables[0],
        np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")[0],
    )
    assert frequency_tables(size) is tables
    assert frequency_tables.cache_info().currsize == 1
    assert all(not value.flags.writeable for value in tables)
    y, x = np.indices((size, size))
    np.testing.assert_array_equal(tables[4], (-1.0) ** (y + x))


@pytest.mark.parametrize("model", ["fourier_ncc", "whitened_fourier_ncc"])
def test_norm_cache_tracks_particle_profile_and_reference_replacement(model):
    values = synthetic_inputs(4, 16)
    config = replace(fixture_config(False), score_model=model)
    particles = _fourier.prepare_stack(values["images"], config)
    references = _fourier.prepare_stack(values["references"], config)
    first = _fourier.reference_norms_for_particle(particles, references, 0)
    assert _fourier.reference_norms_for_particle(particles, references, 0) is first
    second = _fourier.reference_norms_for_particle(particles, references, 1)
    assert (second is first) == (model == "fourier_ncc")
    for index in (1, 0):
        weights = _fourier.score_weights_for_particle(particles, index)
        expected = [np.sum(weights * np.abs(ref) ** 2) for ref in references.fourier]
        np.testing.assert_array_equal(
            _fourier.reference_norms_for_particle(particles, references, index),
            expected,
        )
    updated = _fourier.prepare_stack(values["references"][:, ::-1].copy(), config)
    assert not updated._reference_norm_cache
    norms = _fourier.reference_norms_for_particle(particles, updated, 0)
    assert not np.array_equal(norms, first)


@pytest.mark.parametrize("workflow", ["global", "refine", "reference_free"])
@pytest.mark.parametrize("model", ["fourier_ncc", "whitened_fourier_ncc"])
@pytest.mark.parametrize("scoring", ["fourier", "raster"])
def test_reuse_matches_uncached_scores_and_multi_iteration_updates(
    monkeypatch, workflow, model, scoring
):
    values = synthetic_inputs(4, 16)
    config = replace(
        fixture_config(workflow == "refine"),
        max_iterations=2,
        score_model=model,
        candidate_scoring=scoring,
        mirror_search=True,
    )
    reused = result_arrays(execute(values, config, "cpu", workflow))
    cached_norms = _fourier.reference_norms_for_particle
    angles = _fourier._candidate_angles

    def uncached_norms(particles, references, index):
        references._reference_norm_cache.clear()
        return cached_norms(particles, references, index)

    def uncached_angles(a, b, count, **kwargs):
        return angles(a, b, count)

    monkeypatch.setattr(_fourier, "reference_norms_for_particle", uncached_norms)
    monkeypatch.setattr(_fourier, "_candidate_angles", uncached_angles)
    plain = result_arrays(execute(values, config, "cpu", workflow))
    for name in reused:
        np.testing.assert_array_equal(reused[name], plain[name], err_msg=name)


@pytest.mark.parametrize("backend", ["cpu", "cuda", "cupy"])
def test_reuse_reduces_fft_and_norm_work(backend):
    if not ai.available_alignment_backends()[backend]["available"]:
        pytest.skip(f"{backend} unavailable")
    values = synthetic_inputs(4, 16)
    config = replace(fixture_config(False), max_iterations=2, profile_execution=True)
    global_result = execute(values, config, backend, "global")
    counts = global_result.metadata["performance"]["counters"]
    # N particle FFTs once, K reference FFTs per iteration; previously 2*N*K*I.
    assert counts["polar_angular_fft_calls"] == 4 + 2 * 2
    refined = execute(
        values, replace(fixture_config(True), profile_execution=True), backend, "refine"
    )
    counts = refined.metadata["performance"]["counters"]
    assert counts["reference_norm_evaluations"] == 2
    assert counts["reference_norm_cache_hits"] == 2 * 4 - 1


def test_frequency_build_counter_counts_actual_rebuilds():
    frequency_tables.cache_clear()
    profile = ExecutionProfile()
    token = _ACTIVE.set(profile)
    try:
        for _ in range(10):
            frequency_tables(16)
        frequency_tables(24)
        frequency_tables(16)
        assert profile.counters["frequency_table_builds"] == 3
    finally:
        _ACTIVE.reset(token)


@pytest.mark.parametrize("backend", ["cuda", "cupy"])
def test_gpu_workspace_reuses_scoring_and_update_fft_across_complete_workflow(backend):
    if not ai.available_alignment_backends()[backend]["available"]:
        pytest.skip(f"{backend} unavailable")
    values = synthetic_inputs(4, 16)
    config = replace(
        fixture_config(False), max_iterations=2, profile_execution=True
    )
    for _ in range(2):
        result = execute(values, config, backend, "global")
        counts = result.metadata["performance"]["counters"]
        workspace = result.metadata["gpu_workspace"]
        assert counts["workspace_scoring_cache_uploads"] == 1
        assert counts["workspace_scoring_cache_hits"] == 1
        assert counts["workspace_update_cache_uploads"] == 1
        assert counts["workspace_update_cache_hits"] == 5
        assert counts["workspace_release_calls"] == 1
        assert workspace["closed"]
        assert workspace["roles"]["scoring"]["policy"] == "device"
        assert workspace["roles"]["update"]["policy"] == "device"
        expected_cache_bytes = 2 * np.asarray(
            values["images"], dtype=np.complex64
        ).nbytes
        assert workspace["released_cache_bytes"] == expected_cache_bytes
