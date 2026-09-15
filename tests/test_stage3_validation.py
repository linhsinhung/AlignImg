from __future__ import annotations

import pytest

from tools.stage3_validation import validate_batching_contract


def cases():
    baseline_counters = {
        "coarse_candidates": 800,
        "fine_candidates": 1600,
        "h2d_calls": 220,
        "h2d_bytes": 100_000,
        "d2h_calls": 80,
        "d2h_bytes": 9600,
    }
    current_counters = {
        "coarse_candidates": 800,
        "fine_candidates": 1600,
        "adaptive_particle_batches": 4,
        "adaptive_batched_particles": 64,
        "adaptive_scoring_batches": 12,
        "adaptive_cross_particle_batches": 8,
        "h2d_calls": 80,
        "h2d_bytes": 110_000,
        "d2h_calls": 20,
        "d2h_bytes": 9600,
    }

    def performance(counters, coarse_calls, fine_calls):
        return {
            "counters": counters,
            "stages": {
                "workflow/alignment_engine/candidate_inference/adaptive_controller/coarse_scoring": {
                    "calls": coarse_calls
                },
                "workflow/alignment_engine/candidate_inference/adaptive_controller/fine_scoring": {
                    "calls": fine_calls
                },
            },
        }

    baseline = {"performance": performance(baseline_counters, 64, 64)}
    current = {
        "particle_count": 32,
        "config": {"max_iterations": 2},
        "performance": performance(current_counters, 4, 4),
        "gpu_memory_plans": [],
        "gpu_workspace": {
            "closed": True,
            "budget_bytes": 1_000_000,
            "peak_committed_cache_bytes": 100_000,
        },
    }
    return baseline, current


def test_stage3_batching_contract_accepts_equal_work_with_fewer_syncs():
    baseline, current = cases()
    report = validate_batching_contract("case", baseline, current)

    assert report["coarse_scorer_calls_saved"] == 60
    assert report["fine_scorer_calls_saved"] == 60
    assert report["h2d_calls_saved"] == 140
    assert report["d2h_calls_saved"] == 60
    assert report["h2d_bytes_change"] == 10_000
    assert report["d2h_bytes_change"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda old, new: new["performance"]["counters"].update(
                fine_candidates=1599
            ),
            "fine_candidates changed",
        ),
        (
            lambda old, new: new["performance"]["stages"][
                "workflow/alignment_engine/candidate_inference/adaptive_controller/coarse_scoring"
            ].update(calls=64),
            "call count did not decrease",
        ),
        (
            lambda old, new: new["performance"]["counters"].update(d2h_calls=80),
            "D2H synchronization count did not decrease",
        ),
        (
            lambda old, new: new["performance"]["counters"].update(
                adaptive_cross_particle_batches=0
            ),
            "no cross-particle",
        ),
    ],
)
def test_stage3_batching_contract_rejects_failed_acceptance_conditions(
    mutation, message
):
    baseline, current = cases()
    mutation(baseline, current)
    with pytest.raises(AssertionError, match=message):
        validate_batching_contract("case", baseline, current)
