from __future__ import annotations

import pytest

from pathlib import Path

from tools.stage4_validation import unpack_baseline, validate_shared_mstep_contract


ROOT = Path(__file__).resolve().parents[1]


def test_stage4_frozen_dev5_baseline_is_available(tmp_path: Path):
    baseline = ROOT / "validation-results/performance/stage-3/dev5-delivery"
    if not baseline.is_dir():
        pytest.skip("requires the accepted dev5 source snapshot")
    manifest = unpack_baseline(baseline, tmp_path, verify_current_native=False)
    assert manifest["alignimg_version"] == "2.1.0.dev5"


def cases():
    baseline = {
        "config": {"reference_update": "fourier"},
        "performance": {
            "counters": {
                "h2d_calls": 300,
                "h2d_bytes": 3_000_000,
                "d2h_calls": 30,
                "d2h_bytes": 300_000,
                "workspace_update_cache_requests": 9,
            },
            "stages": {
                "workflow/alignment_engine/full_reference_update/fourier_transform": {
                    "calls": 40
                },
                "workflow/alignment_engine/halfset_diagnostics/half_a_update/fourier_transform": {
                    "calls": 20
                },
                "workflow/alignment_engine/halfset_diagnostics/half_b_update/fourier_transform": {
                    "calls": 20
                },
            },
        },
    }
    current = {
        "config": {"reference_update": "fourier"},
        "halfset_update_policy": "shared_unnormalized_accumulation",
        "performance": {
            "counters": {
                "shared_mstep_candidate_transforms": 10_000,
                "h2d_calls": 140,
                "h2d_bytes": 1_400_000,
                "d2h_calls": 12,
                "d2h_bytes": 299_000,
                "workspace_update_cache_requests": 3,
            },
            "stages": {
                "workflow/alignment_engine/shared_reference_update/fourier_transform": {
                    "calls": 40
                }
            },
        },
        "gpu_memory_plans": [],
        "gpu_workspace": {
            "closed": True,
            "budget_bytes": 10_000_000,
            "peak_committed_cache_bytes": 1_000_000,
        },
    }
    return baseline, current


def test_stage4_contract_accepts_shared_mstep_with_less_work():
    baseline, current = cases()
    report = validate_shared_mstep_contract("case", baseline, current)

    assert report["mstep_transform_calls_saved"] == 40
    assert report["workspace_update_cache_requests_saved"] == 6
    assert report["h2d_calls_saved"] == 160
    assert report["d2h_calls_saved"] == 18


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda old, new: new.update(halfset_update_policy="separate_reference_updates"),
            "shared half-set policy",
        ),
        (
            lambda old, new: new["performance"]["stages"][
                "workflow/alignment_engine/shared_reference_update/fourier_transform"
            ].update(calls=80),
            "transform calls did not decrease",
        ),
        (
            lambda old, new: new["performance"]["counters"].update(h2d_bytes=3_000_001),
            "h2d_bytes increased",
        ),
        (
            lambda old, new: new["gpu_memory_plans"].append(
                {"event": "oom_retry"}
            ),
            "unexpected OOM retry",
        ),
    ],
)
def test_stage4_contract_rejects_failed_acceptance_conditions(mutation, message):
    baseline, current = cases()
    mutation(baseline, current)
    with pytest.raises(AssertionError, match=message):
        validate_shared_mstep_contract("case", baseline, current)
