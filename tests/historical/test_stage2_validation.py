import pytest

from tools.stage2_validation import validate_workspace_contract


def cases(iterations=2, batch_size=512):
    requests = {"scoring": iterations, "update": 3 * iterations}
    cache_bytes = 8192
    old_counters = {
        "h2d_calls": 40,
        "h2d_bytes": 100_000,
        "d2h_bytes": 4096,
    }
    counters = {
        "h2d_calls": 34,
        "h2d_bytes": 60_000,
        "d2h_bytes": 4096,
        "workspace_release_calls": 1,
    }
    roles = {}
    plans = []
    for role, count in requests.items():
        roles[role] = {
            "policy": "device",
            "cache_bytes": cache_bytes,
            "requests": count,
            "uploads": 1,
            "cache_hits": count - 1,
            "host_streamed_requests": 0,
            "upload_oom_fallbacks": 0,
            "evictions": 0,
        }
        counters[f"workspace_{role}_cache_uploads"] = 1
        if count > 1:
            counters[f"workspace_{role}_cache_hits"] = count - 1
        plans.append(
            {
                "stage": "particle_fourier_cache",
                "workspace_role": role,
                "batch_size": batch_size,
            }
        )
    baseline = {"performance": {"counters": old_counters}}
    current = {
        "config": {
            "max_iterations": iterations,
            "halfset_diagnostics": True,
            "batch_size": batch_size,
        },
        "performance": {"counters": counters},
        "gpu_memory_plans": plans,
        "gpu_workspace": {
            "closed": True,
            "budget_bytes": 1_000_000,
            "peak_committed_cache_bytes": 2 * cache_bytes,
            "released_cache_bytes": 2 * cache_bytes,
            "roles": roles,
        },
    }
    return baseline, current


def test_stage2_workspace_contract_accepts_reuse_release_and_transfer_reduction():
    baseline, current = cases()
    report = validate_workspace_contract("case", baseline, current)
    assert report["h2d_calls_saved"] == 6
    assert report["h2d_bytes_saved"] == 40_000
    assert report["d2h_bytes_change"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda old, new: new["gpu_workspace"].update(closed=False), "not released"),
        (
            lambda old, new: new["gpu_workspace"]["roles"]["update"].update(
                policy="host_streamed"
            ),
            "not retained",
        ),
        (
            lambda old, new: new["performance"]["counters"].update(h2d_bytes=100_000),
            "payload did not decrease",
        ),
        (
            lambda old, new: new["gpu_memory_plans"].append(
                {"event": "oom_retry"}
            ),
            "unexpected OOM",
        ),
    ],
)
def test_stage2_workspace_contract_rejects_failed_acceptance_conditions(
    mutation, message
):
    baseline, current = cases()
    mutation(baseline, current)
    with pytest.raises(AssertionError, match=message):
        validate_workspace_contract("case", baseline, current)
