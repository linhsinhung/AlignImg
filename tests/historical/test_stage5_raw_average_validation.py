from __future__ import annotations

import argparse

import pytest

from tools.stage5_raw_average_validation import (
    parse_counts,
    timing_review,
    transfer_summary,
)


def profile(counters):
    return {"stages": {"final_raw_average": {"counters": counters}}}


def test_parse_counts_validates_and_sorts():
    assert parse_counts("5000,1000,3000,1000") == (1000, 3000, 5000)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_counts("1000,nope")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_counts("0,1000")


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(0.95, "material_benefit"), (0.951, "no_material_benefit"), (1.05, "no_material_benefit"), (1.051, "regression")],
)
def test_timing_review_has_explicit_five_percent_gate(ratio, expected):
    assert timing_review(ratio) == expected


def test_transfer_summary_requires_n_to_k_download_reduction():
    before = profile({"h2d_bytes": 4096, "d2h_calls": 4, "d2h_bytes": 4096})
    after = profile(
        {
            "h2d_bytes": 5000,
            "d2h_calls": 1,
            "d2h_bytes": 2048,
            "final_raw_aligned_d2h_bytes_avoided": 4096,
        }
    )

    result = transfer_summary(4, 16, 2, before, after)

    assert result["aligned_d2h_bytes_avoided"] == 4096
    assert result["current_d2h_bytes"] == 2048
    assert result["total_transfer_bytes_saved"] == 1144


def test_transfer_summary_rejects_full_aligned_download():
    before = profile({"h2d_bytes": 4096, "d2h_calls": 1, "d2h_bytes": 4096})
    after = profile(
        {
            "h2d_bytes": 4096,
            "d2h_calls": 1,
            "d2h_bytes": 4096,
            "final_raw_aligned_d2h_bytes_avoided": 4096,
        }
    )

    with pytest.raises(AssertionError, match="K class averages"):
        transfer_summary(4, 16, 2, before, after)
