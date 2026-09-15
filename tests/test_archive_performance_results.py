from __future__ import annotations

import json
import os
from pathlib import Path

from tools.archive_performance_results import apply_archive, inventory, keep_reason


def test_keep_rules_cover_accepted_and_current_records():
    assert keep_reason("stage-0/ACCEPTED.md")
    assert keep_reason("stage-0/baseline-2.0.0/global.npz")
    assert keep_reason("stage-1/dev1-delivery/source.tar.gz")
    assert keep_reason("stage-1/dev1-clean-gpu-cuda-smoke-ab.json")
    assert keep_reason("stage-2/ACCEPTED.md")
    assert keep_reason("stage-2/dev3-cuda-smoke-ab.current.json")
    assert keep_reason("stage-2/dev3-cuda-host-stream-global.json")
    assert keep_reason("stage-2/dev3-delivery/manifest.json")
    assert keep_reason("stage-2/dev3-re2dc-70s-final-n1000.rf.json")
    assert keep_reason("stage-3/dev4-cuda-smoke-ab.current.json")
    assert keep_reason("stage-3/dev4-delivery/manifest.json")
    assert keep_reason("stage-3/dev5-cuda-smoke-ab.current.json")
    assert keep_reason("stage-3/dev5-delivery/manifest.json")
    assert keep_reason("stage-3/dev5-local-known-reference/result.npz")
    assert keep_reason("stage-3/ACCEPTED.md")
    assert keep_reason("stage-4/ACCEPTED.md")
    assert keep_reason("stage-5/ACCEPTED.md")
    assert keep_reason("stage-4/dev6-delivery/manifest.json")
    assert keep_reason("stage-4/dev6-cuda-smoke-ab.current.json")
    assert keep_reason("stage-5/dev7-delivery/manifest.json")
    assert keep_reason("stage-5/dev7-cuda-indexed-smoke-ab.json")
    assert keep_reason("stage-5/dev8-delivery/manifest.json")
    assert keep_reason("stage-5/dev8-cuda-rotation-smoke-ab.json")
    assert keep_reason("stage-5/dev9-delivery/manifest.json")
    assert keep_reason("stage-5/dev9-cuda-fused-smoke-ab.json")
    assert keep_reason("stage-5/dev10-delivery/manifest.json")
    assert keep_reason("stage-5/dev10-cuda-raw-average-scale.json")
    assert keep_reason("stage-6/dev11-delivery/manifest.json")
    assert keep_reason("stage-6/dev11-cuda-mixed-representative-ab.json")
    assert keep_reason("stage-7/2.1.0rc1-delivery/manifest.json")
    assert keep_reason("stage-7/2.1.0-delivery/manifest.json")
    assert keep_reason("stage-7/rc1-final.json")
    assert keep_reason("stage-7/final-cuda-quick.json")
    assert keep_reason("quadratic-refine/BASELINE_2.1.json")
    assert keep_reason("quadratic-refine/dev1-cuda-k1.json")
    assert keep_reason("quadratic-refine/dev1-cuda-k1.inputs.npz")
    assert keep_reason("quadratic-refine/dev1-delivery/source.tar.gz")
    assert keep_reason("quadratic-refine/dev2-delivery/source.tar.gz")
    assert keep_reason("quadratic-refine/dev2-cuda-k1.json")
    assert keep_reason("quadratic-refine/dev3-delivery/source.tar.gz")
    assert keep_reason("quadratic-refine/dev3-cuda-fixed-mra.json")
    assert keep_reason("quadratic-refine/dev4-delivery/source.tar.gz")
    assert keep_reason("quadratic-refine/dev4-cuda-fixed-mra.json")
    assert keep_reason("quadratic-refine/dev5-delivery/source.tar.gz")
    assert keep_reason("quadratic-refine/dev5-cuda-local-n3050-ab.json")
    assert keep_reason(
        "quadratic-refine/dev5-cuda-local-n3050-ab.quadratic_refine.result.npz"
    )
    assert keep_reason("quadratic-refine/2.2.0-delivery/source.tar.gz")
    assert keep_reason("quadratic-refine/final-cuda-quick.json")
    assert keep_reason("quadratic-refine/STAGE2_ACCEPTED.md")
    assert keep_reason("quadratic-refine/STAGE3_DEV3_DIAGNOSIS.md")
    assert keep_reason("stage-2/dev2-cuda-quick.json") is None
    assert keep_reason("stage-2/dev3-cuda-smoke-ab.current.global.result.npz") is None


def test_inventory_is_recursive_and_protects_recent_files(tmp_path: Path):
    stage = tmp_path / "stage-2"
    stage.mkdir()
    old = stage / "dev2-old.json"
    recent = stage / "running.tmp"
    kept = stage / "dev3-cuda-quick.json"
    for path in (old, recent, kept):
        path.write_text("{}", encoding="utf-8")
    os.utime(old, (100.0, 100.0))
    os.utime(kept, (100.0, 100.0))

    records = inventory(tmp_path, minimum_age_minutes=30.0, now=4000.0)
    actions = {record["path"]: record["action"] for record in records}

    assert actions == {
        "stage-2/dev2-old.json": "drop",
        "stage-2/dev3-cuda-quick.json": "keep",
        "stage-2/running.tmp": "active",
    }


def test_apply_archive_preserves_relative_paths(tmp_path: Path):
    dropped = tmp_path / "stage-1" / "old.result.npz"
    kept = tmp_path / "stage-1" / "ACCEPTED.md"
    dropped.parent.mkdir()
    dropped.write_text("old", encoding="utf-8")
    kept.write_text("keep", encoding="utf-8")
    records = inventory(tmp_path, minimum_age_minutes=0.0)

    destination = apply_archive(tmp_path, records)

    assert kept.is_file()
    assert not dropped.exists()
    assert (destination / "stage-1" / dropped.name).read_text(
        encoding="utf-8"
    ) == "old"
    manifest = json.loads(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["moved_file_count"] == 1
    assert manifest["files"][0]["path"] == "stage-1/old.result.npz"
