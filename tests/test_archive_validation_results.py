from __future__ import annotations

import json
import os
from pathlib import Path

from tools.archive_validation_results import apply_archive, inventory, keep_reason


def test_keep_rules_cover_current_and_frozen_milestones():
    assert keep_reason("alignimg-1.10-cuda-quick.json")
    assert keep_reason("alignimg-1.7.1-cuda-quick.json")
    assert keep_reason("re2dc-70s-final-n1000.rf.json")
    assert keep_reason("re2dc-70s-pose-fourier-mstep-1.9.json")
    assert keep_reason("re2dc-70s-pose-fourier-mstep-1.9.initial.result.npz") is None
    assert keep_reason("alignimg-1.6-cuda-quick.json") is None


def test_inventory_protects_recent_unrecognized_files(tmp_path: Path):
    old = tmp_path / "old-smoke.json"
    recent = tmp_path / "running.tmp"
    kept = tmp_path / "alignimg-1.10-cuda-quick.json"
    for path in (old, recent, kept):
        path.write_text("{}", encoding="utf-8")
    os.utime(old, (100.0, 100.0))
    os.utime(kept, (100.0, 100.0))

    records = inventory(tmp_path, minimum_age_minutes=30.0, now=4000.0)
    actions = {record["name"]: record["action"] for record in records}

    assert actions == {
        "alignimg-1.10-cuda-quick.json": "keep",
        "old-smoke.json": "drop",
        "running.tmp": "active",
    }


def test_apply_archive_moves_only_drop_records(tmp_path: Path):
    dropped = tmp_path / "old.json"
    kept = tmp_path / "alignimg-1.10-cuda-quick.json"
    dropped.write_text("old", encoding="utf-8")
    kept.write_text("keep", encoding="utf-8")
    records = [
        {
            "name": dropped.name,
            "bytes": dropped.stat().st_size,
            "modified_utc": "1970-01-01T00:00:00+00:00",
            "age_minutes": 10.0,
            "action": "drop",
            "reason": "test",
        },
        {
            "name": kept.name,
            "bytes": kept.stat().st_size,
            "modified_utc": "1970-01-01T00:00:00+00:00",
            "age_minutes": 10.0,
            "action": "keep",
            "reason": "test",
        },
    ]

    destination = apply_archive(tmp_path, records)

    assert kept.is_file()
    assert not dropped.exists()
    assert (destination / dropped.name).read_text(encoding="utf-8") == "old"
    manifest = json.loads(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["moved_file_count"] == 1
    assert manifest["files"][0]["name"] == dropped.name
