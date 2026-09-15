#!/usr/bin/env python3
"""Move superseded performance validation outputs into a recoverable drop directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fnmatch import fnmatch
import json
from pathlib import Path
import time
from typing import Any


DEFAULT_DIRECTORY = Path("validation-results/performance")

KEEP_DIRECTORIES = (
    "stage-0/baseline-2.0.0",
    "stage-0/dev0-source-clean-delivery",
    "stage-1/dev1-delivery",
    "stage-2/dev3-delivery",
    "stage-3/dev4-delivery",
    "stage-3/dev5-delivery",
    "stage-3/dev5-local-known-reference",
    "stage-4/dev6-delivery",
    "stage-5/dev7-delivery",
    "stage-5/dev8-delivery",
    "stage-5/dev9-delivery",
    "stage-5/dev10-delivery",
    "stage-6/dev11-delivery",
    "stage-7/2.1.0rc1-delivery",
    "stage-7/2.1.0-delivery",
    "quadratic-refine/dev1-delivery",
    "quadratic-refine/dev2-delivery",
    "quadratic-refine/dev3-delivery",
    "quadratic-refine/dev4-delivery",
    "quadratic-refine/dev5-delivery",
    "quadratic-refine/2.2.0-delivery",
)

KEEP_FILES = {
    "stage-0/ACCEPTED.md",
    "stage-0/alignimg-2.1.0.dev0-cpu-quick.json",
    "stage-0/dev0-cpu-smoke-verified.json",
    "stage-0/dev0-cuda-pose-fixed-mra-rerun.json",
    "stage-0/dev0-cuda-quick.json",
    "stage-0/dev0-cuda-representative.json",
    "stage-0/dev0-cuda-smoke.json",
    "stage-0/dev0-cuda-source-clean-check.json",
    "stage-0/dev0-cupy-smoke.json",
    "stage-0/dev0-delivery-checksums.json",
    "stage-1/ACCEPTED.md",
    "stage-1/dev1-clean-gpu-cuda-quick.json",
    "stage-2/ACCEPTED.md",
    "stage-3/ACCEPTED.md",
    "stage-4/ACCEPTED.md",
    "stage-5/ACCEPTED.md",
    "quadratic-refine/BASELINE_2.1.json",
    "quadratic-refine/STAGE2_ACCEPTED.md",
    "quadratic-refine/STAGE3_DEV3_DIAGNOSIS.md",
}

KEEP_PATTERNS = (
    "stage-1/dev1-clean-gpu-*-ab.json",
    "stage-2/dev3-*.json",
    "stage-2/dev3-re2dc-70s-final-n1000*",
    "stage-3/dev4-*.json",
    "stage-3/dev5-*.json",
    "stage-4/dev6-*.json",
    "stage-5/dev7-*.json",
    "stage-5/dev8-*.json",
    "stage-5/dev9-*.json",
    "stage-5/dev10-*.json",
    "stage-6/dev11-*.json",
    "stage-7/rc1-*.json",
    "stage-7/final-*.json",
    "quadratic-refine/dev1-*.json",
    "quadratic-refine/dev1-*.npz",
    "quadratic-refine/dev1-*.mrcs",
    "quadratic-refine/dev2-*.json",
    "quadratic-refine/dev2-*.npz",
    "quadratic-refine/dev2-*.mrcs",
    "quadratic-refine/dev3-*.json",
    "quadratic-refine/dev3-*.npz",
    "quadratic-refine/dev3-*.mrcs",
    "quadratic-refine/dev4-*.json",
    "quadratic-refine/dev4-*.npz",
    "quadratic-refine/dev4-*.mrcs",
    "quadratic-refine/dev5-*.json",
    "quadratic-refine/dev5-*.npz",
    "quadratic-refine/dev5-*.mrc",
    "quadratic-refine/dev5-*.mrcs",
    "quadratic-refine/final-*.json",
)


def keep_reason(relative_path: str) -> str | None:
    if relative_path in KEEP_FILES:
        return "accepted validation record"
    if any(
        relative_path == directory or relative_path.startswith(f"{directory}/")
        for directory in KEEP_DIRECTORIES
    ):
        return "immutable baseline or delivery snapshot"
    if any(fnmatch(relative_path, pattern) for pattern in KEEP_PATTERNS):
        return "current formal validation record"
    return None


def inventory(
    directory: Path, *, minimum_age_minutes: float, now: float | None = None
) -> list[dict[str, Any]]:
    current_time = time.time() if now is None else now
    records: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative == "drop" or relative.startswith("drop/"):
            continue
        stat = path.stat()
        reason = keep_reason(relative)
        age_minutes = (current_time - stat.st_mtime) / 60.0
        if reason is not None:
            action = "keep"
        elif age_minutes < minimum_age_minutes:
            action = "active"
            reason = f"modified less than {minimum_age_minutes:g} minutes ago"
        else:
            action = "drop"
            reason = "superseded intermediate performance output"
        records.append(
            {
                "path": relative,
                "bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(),
                "age_minutes": age_minutes,
                "action": action,
                "reason": reason,
            }
        )
    return records


def human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def print_inventory(records: list[dict[str, Any]]) -> None:
    for record in records:
        print(
            f"{record['action'].upper():6} {human_size(record['bytes']):>10}  "
            f"{record['path']}  # {record['reason']}"
        )
    counts = {
        action: sum(record["action"] == action for record in records)
        for action in ("keep", "active", "drop")
    }
    drop_bytes = sum(
        record["bytes"] for record in records if record["action"] == "drop"
    )
    print(
        f"SUMMARY keep={counts['keep']} active={counts['active']} "
        f"drop={counts['drop']} drop_size={human_size(drop_bytes)}"
    )


def apply_archive(directory: Path, records: list[dict[str, Any]]) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    destination = directory / "drop" / stamp
    destination.mkdir(parents=True, exist_ok=False)
    moved: list[dict[str, Any]] = []
    for record in records:
        if record["action"] != "drop":
            continue
        source = directory / record["path"]
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        moved.append({**record, "destination": str(target)})
    manifest = {
        "schema": "alignimg.performance-results-archive.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(directory),
        "archive_directory": str(destination),
        "moved_file_count": len(moved),
        "moved_bytes": sum(record["bytes"] for record in moved),
        "files": moved,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument(
        "--minimum-age-minutes",
        type=float,
        default=120.0,
        help="Do not move unrecognized files modified more recently than this.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move DROP files after printing the inventory. Default is dry-run.",
    )
    args = parser.parse_args()
    if args.minimum_age_minutes < 0.0:
        parser.error("--minimum-age-minutes must be non-negative")
    if not args.directory.is_dir():
        parser.error(f"performance directory does not exist: {args.directory}")
    return args


def main(args: argparse.Namespace) -> None:
    records = inventory(
        args.directory, minimum_age_minutes=args.minimum_age_minutes
    )
    print_inventory(records)
    if not args.apply:
        print(
            "DRY RUN: no files moved. Add --apply to create "
            "validation-results/performance/drop/."
        )
        return
    destination = apply_archive(args.directory, records)
    print(f"MOVED files into recoverable archive: {destination.resolve()}")


if __name__ == "__main__":
    main(parse_args())
