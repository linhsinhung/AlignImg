#!/usr/bin/env python3
"""Move superseded validation outputs into a recoverable drop directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fnmatch import fnmatch
import json
from pathlib import Path
import time
from typing import Any


DEFAULT_DIRECTORY = Path("validation-results")
KEEP_RULES = (
    ("current 1.10 conformance", "alignimg-1.10-*.json"),
    ("frozen 1.7.1 baseline", "alignimg-1.7.1-*.json"),
    ("1.8 Fourier candidate milestone", "alignimg-1.8-fourier-native-*.json"),
    (
        "1.8 final Fourier workflow milestone",
        "alignimg-1.8-fourier-workflow-*-final.json",
    ),
    ("1.9 release conformance", "alignimg-1.9-*-quick.json"),
    ("1.9 Fourier M-step milestone", "alignimg-1.9-fourier-mstep-*-v2.json"),
    ("active final RF/MRA validation", "re2dc-70s-final-*"),
    ("RELION pose benchmark definition", "re2dc-70s-pose-benchmark-oracle.json"),
    ("cleanup audit record", "cleanup-*.json"),
)
POSE_MILESTONE_PREFIXES = (
    "re2dc-70s-pose-rescue-1.7.1",
    "re2dc-70s-pose-fourier-1.8",
    "re2dc-70s-pose-fourier-mstep-1.9",
    "re2dc-70s-pose-whitened-1.10",
)


def keep_reason(name: str) -> str | None:
    for reason, pattern in KEEP_RULES:
        if fnmatch(name, pattern):
            return reason
    for prefix in POSE_MILESTONE_PREFIXES:
        if name.startswith(prefix) and ".initial." not in name:
            return "RELION pose A/B milestone"
    return None


def inventory(
    directory: Path, *, minimum_age_minutes: float, now: float | None = None
) -> list[dict[str, Any]]:
    current_time = time.time() if now is None else now
    records: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name == "drop" or not path.is_file() or path.is_symlink():
            continue
        stat = path.stat()
        reason = keep_reason(path.name)
        age_minutes = (current_time - stat.st_mtime) / 60.0
        if reason is not None:
            action = "keep"
        elif age_minutes < minimum_age_minutes:
            action = "active"
            reason = f"modified less than {minimum_age_minutes:g} minutes ago"
        else:
            action = "drop"
            reason = "superseded intermediate validation output"
        records.append(
            {
                "name": path.name,
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
            f"{record['name']}  # {record['reason']}"
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
        source = directory / record["name"]
        target = destination / record["name"]
        source.replace(target)
        moved.append({**record, "destination": str(target)})
    manifest = {
        "schema": "alignimg.validation-results-archive.v1",
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
        help="Do not move non-kept files modified more recently than this.",
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
        parser.error(f"validation directory does not exist: {args.directory}")
    return args


def main(args: argparse.Namespace) -> None:
    records = inventory(
        args.directory, minimum_age_minutes=args.minimum_age_minutes
    )
    print_inventory(records)
    if not args.apply:
        print("DRY RUN: no files moved. Add --apply to create validation-results/drop/.")
        return
    destination = apply_archive(args.directory, records)
    print(f"MOVED files into recoverable archive: {destination.resolve()}")


if __name__ == "__main__":
    main(parse_args())
