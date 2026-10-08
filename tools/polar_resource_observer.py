#!/usr/bin/env python3
"""One bounded resource run with external, diagnostic-only GPU process samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

if __package__:
    from tools import polar_231_streaming_validation as streaming
else:
    import polar_231_streaming_validation as streaming

ROOT = Path(__file__).resolve().parents[1]
MAX_SECONDS = 300
POLL_SECONDS = 0.25


def memory_bytes(value):
    """nvidia-smi reports integer MiB; unavailable is not zero."""
    parts = (value or "").split()
    if len(parts) == 2 and parts[0].isdigit() and parts[1] == "MiB":
        return int(parts[0]) * 1024**2
    return None


def capture_sample():
    sample = streaming.device_context_snapshot()
    sample["query_end_monotonic_ns"] = time.perf_counter_ns()
    sample["query_end_wall_time_ns"] = time.time_ns()
    xml = sample.pop("xml", "")
    if sample["status"] != "captured":
        sample.setdefault("error", sample.get("stderr") or "nvidia-smi unavailable")
        return sample
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        sample.update(status="unavailable", error=str(error))
        return sample
    sample["gpus"] = []
    for gpu in root.findall("gpu"):
        processes = []
        for process in gpu.findall("processes/process_info"):
            pid = process.findtext("pid", "")
            processes.append(
                {
                    "pid": int(pid) if pid.isdigit() else None,
                    "type": process.findtext("type"),
                    "name": process.findtext("process_name"),
                    "used_bytes": memory_bytes(process.findtext("used_memory")),
                }
            )
        sample["gpus"].append(
            {
                "uuid": gpu.findtext("uuid"),
                "pci_bus_id": gpu.get("id"),
                **{
                    f"{key}_bytes": memory_bytes(gpu.findtext(f"fb_memory_usage/{key}"))
                    for key in ("used", "free", "reserved")
                },
                "processes": processes,
            }
        )
    return sample


def stop_child(child):
    """Only signal the process group created by this observer."""
    if child.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


def interrupt_observer(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "cupy"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    resource_output = output.with_name(output.stem + ".resource.json")
    for path in (output, resource_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "tools/polar_resource_validation.py"),
        "--suite",
        "resources",
        "--backend",
        args.backend,
        "--batch-size",
        "512",
        "--trace-memory",
        "--output",
        str(resource_output),
    ]
    report = {
        "diagnostic_only": True,
        "status": "incomplete",
        "backend": args.backend,
        "source": streaming.mirror.source_manifest(),
        "command": command,
        "parent_pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "deadline_seconds": MAX_SECONDS,
        "snapshot_timeout_seconds": 5,
        "cleanup_timeout_seconds": 10,
        "target_poll_seconds": POLL_SECONDS,
        "resource_report": str(resource_output),
        "resource_report_status": "missing",
        "timed_out": False,
        "interrupted": False,
        "samples": [],
        "errors": [],
        "limits": "Sampled device/context memory, not exact allocator peaks or continuous coverage. No gate attribution or budget subtraction.",
    }
    started = time.perf_counter()
    child = None
    previous_term_handler = signal.signal(signal.SIGTERM, interrupt_observer)
    try:
        child = subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.DEVNULL, start_new_session=True
        )
        while child.poll() is None:
            if time.perf_counter() - started >= MAX_SECONDS:
                report["timed_out"] = True
                break
            sample_started = time.perf_counter()
            report["samples"].append(capture_sample())
            remaining = MAX_SECONDS - (time.perf_counter() - started)
            if remaining <= 0:
                report["timed_out"] = True
                break
            delay = max(0, POLL_SECONDS - (time.perf_counter() - sample_started))
            try:
                child.wait(timeout=min(delay, remaining))
            except subprocess.TimeoutExpired:
                pass
    except KeyboardInterrupt:
        report["interrupted"] = True
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if child is not None:
            stop_child(child)
        signal.signal(signal.SIGTERM, previous_term_handler)
        report["child_pid"] = child.pid if child else None
        report["child_exit_code"] = child.returncode if child else None
    if resource_output.exists():
        try:
            report["resource_report_status"] = json.loads(resource_output.read_text())[
                "status"
            ]
            report["resource_report_sha256"] = streaming.mirror.sha256(resource_output)
        except (OSError, ValueError, KeyError) as error:
            report["errors"].append(f"Invalid child report: {error}")
    report["child_pid_sample_count"] = sum(
        sample["status"] == "captured"
        and any(
            p["pid"] == report["child_pid"] and p["used_bytes"] is not None
            for gpu in sample.get("gpus", [])
            for p in gpu["processes"]
        )
        for sample in report["samples"]
    )
    report["unavailable_sample_count"] = sum(
        sample["status"] != "captured" for sample in report["samples"]
    )
    if (
        report["child_pid_sample_count"]
        and not report["unavailable_sample_count"]
        and not report["timed_out"]
        and not report["interrupted"]
        and not report["errors"]
        and report["resource_report_status"] != "missing"
        and report["child_exit_code"] is not None
    ):
        report["status"] = "captured"
    report["wall_seconds"] = time.perf_counter() - started
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        f"DIAGNOSTIC {report['status'].upper()}: {output}; resource={report['resource_report_status']}",
        flush=True,
    )
    if report["interrupted"]:
        return 130
    if report["timed_out"]:
        return 124
    return report["child_exit_code"] or (0 if report["status"] == "captured" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
