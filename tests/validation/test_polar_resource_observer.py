import json
import subprocess
from types import SimpleNamespace

import pytest

from tools import polar_resource_observer as observer


XML = """<nvidia_smi_log><gpu id="00000000:81:00.0">
<uuid>GPU-fixture</uuid><fb_memory_usage><used>900 MiB</used>
<free>23000 MiB</free><reserved>N/A</reserved></fb_memory_usage>
<processes>
<process_info><pid>17</pid><type>C</type><process_name>python</process_name>
<used_memory>334 MiB</used_memory></process_info>
<process_info><pid>18</pid><type>G</type><process_name>Xorg</process_name>
<used_memory>146 MiB</used_memory></process_info>
<process_info><pid>19</pid><type>C+G</type><process_name>desktop</process_name>
<used_memory>N/A</used_memory></process_info>
</processes></gpu></nvidia_smi_log>"""


@pytest.mark.parametrize("state", ["captured", "unavailable", "malformed"])
def test_compact_sample_keeps_graphics_and_unknown_values(monkeypatch, state):
    monkeypatch.setattr(
        observer.streaming,
        "device_context_snapshot",
        lambda: {
            "status": "unavailable" if state == "unavailable" else "captured",
            "wall_time_ns": 1,
            "monotonic_ns": 2,
            "xml": "<broken" if state == "malformed" else XML,
            "error": "not installed",
        },
    )
    sample = observer.capture_sample()
    assert sample["query_end_monotonic_ns"] >= sample["monotonic_ns"]
    assert "xml" not in sample  # Do not store hundreds of full-system XML dumps.
    if state != "captured":
        assert sample["status"] == "unavailable" and "error" in sample
        assert "gpus" not in sample
        return
    gpu = sample["gpus"][0]
    assert gpu["uuid"] == "GPU-fixture" and gpu["pci_bus_id"] == "00000000:81:00.0"
    assert gpu["reserved_bytes"] is None and gpu["used_bytes"] == 900 * 1024**2
    assert [p["type"] for p in gpu["processes"]] == ["C", "G", "C+G"]
    assert gpu["processes"][1]["used_bytes"] == 146 * 1024**2
    assert gpu["processes"][2]["used_bytes"] is None


@pytest.mark.parametrize("child_code,sample_ok", [(0, True), (1, True), (0, False)])
def test_one_child_no_retry_original_failure_is_preserved(
    monkeypatch, tmp_path, child_code, sample_ok
):
    launches = []
    output = tmp_path / "observer.json"

    class Child:
        pid = 17
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = child_code
            return child_code

    def launch(command, **kwargs):
        launches.append(command)
        assert kwargs["start_new_session"] and kwargs["stdin"] == subprocess.DEVNULL
        assert "stdout" not in kwargs  # Inherit terminal; no undrained PIPE.
        path = observer.Path(command[command.index("--output") + 1])
        path.write_text(json.dumps({"status": "failed" if child_code else "passed"}))
        return Child()

    monkeypatch.setattr(observer.subprocess, "Popen", launch)
    monkeypatch.setattr(
        observer,
        "capture_sample",
        lambda: {
            "status": "captured" if sample_ok else "unavailable",
            "gpus": [{"processes": [{"pid": 17, "used_bytes": 100}]}],
        },
    )
    assert observer.main(["--backend", "cuda", "--output", str(output)]) == (
        child_code or (0 if sample_ok else 1)
    )
    assert len(launches) == 1 and "--trace-memory" in launches[0]
    report = json.loads(output.read_text())
    assert report["diagnostic_only"] and report["child_pid"] == 17
    assert report["child_exit_code"] == child_code
    assert report["status"] == ("captured" if sample_ok else "incomplete")
    assert report["resource_report_status"] == ("failed" if child_code else "passed")
    with pytest.raises(FileExistsError):
        observer.main(["--backend", "cuda", "--output", str(output)])
    assert len(launches) == 1


def test_deadline_stops_only_owned_child_without_retry(monkeypatch, tmp_path):
    child = SimpleNamespace(pid=12345, returncode=None, poll=lambda: None)
    stopped = []
    monkeypatch.setattr(observer.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(observer, "MAX_SECONDS", 0)
    monkeypatch.setattr(
        observer, "capture_sample", lambda: pytest.fail("past deadline")
    )

    def stop(target):
        assert target is child
        stopped.append(target.pid)
        target.returncode = -15

    monkeypatch.setattr(observer, "stop_child", stop)
    output = tmp_path / "timeout.json"
    assert observer.main(["--backend", "cuda", "--output", str(output)]) == 124
    report = json.loads(output.read_text())
    assert stopped == [12345] and report["status"] == "incomplete"
    assert report["timed_out"] and report["child_exit_code"] == -15


def test_cleanup_escalates_only_own_process_group(monkeypatch):
    signals = []
    waits = []

    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("child", timeout)
        return -9

    child = SimpleNamespace(pid=12345, poll=lambda: None, wait=wait)
    monkeypatch.setattr(
        observer.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    observer.stop_child(child)
    assert signals == [
        (12345, observer.signal.SIGTERM),
        (12345, observer.signal.SIGKILL),
    ]
    assert waits == [5, 5]


@pytest.mark.parametrize("interrupt", [False, True])
def test_partial_monitor_failure_or_sigterm_keeps_incomplete_report(
    monkeypatch, tmp_path, interrupt
):
    output = tmp_path / "partial.json"
    previous_handler = observer.signal.getsignal(observer.signal.SIGTERM)
    stopped = []

    class Child:
        pid = 17
        returncode = None
        calls = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.calls += 1
            if interrupt:
                observer.signal.getsignal(observer.signal.SIGTERM)(
                    observer.signal.SIGTERM, None
                )
            if self.calls == 1:
                raise subprocess.TimeoutExpired("child", timeout)
            self.returncode = 0
            return 0

    child = Child()

    def launch(command, **kwargs):
        observer.Path(command[-1]).write_text(json.dumps({"status": "passed"}))
        return child

    samples = iter(
        [
            {
                "status": "captured",
                "gpus": [{"processes": [{"pid": 17, "used_bytes": 100}]}],
            },
            {"status": "unavailable", "error": "query timeout"},
        ]
    )

    def stop(target):
        stopped.append(target)
        if target.returncode is None:
            target.returncode = -15

    monkeypatch.setattr(observer.subprocess, "Popen", launch)
    monkeypatch.setattr(observer, "capture_sample", lambda: next(samples))
    monkeypatch.setattr(observer, "stop_child", stop)
    assert observer.main(["--backend", "cupy", "--output", str(output)]) == (
        130 if interrupt else 1
    )
    report = json.loads(output.read_text())
    assert report["status"] == "incomplete" and report["child_pid_sample_count"] == 1
    assert report["interrupted"] == interrupt
    assert report["unavailable_sample_count"] == (0 if interrupt else 1)
    assert stopped == [child]
    assert observer.signal.getsignal(observer.signal.SIGTERM) == previous_handler
