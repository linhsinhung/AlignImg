import hashlib
import io
import json
import sys
import tarfile
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from tools import polar_spatial_cache_validation as validation


def frozen_sources():
    exports = "\n".join(
        f"def {prefix}{engine}(*args, **kwargs): return TOKEN"
        for prefix in (
            "run_soft_alignment_",
            "_final_raw_class_averages_",
            "transform_images_",
        )
        for engine in ("cuda", "cupy")
    )
    return {
        "memory": b"TOKEN = 'frozen memory'\n",
        "_polar_memory": b"from .memory import TOKEN\n",
        "_workspace": b"from .memory import TOKEN\n",
        "backend": ("from ._workspace import TOKEN\n" + exports).encode(),
    }


@pytest.mark.parametrize("tamper", [None, "archive", "manifest", "module"])
def test_frozen_snapshot_and_manifest_are_pinned(tmp_path, monkeypatch, tamper):
    sources = frozen_sources()
    files = {
        validation.GPU_PREFIX + name + ".py": hashlib.sha256(value).hexdigest()
        for name, value in sources.items()
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {"files": files, "source_sha256": digest}
    monkeypatch.setattr(validation, "SOURCE_SHA256", digest)
    if tamper == "manifest":
        manifest["source_sha256"] = "invalid"
    if tamper == "module":
        sources["backend"] += b"# altered\n"
    path = tmp_path / "snapshot.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        contents = {"manifest.json": json.dumps(manifest).encode()}
        contents.update(
            {
                validation.GPU_PREFIX + name + ".py": value
                for name, value in sources.items()
            }
        )
        for name, value in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    monkeypatch.setattr(validation, "SNAPSHOT_SHA256", validation.fast.sha256(path))
    if tamper == "archive":
        path.write_bytes(path.read_bytes() + b"changed")
    if tamper:
        with pytest.raises(RuntimeError, match="mismatch"):
            validation.read_frozen_sources(path)
    else:
        loaded, metadata = validation.read_frozen_sources(path)
        assert loaded == sources and metadata == manifest


@pytest.mark.parametrize("engine", ["cuda", "cupy"])
def test_frozen_context_isolates_imports_and_controls_the_correct_planner(
    monkeypatch, engine
):
    import alignimg_gpu

    original_backend = alignimg_gpu.backend
    original_run = getattr(alignimg_gpu, "run_soft_alignment_" + engine)
    monkeypatch.setattr(original_backend, "_native_module", lambda: None)
    with validation.frozen_backend(frozen_sources(), engine) as frozen:
        assert alignimg_gpu.backend is frozen
        assert (
            getattr(alignimg_gpu, "run_soft_alignment_" + engine)() == "frozen memory"
        )
        calls = []

        def planner(*args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                fits_minimum=True,
                batch_size=7,
                budget_bytes=100,
                particle_storage_policy="streaming",
                asdict=lambda: {"batch_size": 7},
            )

        frozen._polar_memory_plan = planner
        records = []
        with validation.resources.controlled_polar_batch(7, records):
            frozen._polar_memory_plan()
        assert calls == [{"force_streaming": True, "particle_limit": 7}]
        assert records == [{"batch_size": 7}]
        assert frozen._polar_memory_plan is planner
    assert alignimg_gpu.backend is original_backend
    assert getattr(alignimg_gpu, "run_soft_alignment_" + engine) is original_run
    assert not any(
        name.startswith("_alignimg_polar_spatial_frozen") for name in sys.modules
    )


def test_only_must_match_suite():
    with pytest.raises(SystemExit):
        validation.parse_args(
            [
                "--suite",
                "smoke",
                "--backend",
                "cuda",
                "--only",
                "local3050",
                "--output",
                "unused.json",
            ]
        )


def test_controlled_cached_batch_does_not_force_streaming(monkeypatch):
    from alignimg_gpu import backend

    calls = []

    def planner(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            fits_minimum=True,
            batch_size=7,
            particle_storage_policy="cached",
            asdict=lambda: {"batch_size": 7},
        )

    monkeypatch.setattr(backend, "_polar_memory_plan", planner)
    records = []
    with validation.controlled_cached_batch(7, records):
        backend._polar_memory_plan()
    assert calls == [{"particle_limit": 7}]
    assert backend._polar_memory_plan is planner and len(records) == 1


def test_cache_evidence_is_not_inferred_from_other_cache_hits():
    entry = {
        "performance": {"counters": {"h2d_bytes": 42, "d2h_bytes": 9}},
        "gpu_workspace": {"roles": {"update": {"cache_hits": 2}}},
    }
    assert validation.cache_evidence(entry)["cache_hits"] == 0
    entry["gpu_workspace"]["roles"]["polar_spatial"] = {"cache_hits": 2, "uploads": 1}
    evidence = validation.cache_evidence(entry)
    assert evidence["cache_hits"] == 2 and evidence["uploads"] == 1
    assert evidence["transfers"] == {"h2d_bytes": 42, "d2h_bytes": 9}


@pytest.mark.parametrize(
    "cache_hit,ratio,status",
    [(True, 0.94, "passed"), (False, 0.9, "uninformative"), (True, 1.06, "failed")],
)
def test_thin_runner_counts_refuses_overwrite_and_preserves_failed_gates(
    tmp_path, monkeypatch, cache_hit, ratio, status
):
    monkeypatch.setattr(validation, "read_frozen_sources", lambda: ({}, {}))
    monkeypatch.setattr(validation, "runtime_identity", lambda *args: ({}, {}))
    monkeypatch.setattr(validation, "frozen_backend", lambda *args: nullcontext())
    monkeypatch.setattr(
        validation.fast,
        "synthetic_cases",
        lambda: {"k1_mirror_off": {"mirror_search": np.array(False)}},
    )
    monkeypatch.setattr(
        validation.resources, "ensure_repeatability", lambda *args: None
    )
    monkeypatch.setattr(validation.resources, "resource_checks", lambda *args: {})
    monkeypatch.setattr(validation, "ensure_cached_repeatability", lambda *args: None)
    calls = []

    def run_case(*args, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(validation.fast, "run_case", run_case)
    monkeypatch.setattr(
        validation,
        "compare_case",
        lambda *args: {
            "within_5_percent_regression_limit": ratio <= 1.05,
            "informative_cache_hit": cache_hit,
            "material_benefit_at_least_5_percent": ratio <= 0.95,
        },
    )
    output = tmp_path / "report.json"
    argv = [
        "--suite",
        "smoke",
        "--backend",
        "cuda",
        "--only",
        "k1_mirror_off",
        "--output",
        str(output),
    ]
    assert validation.main(argv) == (0 if status == "passed" else 1)
    report = json.loads(output.read_text())
    assert report["status"] == status
    assert len(calls) == 2
    assert all(c["measured_repeats"] == 3 and c["profile_execution"] for c in calls)
    with pytest.raises(FileExistsError):
        validation.main(argv)
    assert len(calls) == 2


def test_missing_native_is_failed_not_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "read_frozen_sources", lambda: ({}, {}))

    def unavailable(*args):
        raise RuntimeError("native CUDA extension is missing")

    monkeypatch.setattr(validation, "runtime_identity", unavailable)
    output = tmp_path / "missing.json"
    assert (
        validation.main(
            ["--suite", "smoke", "--backend", "cuda", "--output", str(output)]
        )
        == 1
    )
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and "native CUDA" in report["error"]
