"""Opt-in execution accounting; never used for unprofiled throughput timing."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import time


_ACTIVE: ContextVar[ExecutionProfile | None] = ContextVar(
    "alignimg_profile", default=None
)


class ExecutionProfile:
    def __init__(self):
        self.stages = {}
        self.stack = []
        self.counters = {}
        self.iterations = []
        self.cp = None
        self.hook = None
        self.events = []
        self.memory = None
        self.samples = 0

    def attach_cuda(self, cp):
        if self.cp is not None:
            return
        self.cp = cp
        profile = self
        device = int(cp.cuda.Device().id)
        self.memory = {
            "device_id": device,
            "tracked_pool_live_bytes_peak": 0,
            "sampled_device_used_bytes_peak": 0,
            "sampled_pool_used_bytes_peak": 0,
            "sampled_pool_reserved_bytes_peak": 0,
            "sampling_note": (
                "Allocation hook tracks this call's CuPy pooled allocations, including "
                "reused pool blocks; excludes pre-existing arrays and native CUDA/cuFFT "
                "allocations outside that pool. Device/pool samples are lower bounds "
                "on peaks, not a hard VRAM-limit audit. Device usage includes other processes."
            ),
        }

        class AllocationHook(cp.cuda.MemoryHook):
            name = f"alignimg-profile-{id(profile)}"

            def __init__(self):
                self.live = {}
                self.live_bytes = 0

            def malloc_postprocess(self, **kwargs):
                if kwargs["device_id"] == device and kwargs["mem_ptr"]:
                    key = kwargs["pmem_id"]
                    self.live_bytes -= self.live.get(key, 0)
                    self.live[key] = int(kwargs["mem_size"])
                    self.live_bytes += self.live[key]
                    profile.memory["tracked_pool_live_bytes_peak"] = max(
                        profile.memory["tracked_pool_live_bytes_peak"], self.live_bytes
                    )

            def free_postprocess(self, **kwargs):
                if kwargs["device_id"] == device:
                    self.live_bytes -= self.live.pop(kwargs["pmem_id"], 0)

        self.hook = AllocationHook()
        self.hook.__enter__()
        self.sample_memory()

    def sample_memory(self):
        if self.cp is None:
            return
        free, total = self.cp.cuda.runtime.memGetInfo()
        pool = self.cp.get_default_memory_pool()
        sample = {
            "device_used_bytes": int(total - free),
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_reserved_bytes": int(pool.total_bytes()),
        }
        if self.samples == 0:
            self.memory["entry"] = sample
        self.memory["last"] = sample
        self.memory["device_total_bytes"] = int(total)
        for name, value in sample.items():
            key = f"sampled_{name}_peak"
            self.memory[key] = max(self.memory[key], value)
        self.samples += 1
        self.memory["sample_count"] = self.samples

    @contextmanager
    def stage(self, name: str, *, cuda: bool = False):
        path = "/".join([frame["name"] for frame in self.stack] + [name])
        stats = self.stages.setdefault(
            path,
            {
                "calls": 0,
                "wall_seconds": 0.0,
                "exclusive_wall_seconds": 0.0,
                "cuda_event_span_seconds": None,
                "counters": {},
            },
        )
        event = None
        if cuda and self.cp is not None:
            event = self.cp.cuda.Event()
            event.record(self.cp.cuda.get_current_stream())
        frame = {"name": name, "child_seconds": 0.0, "stats": stats}
        self.stack.append(frame)
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.stack.pop()
            stats["calls"] += 1
            stats["wall_seconds"] += elapsed
            stats["exclusive_wall_seconds"] += max(
                0.0, elapsed - frame["child_seconds"]
            )
            if self.stack:
                self.stack[-1]["child_seconds"] += elapsed
            if event is not None:
                stop = self.cp.cuda.Event()
                stop.record(self.cp.cuda.get_current_stream())
                self.events.append((event, stop, stats))
                if len(self.events) >= 128:
                    self.flush_events()
            if not self.stack or stats["calls"] % 64 == 1:
                self.sample_memory()

    def count(self, name: str, value: int = 1):
        self.counters[name] = self.counters.get(name, 0) + int(value)
        for frame in self.stack:
            counters = frame["stats"]["counters"]
            counters[name] = counters.get(name, 0) + int(value)

    def flush_events(self):
        if not self.events:
            return
        self.events[-1][1].synchronize()
        for start, stop, stats in self.events:
            elapsed = float(self.cp.cuda.get_elapsed_time(start, stop)) / 1000.0
            stats["cuda_event_span_seconds"] = (
                stats["cuda_event_span_seconds"] or 0.0
            ) + elapsed
        self.events.clear()

    def close(self):
        try:
            self.flush_events()
            self.sample_memory()
        finally:
            if self.hook is not None:
                self.hook.__exit__(None, None, None)

    def asdict(self):
        return {
            "schema_version": 1,
            "profiled": True,
            "timing_note": (
                "Hierarchical inclusive wall/event spans overlap: do not sum parent "
                "and child stages. Exclusive wall time is not CPU busy time and may "
                "include GPU waits. CUDA events measure instrumented stream spans, "
                "not kernel utilization. Profiling adds overhead/synchronization; "
                "use separate unprofiled runs for throughput. Legacy timers are unchanged."
            ),
            "transfer_note": (
                "Counts explicit AlignImg host/device payloads and native pose copies; "
                "not driver/JIT/library-internal traffic. D2H calls are blocking boundaries."
            ),
            "stages": self.stages,
            "counters": self.counters,
            "iterations": self.iterations,
            "gpu_memory": self.memory,
        }


def current_profile():
    return _ACTIVE.get()


def profile_scope(name: str, *, cuda: bool = False):
    profile = _ACTIVE.get()
    return nullcontext() if profile is None else profile.stage(name, cuda=cuda)


def count(name: str, value: int = 1):
    profile = _ACTIVE.get()
    if profile is not None:
        profile.count(name, value)


def profile_call(name, function, *args, cuda=False, **kwargs):
    profile = _ACTIVE.get()
    if profile is None:
        return function(*args, **kwargs)
    with profile.stage(name, cuda=cuda):
        return function(*args, **kwargs)


def profile_stage(name: str, *, cuda: bool = False):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            return profile_call(name, function, *args, cuda=cuda, **kwargs)

        return wrapped

    return decorate


def profile_workflow(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        config = kwargs.get("config")
        if not getattr(config, "profile_execution", False) or _ACTIVE.get() is not None:
            return function(*args, **kwargs)
        profile = ExecutionProfile()
        token = _ACTIVE.set(profile)
        try:
            with profile.stage("workflow"):
                result = function(*args, **kwargs)
        finally:
            try:
                profile.close()
            finally:
                _ACTIVE.reset(token)
        result.metadata["performance"] = profile.asdict()
        return result

    return wrapped
