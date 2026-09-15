"""Workflow-scoped device caches with one shared VRAM budget."""

from __future__ import annotations

import numpy as np

from alignimg._profiling import count as profile_count

from .memory import plan_batch_size


class WorkflowGpuWorkspace:
    """Own scoring/update Fourier caches for one alignment workflow."""

    def __init__(self, cp, config, memory_records):
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        base = plan_batch_size(
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
            memory_fraction=float(config.memory_fraction),
            fixed_bytes=0,
            bytes_per_item=1,
            requested_batch_size=config.batch_size,
        )
        self._cp = cp
        self._config = config
        self._memory_records = memory_records
        self._free_bytes = int(free_bytes)
        self._total_bytes = int(total_bytes)
        self._budget_bytes = int(base.budget_bytes)
        self._reserve_bytes = int(base.reserve_bytes)
        self._cache = {}
        self._source_tokens = {}
        self._roles = {}
        self._committed_bytes = 0
        self._peak_committed_bytes = 0
        self._closed = False
        self._released_bytes = 0
        self._memory_records.append(
            {
                "stage": "workflow_workspace",
                "event": "create",
                "free_bytes": self._free_bytes,
                "total_bytes": self._total_bytes,
                "budget_bytes": self._budget_bytes,
                "reserve_bytes": self._reserve_bytes,
            }
        )

    @staticmethod
    def _source_token(values):
        array = np.asarray(values)
        return id(values), array.shape, array.strides, array.dtype.str

    @staticmethod
    def _is_oom(error: Exception) -> bool:
        return (
            "out of memory" in str(error).lower()
            or error.__class__.__name__ == "OutOfMemoryError"
        )

    def _plan(
        self,
        fixed_bytes: int,
        bytes_per_item: int,
        extra_cache_bytes: int = 0,
        requested_batch_size: int | None = None,
    ):
        return plan_batch_size(
            free_bytes=self._free_bytes,
            total_bytes=self._total_bytes,
            memory_fraction=float(self._config.memory_fraction),
            fixed_bytes=(
                self._committed_bytes + int(fixed_bytes) + int(extra_cache_bytes)
            ),
            bytes_per_item=int(bytes_per_item),
            requested_batch_size=(
                self._config.batch_size
                if requested_batch_size is None
                else requested_batch_size
            ),
        )

    def acquire_fourier(
        self,
        role: str,
        values,
        *,
        fixed_bytes: int,
        bytes_per_item: int,
        requested_batch_size: int | None = None,
        upload,
    ):
        """Return a retained device array or a host-streaming plan."""
        if self._closed:
            raise RuntimeError("GPU workspace is closed")
        if role not in {"scoring", "update"}:
            raise ValueError("workspace Fourier role must be 'scoring' or 'update'")
        token = self._source_token(values)
        if role in self._source_tokens and self._source_tokens[role] != token:
            raise ValueError(f"workspace {role} Fourier source changed within a workflow")
        self._source_tokens.setdefault(role, token)
        state = self._roles.setdefault(
            role,
            {
                "policy": None,
                "cache_bytes": int(np.asarray(values).nbytes),
                "requests": 0,
                "cache_hits": 0,
                "uploads": 0,
                "host_streamed_requests": 0,
                "upload_oom_fallbacks": 0,
                "evictions": 0,
            },
        )
        state["requests"] += 1
        profile_count(f"workspace_{role}_cache_requests")

        if role in self._cache:
            state["cache_hits"] += 1
            profile_count(f"workspace_{role}_cache_hits")
            plan = self._plan(
                fixed_bytes, bytes_per_item, requested_batch_size=requested_batch_size
            )
            record = {
                "stage": "particle_fourier_cache",
                "workspace_role": role,
                "event": "reuse",
                "policy": "device",
                "cache_bytes": state["cache_bytes"],
                **plan.asdict(),
            }
            self._memory_records.append(record)
            return self._cache[role], plan

        if state["policy"] == "host_streamed":
            state["host_streamed_requests"] += 1
            profile_count(f"workspace_{role}_host_streamed_requests")
            plan = self._plan(
                fixed_bytes, bytes_per_item, requested_batch_size=requested_batch_size
            )
            self._memory_records.append(
                {
                    "stage": "particle_fourier_cache",
                    "workspace_role": role,
                    "event": "reuse",
                    "policy": "host_streamed",
                    "cache_bytes": state["cache_bytes"],
                    **plan.asdict(),
                }
            )
            return None, plan

        cache_bytes = state["cache_bytes"]
        cached_plan = self._plan(
            fixed_bytes,
            bytes_per_item,
            cache_bytes,
            requested_batch_size=requested_batch_size,
        )
        cache_fits = (
            cached_plan.budget_bytes
            >= cached_plan.fixed_bytes + cached_plan.bytes_per_item
        )
        if cache_fits:
            try:
                device = upload(values)
            except Exception as error:
                if not self._is_oom(error):
                    raise
                state["upload_oom_fallbacks"] += 1
                profile_count(f"workspace_{role}_upload_oom_fallbacks")
            else:
                self._cache[role] = device
                state["policy"] = "device"
                state["uploads"] += 1
                self._committed_bytes += cache_bytes
                self._peak_committed_bytes = max(
                    self._peak_committed_bytes, self._committed_bytes
                )
                profile_count(f"workspace_{role}_cache_uploads")
                profile_count(f"workspace_{role}_cache_upload_bytes", cache_bytes)
                self._memory_records.append(
                    {
                        "stage": "particle_fourier_cache",
                        "workspace_role": role,
                        "event": "upload",
                        "policy": "device",
                        "cache_bytes": cache_bytes,
                        **cached_plan.asdict(),
                    }
                )
                return device, cached_plan

        state["policy"] = "host_streamed"
        state["host_streamed_requests"] += 1
        profile_count(f"workspace_{role}_host_streamed_requests")
        streamed_plan = self._plan(
            fixed_bytes, bytes_per_item, requested_batch_size=requested_batch_size
        )
        self._memory_records.append(
            {
                "stage": "particle_fourier_cache",
                "workspace_role": role,
                "event": "upload_oom_fallback" if state["upload_oom_fallbacks"] else "plan",
                "policy": "host_streamed",
                "cache_bytes": cache_bytes,
                **streamed_plan.asdict(),
            }
        )
        return None, streamed_plan

    def disable_cache(self, role: str, *, reason: str) -> bool:
        """Evict one live cache so an OOM retry can fall back to streaming."""
        device = self._cache.pop(role, None)
        if device is None:
            return False
        state = self._roles[role]
        cache_bytes = int(state["cache_bytes"])
        del device
        self._committed_bytes -= cache_bytes
        self._released_bytes += cache_bytes
        state["policy"] = "host_streamed"
        state["evictions"] += 1
        profile_count(f"workspace_{role}_cache_evictions")
        self._memory_records.append(
            {
                "stage": "workflow_workspace",
                "event": "evict",
                "workspace_role": role,
                "reason": reason,
                "evicted_bytes": cache_bytes,
            }
        )
        return True

    def close(self):
        if self._closed:
            return
        released_now = self._committed_bytes
        self._released_bytes += released_now
        self._cache.clear()
        self._committed_bytes = 0
        self._closed = True
        profile_count("workspace_release_calls")
        profile_count("workspace_released_bytes", self._released_bytes)
        self._memory_records.append(
            {
                "stage": "workflow_workspace",
                "event": "release",
                "released_bytes": released_now,
                "total_released_bytes": self._released_bytes,
                "release_policy": "return_to_cupy_memory_pool",
            }
        )

    def summary(self):
        return {
            "scope": "single_alignment_workflow",
            "initial_free_bytes": self._free_bytes,
            "device_total_bytes": self._total_bytes,
            "budget_bytes": self._budget_bytes,
            "reserve_bytes": self._reserve_bytes,
            "peak_committed_cache_bytes": self._peak_committed_bytes,
            "released_cache_bytes": self._released_bytes,
            "closed": self._closed,
            "release_policy": "return_to_cupy_memory_pool",
            "roles": {name: dict(state) for name, state in self._roles.items()},
        }
