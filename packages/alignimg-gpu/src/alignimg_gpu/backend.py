"""CuPy implementation of AlignImg's GPU-accelerated inner loops."""

from __future__ import annotations

from dataclasses import replace
import platform
import time

import cv2
import numpy as np

from alignimg._adaptive import (
    CELL_DTYPE,
    MAX_ADAPTIVE_PARTICLES_PER_BATCH,
    infer_adaptive_candidates,
)
from alignimg._quadratic import (
    TranslationProfiles,
    _integer_window,
    infer_quadratic_candidates,
)
from alignimg._engine import (
    _SharedReferenceUpdate,
    _center_reference,
    _finalize_spatial_reference_sums,
    _validate_halfset_membership,
    run_soft_alignment_cpu,
)
from alignimg._fourier import (
    _angle_difference,
    _angles_per_reference,
    _candidate_angles,
    soft_circular_mask,
)
from alignimg._geometry import mirror_x_integer_origin, validate_even_square
from alignimg.models import AlignmentConfig, AlignmentResult, PoseSet
from alignimg._profiling import count as profile_count, current_profile, profile_call, profile_scope, profile_stage

from .memory import MemoryPlan, plan_batch_size
from ._workspace import WorkflowGpuWorkspace


def _asdevice(values, **kwargs):
    # Callers already validate CUDA; do not add a device probe per conversion.
    import cupy as cp
    if current_profile() is None or isinstance(values, cp.ndarray):
        return cp.asarray(values, **kwargs)
    with profile_scope("host_to_device", cuda=True):
        output = cp.asarray(values, **kwargs)
        profile_count("h2d_calls")
        profile_count("h2d_bytes", output.nbytes)
        return output


def _ashost(values):
    import cupy as cp
    if current_profile() is None:
        return cp.asnumpy(values)
    with profile_scope("device_to_host", cuda=True):
        output = cp.asnumpy(values)
        profile_count("d2h_calls")
        profile_count("d2h_bytes", output.nbytes)
        return output


def _record_native_pose_upload(count):
    profile_count("native_pose_h2d_calls", 4)
    profile_count("h2d_calls", 4)
    profile_count("h2d_bytes", int(count) * 13)


_TRANSFORM_KERNEL = r"""
extern "C" __global__
void transform_bilinear_wrap(
    const float* images, float* output, const float* angles,
    const float* shifts_y, const float* shifts_x,
    const unsigned char* mirrors, const int count, const int size)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int pixels = size * size;
    if (index >= count * pixels) return;
    const int image = index / pixels;
    const int offset = index - image * pixels;
    const int y = offset / size;
    const int x = offset - y * size;
    const float center = size * 0.5f;
    const float radians = angles[image] * 0.017453292519943295f;
    const float cosine = cosf(radians);
    const float sine = sinf(radians);
    const float out_x = x - shifts_x[image] - center;
    const float out_y = y - shifts_y[image] - center;
    float source_x = cosine * out_x - sine * out_y + center;
    float source_y = sine * out_x + cosine * out_y + center;
    if (mirrors[image]) source_x = 2.0f * center - source_x;
    source_x = fmodf(fmodf(source_x, (float)size) + size, (float)size);
    source_y = fmodf(fmodf(source_y, (float)size) + size, (float)size);
    const int x0 = (int)floorf(source_x);
    const int y0 = (int)floorf(source_y);
    const int x1 = (x0 + 1) % size;
    const int y1 = (y0 + 1) % size;
    const float wx = source_x - x0;
    const float wy = source_y - y0;
    const float* source = images + image * pixels;
    const float top = source[y0 * size + x0] * (1.0f - wx) + source[y0 * size + x1] * wx;
    const float bottom = source[y1 * size + x0] * (1.0f - wx) + source[y1 * size + x1] * wx;
    output[index] = top * (1.0f - wy) + bottom * wy;
}
"""


_FOURIER_TRANSFORM_KERNEL = r"""
__device__ __forceinline__ int wrap_index(const int value, const int size)
{
    const int result = value % size;
    return result < 0 ? result + size : result;
}

extern "C" __global__
void transform_fourier_bilinear(
    const float2* fourier, float2* output, const float* angles,
    const float* shifts_y, const float* shifts_x,
    const unsigned char* mirrors, const int count, const int size)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int pixels = size * size;
    if (index >= count * pixels) return;
    const int image = index / pixels;
    const int offset = index - image * pixels;
    const int y = offset / size;
    const int x = offset - y * size;
    const int ky = y < size / 2 ? y : y - size;
    const int kx = x < size / 2 ? x : x - size;
    const float radians = angles[image] * 0.017453292519943295f;
    const float cosine = cosf(radians);
    const float sine = sinf(radians);
    const float source_y = cosine * ky + sine * kx;
    float source_x = cosine * kx - sine * ky;
    if (mirrors[image]) source_x = -source_x;
    const int floor_y = (int)floorf(source_y);
    const int floor_x = (int)floorf(source_x);
    const int y0 = wrap_index(floor_y, size);
    const int x0 = wrap_index(floor_x, size);
    const int y1 = (y0 + 1) % size;
    const int x1 = (x0 + 1) % size;
    const float wy = source_y - floor_y;
    const float wx = source_x - floor_x;
    const float2* source = fourier + image * pixels;
    float2 v00 = source[y0 * size + x0];
    float2 v01 = source[y0 * size + x1];
    float2 v10 = source[y1 * size + x0];
    float2 v11 = source[y1 * size + x1];
    const float p00 = ((y0 + x0) & 1) ? -1.0f : 1.0f;
    const float p01 = ((y0 + x1) & 1) ? -1.0f : 1.0f;
    const float p10 = ((y1 + x0) & 1) ? -1.0f : 1.0f;
    const float p11 = ((y1 + x1) & 1) ? -1.0f : 1.0f;
    v00.x *= p00; v00.y *= p00;
    v01.x *= p01; v01.y *= p01;
    v10.x *= p10; v10.y *= p10;
    v11.x *= p11; v11.y *= p11;
    float2 top;
    top.x = v00.x * (1.0f - wx) + v01.x * wx;
    top.y = v00.y * (1.0f - wx) + v01.y * wx;
    float2 bottom;
    bottom.x = v10.x * (1.0f - wx) + v11.x * wx;
    bottom.y = v10.y * (1.0f - wx) + v11.y * wx;
    float2 value;
    value.x = top.x * (1.0f - wy) + bottom.x * wy;
    value.y = top.y * (1.0f - wy) + bottom.y * wy;
    const float center_phase = ((y + x) & 1) ? -1.0f : 1.0f;
    value.x *= center_phase;
    value.y *= center_phase;
    const float phase = -6.283185307179586f
        * (shifts_y[image] * ky + shifts_x[image] * kx) / size;
    const float phase_cosine = cosf(phase);
    const float phase_sine = sinf(phase);
    output[index] = make_float2(
        value.x * phase_cosine - value.y * phase_sine,
        value.x * phase_sine + value.y * phase_cosine
    );
}
"""


def _cupy():
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        raise RuntimeError(
            "alignimg-gpu native/CuPy engines support Linux x86-64 only."
        )
    try:
        import cupy as cp
    except ImportError as error:
        raise RuntimeError(
            "alignimg-gpu requires a CuPy distribution compatible with the installed CUDA runtime."
        ) from error
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device was detected")
    except Exception as error:
        raise RuntimeError(f"CUDA initialization failed: {error}") from error
    return cp


def _native_module():
    try:
        from . import _native
    except (ImportError, OSError):
        return None
    return _native


def backend_status() -> dict[str, dict[str, object]]:
    """Report runtime availability without silently selecting another engine."""
    native = _native_module()
    try:
        _cupy()
    except RuntimeError as error:
        cupy_available = False
        cupy_error = str(error)
    else:
        cupy_available = True
        cupy_error = None
    return {
        "cuda": {
            "available": native is not None and cupy_available,
            "description": "Native CUDA indexed Fourier scoring and fused FP64 M-step accumulation with a CuPy controller.",
            "error": None
            if native is not None
            else "native CUDA extension is not installed",
        },
        "cupy": {
            "available": cupy_available,
            "description": "CuPy fallback engine using a runtime-compiled transform kernel.",
            "error": cupy_error,
        },
    }


def _memory_plan(
    config: AlignmentConfig, size: int, reference_count: int = 1
) -> MemoryPlan:
    cp = _cupy()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    pixels = size * size
    # Input/output, complex FFT/cross-power, real correlation and cuFFT workspace.
    bytes_per_candidate = max(1, pixels * 32)
    fixed_bytes = max(0, int(reference_count)) * pixels * 16
    return plan_batch_size(
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
        memory_fraction=float(config.memory_fraction),
        fixed_bytes=fixed_bytes,
        bytes_per_item=bytes_per_candidate,
        requested_batch_size=config.batch_size,
    )


def _batch_limit(config: AlignmentConfig, size: int, reference_count: int = 1) -> int:
    return _memory_plan(config, size, reference_count).batch_size


@profile_stage("particle_fourier_cache")
def _plan_particle_fourier_cache(
    fourier: np.ndarray,
    config: AlignmentConfig,
    *,
    fixed_bytes: int,
    bytes_per_item: int,
):
    """Cache particle DFTs when they fit while preserving the configured reserve."""
    cp = _cupy()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    cache_bytes = int(np.asarray(fourier).nbytes)
    cached_plan = plan_batch_size(
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
        memory_fraction=float(config.memory_fraction),
        fixed_bytes=int(fixed_bytes) + cache_bytes,
        bytes_per_item=int(bytes_per_item),
        requested_batch_size=config.batch_size,
    )
    cache_fits = (
        cached_plan.budget_bytes >= cached_plan.fixed_bytes + cached_plan.bytes_per_item
    )
    if cache_fits:
        return (
            _asdevice(fourier),
            cached_plan,
            {
                "stage": "particle_fourier_cache",
                "policy": "device",
                "cache_bytes": cache_bytes,
                **cached_plan.asdict(),
            },
        )
    streamed_plan = plan_batch_size(
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
        memory_fraction=float(config.memory_fraction),
        fixed_bytes=int(fixed_bytes),
        bytes_per_item=int(bytes_per_item),
        requested_batch_size=config.batch_size,
    )
    return (
        None,
        streamed_plan,
        {
            "stage": "particle_fourier_cache",
            "policy": "host_streamed",
            "cache_bytes": cache_bytes,
            **streamed_plan.asdict(),
        },
    )


def _acquire_particle_fourier(
    fourier,
    config,
    *,
    role,
    fixed_bytes,
    bytes_per_item,
    workspace=None,
    memory_records=None,
):
    if workspace is None:
        device, plan, record = _plan_particle_fourier_cache(
            fourier,
            config,
            fixed_bytes=fixed_bytes,
            bytes_per_item=bytes_per_item,
        )
        if memory_records is not None:
            memory_records.append(record)
        return device, plan
    return profile_call(
        "particle_fourier_cache",
        workspace.acquire_fourier,
        role,
        fourier,
        fixed_bytes=fixed_bytes,
        bytes_per_item=bytes_per_item,
        requested_batch_size=config.batch_size,
        upload=_asdevice,
    )


@profile_stage("spatial_transform", cuda=True)
def _transform_batch_cupy(images, angles, shifts_y, shifts_x, mirrors):
    cp = _cupy()
    source = _asdevice(images, dtype=cp.float32, order="C")
    count, size, width = source.shape
    if size != width:
        raise ValueError("alignimg-gpu requires square images.")
    validate_even_square((int(size), int(width)))
    output = cp.empty_like(source)
    kernel = cp.RawKernel(_TRANSFORM_KERNEL, "transform_bilinear_wrap")
    threads = 256
    blocks = (source.size + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            source,
            output,
            _asdevice(angles, dtype=cp.float32),
            _asdevice(shifts_y, dtype=cp.float32),
            _asdevice(shifts_x, dtype=cp.float32),
            _asdevice(mirrors, dtype=cp.uint8),
            np.int32(count),
            np.int32(size),
        ),
    )
    return output


_NATIVE_SESSIONS: dict[tuple[int, int], object] = {}


def _native_session(cp, native, size: int):
    device_id = int(cp.cuda.Device().id)
    key = (device_id, int(size))
    session = _NATIVE_SESSIONS.get(key)
    if session is None:
        session = native.TransformSession(device_id, int(size))
        _NATIVE_SESSIONS[key] = session
    return session


@profile_stage("spatial_transform", cuda=True)
def _transform_batch_cuda(images, angles, shifts_y, shifts_x, mirrors):
    cp = _cupy()
    native = _native_module()
    if native is None:
        raise RuntimeError(
            "native CUDA extension is unavailable; install a compiled alignimg-gpu wheel "
            "or request backend='cupy'."
        )
    source = _asdevice(images, dtype=cp.float32, order="C")
    count, size, width = source.shape
    if size != width:
        raise ValueError("alignimg-gpu requires square images.")
    validate_even_square((int(size), int(width)))
    output = cp.empty_like(source)
    session = _native_session(cp, native, int(size))
    _record_native_pose_upload(count)
    session.transform_device(
        int(source.data.ptr),
        int(output.data.ptr),
        np.ascontiguousarray(angles, dtype=np.float32),
        np.ascontiguousarray(shifts_y, dtype=np.float32),
        np.ascontiguousarray(shifts_x, dtype=np.float32),
        np.ascontiguousarray(mirrors, dtype=np.uint8),
        int(cp.cuda.get_current_stream().ptr),
    )
    return output


@profile_stage("fourier_transform", cuda=True)
def _transform_fourier_batch_cupy(fourier, angles, shifts_y, shifts_x, mirrors):
    cp = _cupy()
    source = _asdevice(fourier, dtype=cp.complex64, order="C")
    count, size, width = source.shape
    if size != width:
        raise ValueError("alignimg-gpu requires square Fourier images.")
    validate_even_square((int(size), int(width)))
    output = cp.empty_like(source)
    kernel = cp.RawKernel(
        _FOURIER_TRANSFORM_KERNEL,
        "transform_fourier_bilinear",
    )
    threads = 256
    blocks = (source.size + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            source,
            output,
            _asdevice(angles, dtype=cp.float32),
            _asdevice(shifts_y, dtype=cp.float32),
            _asdevice(shifts_x, dtype=cp.float32),
            _asdevice(mirrors, dtype=cp.uint8),
            np.int32(count),
            np.int32(size),
        ),
    )
    return output


@profile_stage("fourier_transform", cuda=True)
def _transform_fourier_batch_cuda(fourier, angles, shifts_y, shifts_x, mirrors):
    cp = _cupy()
    native = _native_module()
    if native is None:
        raise RuntimeError(
            "native CUDA extension is unavailable; install a compiled alignimg-gpu wheel "
            "or request backend='cupy'."
        )
    source = _asdevice(fourier, dtype=cp.complex64, order="C")
    count, size, width = source.shape
    if size != width:
        raise ValueError("alignimg-gpu requires square Fourier images.")
    validate_even_square((int(size), int(width)))
    output = cp.empty_like(source)
    session = _native_session(cp, native, int(size))
    _record_native_pose_upload(count)
    session.transform_fourier_device(
        int(source.data.ptr),
        int(output.data.ptr),
        np.ascontiguousarray(angles, dtype=np.float32),
        np.ascontiguousarray(shifts_y, dtype=np.float32),
        np.ascontiguousarray(shifts_x, dtype=np.float32),
        np.ascontiguousarray(mirrors, dtype=np.uint8),
        int(cp.cuda.get_current_stream().ptr),
    )
    return output


@profile_stage("fourier_transform", cuda=True)
def _transform_fourier_indexed_batch_cuda(
    fourier, particle_indices, angles, shifts_y, shifts_x, mirrors
):
    """Transform candidates directly from a workflow-cached particle FFT stack."""
    cp = _cupy()
    native = _native_module()
    if native is None:
        raise RuntimeError(
            "native CUDA extension is unavailable; install a compiled alignimg-gpu wheel."
        )
    source = _asdevice(fourier, dtype=cp.complex64, order="C")
    if source.ndim != 3:
        raise ValueError("cached Fourier particles must have shape (N, H, H).")
    source_count, size, width = source.shape
    if size != width:
        raise ValueError("alignimg-gpu requires square Fourier images.")
    validate_even_square((int(size), int(width)))
    indices = _asdevice(particle_indices, dtype=cp.int32, order="C")
    if indices.ndim != 1 or int(indices.size) != len(angles):
        raise ValueError("particle indices and poses must have equal length.")
    count = int(indices.size)
    output = cp.empty((count, size, size), dtype=cp.complex64)
    session = _native_session(cp, native, int(size))
    _record_native_pose_upload(count)
    profile_count("native_indexed_fourier_transform_calls")
    profile_count("native_indexed_fourier_transform_candidates", count)
    profile_count("device_gather_bytes_avoided", count * int(size) * int(size) * 8)
    session.transform_fourier_indexed_device(
        int(source.data.ptr),
        int(indices.data.ptr),
        int(output.data.ptr),
        int(source_count),
        np.ascontiguousarray(angles, dtype=np.float32),
        np.ascontiguousarray(shifts_y, dtype=np.float32),
        np.ascontiguousarray(shifts_x, dtype=np.float32),
        np.ascontiguousarray(mirrors, dtype=np.uint8),
        int(cp.cuda.get_current_stream().ptr),
    )
    return output


@profile_stage("native_quadratic_peak", cuda=True)
def _quadratic_peak_cuda(objective_surface, image_size: int):
    """Select bounded peaks and fit independent axes in the native extension."""
    cp = _cupy()
    native = _native_module()
    if native is None:
        raise RuntimeError("native CUDA extension is unavailable")
    objective = cp.ascontiguousarray(objective_surface, dtype=cp.float64)
    if objective.ndim != 3:
        raise ValueError("quadratic objective surface must have shape (N, Y, X)")
    count, height, width = map(int, objective.shape)
    peak_y = cp.empty(count, dtype=cp.int32)
    peak_x = cp.empty(count, dtype=cp.int32)
    offset_y = cp.empty(count, dtype=cp.float64)
    offset_x = cp.empty(count, dtype=cp.float64)
    valid_y = cp.empty(count, dtype=cp.uint8)
    valid_x = cp.empty(count, dtype=cp.uint8)
    reason_y = cp.empty(count, dtype=cp.uint8)
    reason_x = cp.empty(count, dtype=cp.uint8)
    session = _native_session(cp, native, int(image_size))
    session.quadratic_peak_device(
        int(objective.data.ptr),
        int(peak_y.data.ptr),
        int(peak_x.data.ptr),
        int(offset_y.data.ptr),
        int(offset_x.data.ptr),
        int(valid_y.data.ptr),
        int(valid_x.data.ptr),
        int(reason_y.data.ptr),
        int(reason_x.data.ptr),
        count,
        height,
        width,
        int(cp.cuda.get_current_stream().ptr),
    )
    profile_count("native_quadratic_peak_calls")
    profile_count("native_quadratic_peak_surfaces", count)
    return (
        peak_y,
        peak_x,
        offset_y,
        offset_x,
        valid_y.astype(cp.bool_),
        valid_x.astype(cp.bool_),
        reason_y,
        reason_x,
    )


@profile_stage("cupy_quadratic_peak", cuda=True)
def _quadratic_peak_cupy(objective_surface):
    """Select deterministic peaks and fit independent axes with CuPy."""
    cp = _cupy()
    objective = cp.ascontiguousarray(objective_surface, dtype=cp.float64)
    if objective.ndim != 3:
        raise ValueError("quadratic objective surface must have shape (N, Y, X)")
    count, height, width = map(int, objective.shape)
    flat = cp.argmax(objective.reshape(count, -1), axis=1).astype(cp.int32)
    peak_y = flat // width
    peak_x = flat % width
    rows = cp.arange(count, dtype=cp.int32)
    center = objective[rows, peak_y, peak_x]
    interior_y = (peak_y > 0) & (peak_y < height - 1)
    interior_x = (peak_x > 0) & (peak_x < width - 1)
    if height >= 3:
        safe_y = cp.clip(peak_y, 1, height - 2)
        left_y = objective[rows, safe_y - 1, peak_x]
        right_y = objective[rows, safe_y + 1, peak_x]
    else:
        left_y = right_y = center
    if width >= 3:
        safe_x = cp.clip(peak_x, 1, width - 2)
        left_x = objective[rows, peak_y, safe_x - 1]
        right_x = objective[rows, peak_y, safe_x + 1]
    else:
        left_x = right_x = center

    def fit(left, right, interior):
        denominator = left - 2.0 * center + right
        epsilon = 32.0 * np.finfo(np.float64).eps * cp.maximum(
            cp.maximum(cp.abs(left), cp.maximum(cp.abs(center), cp.abs(right))),
            1.0,
        )
        concave = interior & (denominator < -epsilon)
        offset = cp.where(
            concave,
            0.5 * (left - right) / cp.where(concave, denominator, -1.0),
            0.0,
        )
        valid = concave & cp.isfinite(offset) & (cp.abs(offset) <= 0.5 + 1e-12)
        reason = cp.where(
            ~interior,
            1,
            cp.where(~concave, 2, cp.where(~valid, 3, 0)),
        ).astype(cp.uint8)
        return offset, valid, reason

    offset_y, valid_y, reason_y = fit(left_y, right_y, interior_y)
    offset_x, valid_x, reason_x = fit(left_x, right_x, interior_x)
    profile_count("cupy_quadratic_peak_calls")
    profile_count("cupy_quadratic_peak_surfaces", count)
    return (
        peak_y,
        peak_x,
        offset_y,
        offset_x,
        valid_y,
        valid_x,
        reason_y,
        reason_x,
    )


@profile_stage("quadratic_exact_translation_rescore", cuda=True)
def _quadratic_exact_translation_rescore_gpu(
    transformed,
    selected_reference,
    score_weights,
    denominator,
    frequency_y,
    frequency_x,
    integer_y,
    integer_x,
    integer_score,
    integer_objective,
    y_offset,
    x_offset,
    valid_y,
    valid_x,
    *,
    temperature: float,
    center_y: float,
    center_x: float,
    pose_shift_sigma: float,
):
    """Exactly rescore one batch of fitted translation proposals on GPU."""
    cp = _cupy()
    fitted_y = integer_y + cp.where(valid_y, y_offset, 0.0)
    fitted_x = integer_x + cp.where(valid_x, x_offset, 0.0)
    phase = cp.exp(
        -2j
        * cp.pi
        * (
            fitted_y[:, None, None] * frequency_y[None]
            + fitted_x[:, None, None] * frequency_x[None]
        )
    )
    shifted = transformed * phase
    fitted_score = cp.real(
        cp.sum(
            score_weights[None] * shifted * cp.conj(selected_reference)[None],
            axis=(1, 2),
        )
    ) / cp.maximum(denominator, 1e-12)
    fitted_objective = fitted_score / float(temperature)
    fitted_objective -= 0.5 * (
        ((fitted_y - center_y) / pose_shift_sigma) ** 2
        + ((fitted_x - center_x) / pose_shift_sigma) ** 2
    )
    proposed_count = valid_y.astype(cp.int32) + valid_x.astype(cp.int32)
    accept = (proposed_count > 0) & (
        fitted_objective > integer_objective + 1e-12
    )
    return (
        cp.where(accept, fitted_y, integer_y),
        cp.where(accept, fitted_x, integer_x),
        cp.where(accept, fitted_score, integer_score),
        cp.where(accept, fitted_objective, integer_objective),
        proposed_count,
        accept,
        fitted_objective,
    )


@profile_stage("fused_fourier_accumulation", cuda=True)
def _accumulate_fourier_indexed_batch_cuda(
    fourier,
    particle_indices,
    angles,
    shifts_y,
    shifts_x,
    mirrors,
    accumulator_ids,
    weights,
    sums_real,
    sums_imag,
    total_weights,
):
    """Transform indexed candidates and add them directly to FP64 sums."""
    cp = _cupy()
    native = _native_module()
    if native is None:
        raise RuntimeError(
            "native CUDA extension is unavailable; install a compiled alignimg-gpu wheel."
        )
    source = _asdevice(fourier, dtype=cp.complex64, order="C")
    if source.ndim != 3:
        raise ValueError("cached Fourier particles must have shape (N, H, H).")
    source_count, size, width = source.shape
    if size != width:
        raise ValueError("alignimg-gpu requires square Fourier images.")
    validate_even_square((int(size), int(width)))
    if (
        sums_real.shape != sums_imag.shape
        or sums_real.ndim != 3
        or sums_real.shape[1:] != (size, size)
        or sums_real.dtype != cp.float64
        or sums_imag.dtype != cp.float64
    ):
        raise ValueError("Fourier accumulators must be matching FP64 image stacks")
    accumulator_count = int(sums_real.shape[0])
    if total_weights.shape != (accumulator_count,) or total_weights.dtype != cp.float64:
        raise ValueError("total_weights must be an FP64 vector matching accumulators")

    angle_values = np.ascontiguousarray(angles, dtype=np.float32)
    shift_y_values = np.ascontiguousarray(shifts_y, dtype=np.float32)
    shift_x_values = np.ascontiguousarray(shifts_x, dtype=np.float32)
    mirror_values = np.ascontiguousarray(mirrors, dtype=np.uint8)
    accumulator_values = np.ascontiguousarray(accumulator_ids, dtype=np.int32)
    weight_values = np.ascontiguousarray(weights, dtype=np.float64)
    count = len(angle_values)
    if not (
        shift_y_values.shape
        == shift_x_values.shape
        == mirror_values.shape
        == accumulator_values.shape
        == weight_values.shape
        == (count,)
    ):
        raise ValueError("candidate arrays must be one-dimensional and equal")
    if count and (
        int(accumulator_values.min()) < 0
        or int(accumulator_values.max()) >= accumulator_count
    ):
        raise ValueError("accumulator ids are out of range")
    indices = _asdevice(particle_indices, dtype=cp.int32, order="C")
    if indices.ndim != 1 or int(indices.size) != count:
        raise ValueError("particle indices and candidates must have equal length")
    if count == 0:
        return

    session = _native_session(cp, native, int(size))
    _record_native_pose_upload(count)
    profile_count("h2d_calls", 2)
    profile_count("h2d_bytes", count * 12)
    profile_count("native_indexed_fourier_transform_calls")
    profile_count("native_indexed_fourier_transform_candidates", count)
    profile_count("native_fused_fourier_accumulation_calls")
    profile_count("native_fused_fourier_accumulation_candidates", count)
    profile_count("device_gather_bytes_avoided", count * int(size) * int(size) * 8)
    profile_count(
        "fourier_transformed_batch_bytes_avoided",
        count * int(size) * int(size) * 8,
    )
    profile_count(
        "fourier_fp64_scatter_input_bytes_avoided",
        count * int(size) * int(size) * 16,
    )
    session.accumulate_fourier_indexed_device(
        int(source.data.ptr),
        int(indices.data.ptr),
        int(sums_real.data.ptr),
        int(sums_imag.data.ptr),
        int(total_weights.data.ptr),
        int(source_count),
        int(accumulator_count),
        angle_values,
        shift_y_values,
        shift_x_values,
        mirror_values,
        accumulator_values,
        weight_values,
        int(cp.cuda.get_current_stream().ptr),
    )


def _transform_cached_fourier(
    device_fourier,
    host_fourier,
    particle_indices,
    angles,
    shifts_y,
    shifts_x,
    mirrors,
    *,
    fourier_transform_batch,
    indexed_fourier_transform_batch=None,
    device_particle_indices=None,
):
    if device_fourier is not None:
        if indexed_fourier_transform_batch is not None:
            return indexed_fourier_transform_batch(
                device_fourier,
                (
                    particle_indices
                    if device_particle_indices is None
                    else device_particle_indices
                ),
                angles,
                shifts_y,
                shifts_x,
                mirrors,
            )
        source = device_fourier[
            _asdevice(particle_indices)
            if device_particle_indices is None
            else device_particle_indices
        ]
    else:
        source = host_fourier[np.asarray(particle_indices, dtype=np.int32)]
    return fourier_transform_batch(
        source, angles, shifts_y, shifts_x, mirrors
    )


def _transform_fourier_engine(
    fourier: np.ndarray,
    poses: PoseSet,
    transform_batch,
) -> np.ndarray:
    values = np.asarray(fourier, dtype=np.complex64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("fourier must have square shape (N, H, H).")
    validate_even_square(tuple(values.shape[1:]))
    if len(values) != len(poses):
        raise ValueError(
            "poses and Fourier images must contain the same number of items."
        )
    output = transform_batch(
        values,
        poses.angle_deg,
        poses.shift_y_px,
        poses.shift_x_px,
        poses.mirror,
    )
    return _ashost(output).astype(np.complex64, copy=False)


def transform_fourier_cupy(fourier: np.ndarray, poses: PoseSet) -> np.ndarray:
    """Experimental CuPy Fourier-native transform used by conformance tests."""
    return _transform_fourier_engine(fourier, poses, _transform_fourier_batch_cupy)


def transform_fourier_cuda(fourier: np.ndarray, poses: PoseSet) -> np.ndarray:
    """Experimental native-CUDA Fourier transform used by conformance tests."""
    return _transform_fourier_engine(fourier, poses, _transform_fourier_batch_cuda)


def _score_fourier_candidates_engine(
    particle_fourier: np.ndarray,
    reference_fourier: np.ndarray,
    cells: np.ndarray,
    frequency_mask: np.ndarray,
    config: AlignmentConfig,
    transform_batch,
) -> np.ndarray:
    particle = np.asarray(particle_fourier, dtype=np.complex64)
    references = np.asarray(reference_fourier, dtype=np.complex64)
    values = np.asarray(cells, dtype=CELL_DTYPE)
    mask = np.asarray(frequency_mask, dtype=np.float32)
    if particle.ndim != 2:
        raise ValueError("particle_fourier must be two-dimensional.")
    size = validate_even_square(tuple(particle.shape))
    if references.ndim != 3 or references.shape[1:] != particle.shape:
        raise ValueError("reference_fourier must have shape (K, H, H).")
    if mask.shape != particle.shape:
        raise ValueError("frequency_mask must match the Fourier image shape.")
    if len(values) == 0:
        return np.empty(0, dtype=np.float32)
    if np.any(values["reference_index"] < 0) or np.any(
        values["reference_index"] >= len(references)
    ):
        raise ValueError("candidate reference_index is out of range.")

    cp = _cupy()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    plan = plan_batch_size(
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
        memory_fraction=float(config.memory_fraction),
        fixed_bytes=len(references) * size * size * 12,
        bytes_per_item=max(1, size * size * 24),
        requested_batch_size=config.batch_size,
    )
    device_particle = _asdevice(particle)
    device_references = _asdevice(references)
    device_mask = _asdevice(mask)
    reference_norm = cp.sum(
        device_mask[None] * cp.abs(device_references) ** 2,
        axis=(1, 2),
    )
    output = np.empty(len(values), dtype=np.float32)
    for start in range(0, len(values), plan.batch_size):
        stop = min(start + plan.batch_size, len(values))
        batch = values[start:stop]
        count = len(batch)
        transformed = transform_batch(
            cp.broadcast_to(device_particle, (count, size, size)),
            batch["angle_deg"].astype(np.float32),
            batch["shift_y_px"].astype(np.float32),
            batch["shift_x_px"].astype(np.float32),
            batch["mirror"].astype(np.bool_),
        )
        reference_ids = _asdevice(batch["reference_index"], dtype=cp.int32)
        selected_references = device_references[reference_ids]
        numerator = cp.real(
            cp.sum(
                device_mask[None] * transformed * cp.conj(selected_references),
                axis=(1, 2),
            )
        )
        transformed_norm = cp.sum(
            device_mask[None] * cp.abs(transformed) ** 2,
            axis=(1, 2),
        )
        denominator = cp.sqrt(transformed_norm * reference_norm[reference_ids])
        output[start:stop] = _ashost(
            numerator / cp.maximum(denominator, 1e-12)
        ).astype(np.float32)
    return output


def score_fourier_candidates_cupy(
    particle_fourier: np.ndarray,
    reference_fourier: np.ndarray,
    cells: np.ndarray,
    frequency_mask: np.ndarray,
    config: AlignmentConfig,
) -> np.ndarray:
    """Experimental VRAM-bounded CuPy Fourier-native candidate scorer."""
    return _score_fourier_candidates_engine(
        particle_fourier,
        reference_fourier,
        cells,
        frequency_mask,
        config,
        _transform_fourier_batch_cupy,
    )


def score_fourier_candidates_cuda(
    particle_fourier: np.ndarray,
    reference_fourier: np.ndarray,
    cells: np.ndarray,
    frequency_mask: np.ndarray,
    config: AlignmentConfig,
) -> np.ndarray:
    """Experimental VRAM-bounded native-CUDA Fourier candidate scorer."""
    return _score_fourier_candidates_engine(
        particle_fourier,
        reference_fourier,
        cells,
        frequency_mask,
        config,
        _transform_fourier_batch_cuda,
    )


def _transform_images_engine(
    images: np.ndarray, poses: PoseSet, transform_batch
) -> np.ndarray:
    values = np.asarray(images, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("images must have square shape (N, H, H).")
    validate_even_square(tuple(values.shape[1:]))
    if len(values) != len(poses):
        raise ValueError("poses and images must contain the same number of items.")
    output = transform_batch(
        values,
        poses.angle_deg,
        poses.shift_y_px,
        poses.shift_x_px,
        poses.mirror,
    )
    return _ashost(output).astype(np.float32, copy=False)


def transform_images_cupy(images: np.ndarray, poses: PoseSet) -> np.ndarray:
    return _transform_images_engine(images, poses, _transform_batch_cupy)


def transform_images_cuda(images: np.ndarray, poses: PoseSet) -> np.ndarray:
    return _transform_images_engine(images, poses, _transform_batch_cuda)


def transform_images_gpu(images: np.ndarray, poses: PoseSet) -> np.ndarray:
    """Compatibility alias selecting native CUDA when it is installed."""
    transform = (
        _transform_batch_cuda if _native_module() is not None else _transform_batch_cupy
    )
    return _transform_images_engine(images, poses, transform)


def _final_raw_class_averages_gpu_engine(
    images,
    poses,
    assignments,
    weights,
    references,
    config,
    *,
    transform_batch,
    engine,
):
    """Apply final poses and retain only FP64 class sums on the device."""
    cp = _cupy()
    values = np.asarray(images, dtype=np.float32)
    reference_values = np.asarray(references, dtype=np.float32)
    assignment_values = np.asarray(assignments, dtype=np.int32)
    weight_values = np.asarray(weights, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[1] != values.shape[2]
        or reference_values.ndim != 3
        or reference_values.shape[1:] != values.shape[1:]
    ):
        raise ValueError("raw images and references must be matching square stacks")
    validate_even_square(tuple(values.shape[1:]))
    if len(poses) != len(values):
        raise ValueError("poses and raw images must contain the same number of items")
    if assignment_values.shape != (len(values),) or weight_values.shape != (len(values),):
        raise ValueError("assignments and weights must match the raw image count")
    reference_count = len(reference_values)
    if reference_count < 1 or (
        len(assignment_values)
        and (
            int(assignment_values.min()) < 0
            or int(assignment_values.max()) >= reference_count
        )
    ):
        raise ValueError("raw-average assignments are out of range")
    if not np.all(np.isfinite(weight_values)) or np.any(weight_values < 0.0):
        raise ValueError("raw-average weights must be finite and non-negative")

    size = int(values.shape[1])
    pixels = size * size
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    fixed_bytes = (
        len(values) * 12
        + reference_count * pixels * 12
        + pixels * 4
        + reference_count * 8
    )
    plan = plan_batch_size(
        free_bytes=int(free_bytes),
        total_bytes=int(total_bytes),
        memory_fraction=float(config.memory_fraction),
        fixed_bytes=fixed_bytes,
        bytes_per_item=max(1, pixels * 32),
        requested_batch_size=config.batch_size,
    )
    memory_events = [{"stage": "final_raw_average", **plan.asdict()}]
    total_weight_values = np.zeros(reference_count, dtype=np.float64)
    np.add.at(total_weight_values, assignment_values, weight_values)
    nonempty = total_weight_values > 1e-8
    nonempty_indices = np.flatnonzero(nonempty).astype(np.int32)
    limit = int(plan.batch_size)

    while True:
        device_assignments = device_weights = sums = aligned = None
        averages = selected = denominators = mask = output = None
        try:
            device_assignments = _asdevice(assignment_values, dtype=cp.int32)
            device_weights = _asdevice(weight_values, dtype=cp.float64)
            sums = cp.zeros((reference_count, size, size), dtype=cp.float64)
            batches = 0
            for start in range(0, len(values), limit):
                stop = min(start + limit, len(values))
                aligned = transform_batch(
                    values[start:stop],
                    poses.angle_deg[start:stop],
                    poses.shift_y_px[start:stop],
                    poses.shift_x_px[start:stop],
                    poses.mirror[start:stop],
                )
                with profile_scope("raw_weighted_accumulation", cuda=True):
                    cp.add.at(
                        sums,
                        device_assignments[start:stop],
                        aligned.astype(cp.float64)
                        * device_weights[start:stop, None, None],
                    )
                batches += 1

            with profile_scope("raw_average_finalize", cuda=True):
                averages = _asdevice(reference_values, dtype=cp.float32)
                if len(nonempty_indices):
                    selected = _asdevice(nonempty_indices, dtype=cp.int32)
                    denominators = _asdevice(
                        total_weight_values[nonempty], dtype=cp.float64
                    )
                    averages[selected] = (
                        sums[selected] / denominators[:, None, None]
                    ).astype(cp.float32)
                    mask = _asdevice(
                        soft_circular_mask(
                            size, config.mask_radius, config.mask_soft_edge
                        ),
                        dtype=cp.float32,
                    )
                    averages[selected] *= mask[None]
                output = _ashost(averages).astype(np.float32, copy=False)
            profile_count("final_raw_gpu_accumulation_batches", batches)
            profile_count("final_raw_gpu_accumulation_particles", len(values))
            profile_count("final_raw_aligned_d2h_bytes_avoided", values.nbytes)
            profile_count("final_raw_output_d2h_bytes", output.nbytes)
            return output, {
                "class_average_transform_backend": engine,
                "class_average_batch_size": limit,
                "class_average_component_count": reference_count,
                "class_average_empty_components": np.flatnonzero(~nonempty),
                "class_average_gpu_accumulation": "cupy_fp64_device",
                "class_average_gpu_memory_plan": plan.asdict(),
                "class_average_gpu_memory_events": memory_events,
            }
        except Exception as error:
            if not _is_oom(error) or limit == 1:
                raise
            limit = max(1, limit // 2)
            device_assignments = device_weights = sums = aligned = None
            averages = selected = denominators = mask = output = None
            cp.get_default_memory_pool().free_all_blocks()
            memory_events.append(
                {
                    "stage": "final_raw_average",
                    "event": "oom_retry",
                    "batch_size": limit,
                }
            )


def _final_raw_class_averages_cuda(*args):
    return _final_raw_class_averages_gpu_engine(
        *args, transform_batch=_transform_batch_cuda, engine="cuda"
    )


def _final_raw_class_averages_cupy(*args):
    return _final_raw_class_averages_gpu_engine(
        *args, transform_batch=_transform_batch_cupy, engine="cupy"
    )


def _final_raw_class_averages_gpu(*args):
    if _native_module() is not None:
        return _final_raw_class_averages_cuda(*args)
    return _final_raw_class_averages_cupy(*args)


def _gpu_candidate_inference_once(
    particles,
    references,
    config: AlignmentConfig,
    class_priors: np.ndarray,
    temperature: float,
    initial_poses: PoseSet | None,
    *,
    transform_batch=_transform_batch_cupy,
    fourier_transform_batch=_transform_fourier_batch_cupy,
    indexed_fourier_transform_batch=None,
    workspace_fourier_source=None,
    workspace_particle_indices=None,
    workspace=None,
    memory_records=None,
):
    cp = _cupy()
    particle_count = len(particles.spatial)
    reference_count = len(references.spatial)
    top_l = config.top_l
    active_reference_counts = np.count_nonzero(class_priors > 0.0, axis=1)
    angles_per_reference = np.asarray(
        [
            _angles_per_reference(config, int(count))
            for count in active_reference_counts
        ],
        dtype=np.int32,
    )
    translations_per_angle = np.maximum(
        1, np.ceil(top_l / active_reference_counts).astype(np.int32)
    )
    proposals: list[tuple[int, int, float, bool, float, float, int]] = []
    for particle in range(particle_count):
        mirrors = (False, True) if config.mirror_search else (False,)
        for mirrored in mirrors:
            polar = particles.polar[particle]
            if mirrored:
                mirrored_fft = np.fft.fft2(
                    mirror_x_integer_origin(particles.spatial[particle])
                )
                from alignimg._fourier import _polar_stack

                polar = _polar_stack(mirrored_fft[None], config.angle_samples)[0]
            polar_fft = None
            if initial_poses is None:
                if mirrored:
                    profile_count("polar_angular_fft_calls")
                    polar_fft = np.fft.fft(polar, axis=0)
                else:
                    polar_fft = particles.polar_fourier[particle]
            for reference in range(reference_count):
                prior = float(class_priors[particle, reference])
                if prior <= 0:
                    continue
                if initial_poses is None:
                    angles = _candidate_angles(
                        polar,
                        references.polar[reference],
                        int(angles_per_reference[particle]),
                        subject_fft=polar_fft,
                        reference_fft=references.polar_fourier[reference],
                    )
                    center_y = center_x = 0.0
                else:
                    center = float(initial_poses.angle_deg[particle])
                    step = 360.0 / config.angle_samples
                    angle_count = int(angles_per_reference[particle])
                    offsets = (np.arange(angle_count) - (angle_count - 1) / 2) * step
                    angles = [center + float(offset) for offset in offsets]
                    center_y = float(initial_poses.shift_y_px[particle])
                    center_x = float(initial_poses.shift_x_px[particle])
                for angle in angles:
                    candidate_id = (
                        (reference * 2 + int(mirrored)) * config.angle_samples
                    ) + int(round((angle % 360.0) * config.angle_samples / 360.0))
                    proposals.append(
                        (
                            particle,
                            reference,
                            angle,
                            mirrored,
                            center_y,
                            center_x,
                            candidate_id,
                        )
                    )

    per_particle: list[list[tuple]] = [[] for _ in range(particle_count)]
    size = particles.spatial.shape[1]
    score_weight_profiles = _asdevice(particles.score_weight_profiles)
    score_weight_bins = _asdevice(particles.score_weight_bins)
    frequency_mask = _asdevice(particles.frequency_mask)
    weight_fixed_bytes = (
        particles.score_weight_profiles.nbytes
        + particles.score_weight_bins.nbytes
        + particles.frequency_mask.nbytes
    )
    reference_fft = _asdevice(references.fourier)
    shared_reference_norm = None
    if len(score_weight_profiles) == 1:
        with profile_scope("reference_norm", cuda=True):
            weights = frequency_mask * score_weight_profiles[0][score_weight_bins]
            shared_reference_norm = cp.sum(
                weights[None] * cp.abs(reference_fft) ** 2, axis=(1, 2)
            )
            profile_count("reference_norm_evaluations", reference_count)
        del weights
        weight_fixed_bytes += shared_reference_norm.nbytes
    shift_offsets = np.arange(
        -config.translation_range,
        config.translation_range + 0.5 * config.translation_step,
        config.translation_step,
        dtype=np.float32,
    )
    offset_y, offset_x = np.meshgrid(shift_offsets, shift_offsets, indexing="ij")
    offset_y, offset_x = offset_y.ravel(), offset_x.ravel()
    particle_fft = None
    if config.candidate_scoring == "fourier":
        cache_source = (
            particles.fourier
            if workspace_fourier_source is None
            else workspace_fourier_source
        )
        particle_index_map = (
            None
            if workspace_particle_indices is None
            else np.asarray(workspace_particle_indices, dtype=np.int32)
        )
        if particle_index_map is not None and particle_index_map.shape != (
            particle_count,
        ):
            raise ValueError("workspace_particle_indices must have shape (N,)")
        particle_fft, plan = _acquire_particle_fourier(
            cache_source,
            config,
            role="scoring",
            fixed_bytes=reference_count * size * size * 16 + weight_fixed_bytes,
            bytes_per_item=max(1, size * size * 32),
            workspace=workspace,
            memory_records=memory_records,
        )
        batch_limit = plan.batch_size
    else:
        batch_limit = _batch_limit(config, size, reference_count)
    for start in range(0, len(proposals), batch_limit):
        batch = proposals[start : start + batch_limit]
        particle_ids = np.asarray([item[0] for item in batch], dtype=np.int32)
        reference_ids = np.asarray([item[1] for item in batch], dtype=np.int32)
        device_particle_ids = _asdevice(particle_ids)
        profile_ids = (
            device_particle_ids
            if len(score_weight_profiles) != 1
            else cp.zeros(len(batch), dtype=cp.int32)
        )
        score_weights = (
            frequency_mask[None]
            * score_weight_profiles[profile_ids][:, score_weight_bins]
        )
        angles = np.asarray([item[2] for item in batch], dtype=np.float32)
        mirrors = np.asarray([item[3] for item in batch], dtype=np.bool_)
        if config.candidate_scoring == "fourier":
            cache_particle_ids = (
                particle_ids
                if particle_index_map is None
                else particle_index_map[particle_ids]
            )
            transformed_fft = _transform_cached_fourier(
                particle_fft,
                cache_source,
                cache_particle_ids,
                angles,
                np.zeros(len(batch), dtype=np.float32),
                np.zeros(len(batch), dtype=np.float32),
                mirrors,
                fourier_transform_batch=fourier_transform_batch,
                indexed_fourier_transform_batch=indexed_fourier_transform_batch,
                device_particle_indices=(
                    device_particle_ids if particle_index_map is None else None
                ),
            )
        else:
            rotated = transform_batch(
                particles.spatial[particle_ids],
                angles,
                np.zeros(len(batch), dtype=np.float32),
                np.zeros(len(batch), dtype=np.float32),
                mirrors,
            )
            transformed_fft = cp.fft.fft2(rotated, axes=(-2, -1))
        with profile_scope("fourier_ncc", cuda=True):
            device_reference_ids = _asdevice(reference_ids)
            selected_reference_fft = reference_fft[device_reference_ids]
            cross_power = (
                score_weights * transformed_fft * cp.conj(selected_reference_fft)
            )
            correlation = cp.fft.ifft2(cross_power, axes=(-2, -1)).real * (size * size)
            transformed_norm = cp.sum(
                score_weights * cp.abs(transformed_fft) ** 2, axis=(1, 2)
            )
            if shared_reference_norm is None:
                reference_norm = cp.sum(
                    score_weights * cp.abs(selected_reference_fft) ** 2,
                    axis=(1, 2),
                )
                profile_count("reference_norm_evaluations", len(batch))
            else:
                reference_norm = shared_reference_norm[device_reference_ids]
                profile_count("reference_norm_cache_hits", len(batch))
            denominator = cp.sqrt(transformed_norm * reference_norm)
            shifts_y = np.stack([item[4] + offset_y for item in batch])
            shifts_x = np.stack([item[5] + offset_x for item in batch])
            iy = _asdevice(np.rint(-shifts_y).astype(np.int32) % size)
            ix = _asdevice(np.rint(-shifts_x).astype(np.int32) % size)
            rows = cp.arange(len(batch), dtype=cp.int32)[:, None]
            scores = correlation[rows, iy, ix] / cp.maximum(denominator[:, None], 1e-12)
            host_scores = _ashost(scores)
        for row, proposal in enumerate(batch):
            particle, reference, angle, mirrored, _, _, candidate_id = proposal
            order = np.argsort(-host_scores[row], kind="stable")[
                : int(translations_per_angle[particle])
            ]
            for translation_index in order:
                dy = float(shifts_y[row, translation_index])
                dx = float(shifts_x[row, translation_index])
                score = float(host_scores[row, translation_index])
                log_posterior = score / temperature + np.log(
                    max(float(class_priors[particle, reference]), 1e-30)
                )
                if initial_poses is not None:
                    da = _angle_difference(angle, initial_poses.angle_deg[particle])
                    ddy = dy - float(initial_poses.shift_y_px[particle])
                    ddx = dx - float(initial_poses.shift_x_px[particle])
                    log_posterior -= 0.5 * (da / config.pose_angle_sigma) ** 2
                    log_posterior -= 0.5 * (
                        (ddy / config.pose_shift_sigma) ** 2
                        + (ddx / config.pose_shift_sigma) ** 2
                    )
                per_particle[particle].append(
                    (
                        log_posterior,
                        score,
                        reference,
                        angle,
                        dy,
                        dx,
                        mirrored,
                        candidate_id,
                    )
                )

    shape = (particle_count, top_l)
    result = {
        "reference_index": np.zeros(shape, dtype=np.int32),
        "angle_deg": np.zeros(shape, dtype=np.float32),
        "shift_y_px": np.zeros(shape, dtype=np.float32),
        "shift_x_px": np.zeros(shape, dtype=np.float32),
        "mirror": np.zeros(shape, dtype=np.bool_),
        "score": np.full(shape, -np.inf, dtype=np.float32),
        "posterior": np.zeros(shape, dtype=np.float32),
    }
    for particle, candidates in enumerate(per_particle):
        if not candidates:
            raise ValueError(
                f"particle {particle} has no allowed reference candidates."
            )
        candidates.sort(key=lambda item: (-item[0], item[7]))
        selected = candidates[:top_l]
        logits = np.asarray([item[0] for item in selected], dtype=np.float64)
        posterior = np.exp(logits - logits.max())
        posterior /= posterior.sum()
        for column, item in enumerate(selected):
            result["reference_index"][particle, column] = item[2]
            result["angle_deg"][particle, column] = (item[3] + 180.0) % 360.0 - 180.0
            result["shift_y_px"][particle, column] = item[4]
            result["shift_x_px"][particle, column] = item[5]
            result["mirror"][particle, column] = item[6]
            result["score"][particle, column] = item[1]
            result["posterior"][particle, column] = posterior[column]
    return result


def _gpu_adaptive_candidate_inference_once(
    particles,
    references,
    config: AlignmentConfig,
    class_priors: np.ndarray,
    temperature: float,
    initial_poses: PoseSet | None,
    *,
    rescue_mask: np.ndarray | None,
    transform_batch=_transform_batch_cupy,
    fourier_transform_batch=_transform_fourier_batch_cupy,
    indexed_fourier_transform_batch=None,
    workspace=None,
    memory_records=None,
):
    cp = _cupy()
    particle_count = len(particles.spatial)
    size = int(particles.spatial.shape[1])
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    weight_fixed_bytes = (
        particles.score_weight_profiles.nbytes
        + particles.score_weight_bins.nbytes
        + particles.frequency_mask.nbytes
        + size * size * 4
        + len(references.spatial) * 4
        + len(particles.score_weight_profiles) * len(references.spatial) * 4
    )
    particle_fft = None
    if config.candidate_scoring == "fourier":
        particle_fft, plan = _acquire_particle_fourier(
            particles.fourier,
            config,
            role="scoring",
            fixed_bytes=len(references.spatial) * size * size * 16
            + weight_fixed_bytes,
            bytes_per_item=max(1, size * size * 64),
            workspace=workspace,
            memory_records=memory_records,
        )
    else:
        plan = plan_batch_size(
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
            memory_fraction=float(config.memory_fraction),
            fixed_bytes=len(references.spatial) * size * size * 16
            + weight_fixed_bytes,
            bytes_per_item=max(1, size * size * 64),
            requested_batch_size=config.batch_size,
        )
    if memory_records is not None:
        memory_records.append({"stage": "adaptive_flat_scoring", **plan.asdict()})
        memory_records.append(
            {
                "stage": "adaptive_particle_batching",
                "particle_batch_size": min(
                    particle_count,
                    (
                        MAX_ADAPTIVE_PARTICLES_PER_BATCH
                        if config.batch_size is None
                        else int(config.batch_size)
                    ),
                    MAX_ADAPTIVE_PARTICLES_PER_BATCH,
                ),
                "candidate_batch_size": plan.batch_size,
            }
        )
    frequency = _asdevice(np.fft.fftfreq(size), dtype=cp.float32)
    frequency_y, frequency_x = cp.meshgrid(frequency, frequency, indexing="ij")
    score_weight_profiles = _asdevice(
        particles.score_weight_profiles, dtype=cp.float32
    )
    score_weight_bins = _asdevice(particles.score_weight_bins)
    frequency_mask = _asdevice(particles.frequency_mask, dtype=cp.float32)
    reference_fft = _asdevice(references.fourier)

    profile_count_value = len(score_weight_profiles)
    reference_norm = cp.empty(
        (profile_count_value, len(reference_fft)), dtype=cp.float32
    )
    reference_norm_ready = np.zeros(profile_count_value, dtype=np.bool_)
    shared_score_weights = None

    def score_candidate_batches(
        _,
        __,
        particle_indices: np.ndarray,
        cell_batches: list[np.ndarray],
    ) -> list[np.ndarray]:
        nonlocal shared_score_weights
        indices = np.asarray(particle_indices, dtype=np.int32)
        if indices.ndim != 1 or len(indices) != len(cell_batches):
            raise ValueError("adaptive particle indices and cell batches must match")
        profile_indices = (
            np.zeros(len(indices), dtype=np.int32)
            if profile_count_value == 1
            else indices
        )
        profile_count("reference_norm_requests", len(profile_indices))
        for profile_index_value in profile_indices:
            profile_index = int(profile_index_value)
            if reference_norm_ready[profile_index]:
                profile_count("reference_norm_cache_hits")
                continue
            with profile_scope("reference_norm", cuda=True):
                score_weights = (
                    frequency_mask
                    * score_weight_profiles[profile_index][score_weight_bins]
                )
                reference_norm[profile_index] = cp.sum(
                    score_weights[None] * cp.abs(reference_fft) ** 2,
                    axis=(1, 2),
                )
                profile_count("reference_norm_evaluations", len(reference_fft))
            reference_norm_ready[profile_index] = True
            if profile_count_value == 1:
                shared_score_weights = score_weights

        lengths = np.asarray([len(values) for values in cell_batches], dtype=np.int64)
        if np.any(lengths == 0):
            raise ValueError("adaptive cell batches must be non-empty")
        values = np.concatenate(
            [np.asarray(batch, dtype=CELL_DTYPE) for batch in cell_batches]
        )
        flat_particle_indices = np.repeat(indices, lengths)
        output = np.empty(len(values), dtype=np.float32)
        for start in range(0, len(values), plan.batch_size):
            stop = min(start + plan.batch_size, len(values))
            batch = values[start:stop]
            batch_particle_indices = flat_particle_indices[start:stop]
            count = len(batch)
            device_particle_indices = _asdevice(
                batch_particle_indices, dtype=cp.int32
            )
            if config.candidate_scoring == "fourier":
                shifted = _transform_cached_fourier(
                    particle_fft,
                    particles.fourier,
                    batch_particle_indices,
                    batch["angle_deg"].astype(np.float32),
                    batch["shift_y_px"].astype(np.float32),
                    batch["shift_x_px"].astype(np.float32),
                    batch["mirror"].astype(np.bool_),
                    fourier_transform_batch=fourier_transform_batch,
                    indexed_fourier_transform_batch=indexed_fourier_transform_batch,
                    device_particle_indices=device_particle_indices,
                )
            else:
                rotated = transform_batch(
                    particles.spatial[batch_particle_indices],
                    batch["angle_deg"].astype(np.float32),
                    np.zeros(count, dtype=np.float32),
                    np.zeros(count, dtype=np.float32),
                    batch["mirror"].astype(np.bool_),
                )
                transformed_fft = cp.fft.fft2(rotated, axes=(-2, -1))
                shift_y = _asdevice(batch["shift_y_px"], dtype=cp.float32)
                shift_x = _asdevice(batch["shift_x_px"], dtype=cp.float32)
                phase = cp.exp(
                    -2j
                    * cp.pi
                    * (
                        shift_y[:, None, None] * frequency_y[None]
                        + shift_x[:, None, None] * frequency_x[None]
                    )
                )
                shifted = transformed_fft * phase
            with profile_scope("fourier_ncc", cuda=True):
                reference_indices = _asdevice(
                    batch["reference_index"], dtype=cp.int32
                )
                selected_reference = reference_fft[reference_indices]
                if profile_count_value == 1:
                    score_weights = shared_score_weights[None]
                    candidate_reference_norm = reference_norm[0, reference_indices]
                else:
                    score_weights = (
                        frequency_mask[None]
                        * score_weight_profiles[device_particle_indices][
                            :, score_weight_bins
                        ]
                    )
                    candidate_reference_norm = reference_norm[
                        device_particle_indices, reference_indices
                    ]
                numerator = cp.real(
                    cp.sum(
                        score_weights * shifted * cp.conj(selected_reference),
                        axis=(1, 2),
                    )
                )
                transformed_norm = cp.sum(
                    score_weights * cp.abs(shifted) ** 2,
                    axis=(1, 2),
                )
                denominator = cp.sqrt(
                    transformed_norm * candidate_reference_norm
                )
                output[start:stop] = _ashost(
                    numerator / cp.maximum(denominator, 1e-12)
                ).astype(np.float32)
            profile_count("adaptive_scoring_batches")
            profile_count("adaptive_scored_candidates", count)
            if len(np.unique(batch_particle_indices)) > 1:
                profile_count("adaptive_cross_particle_batches")

        offsets = np.concatenate(([0], np.cumsum(lengths)))
        return [
            output[int(offsets[index]) : int(offsets[index + 1])]
            for index in range(len(lengths))
        ]

    def proposal_fallback(*args):
        rescue_indices = np.flatnonzero(np.asarray(rescue_mask, dtype=np.bool_))
        return _gpu_candidate_inference_once(
            *args,
            transform_batch=transform_batch,
            fourier_transform_batch=fourier_transform_batch,
            indexed_fourier_transform_batch=indexed_fourier_transform_batch,
            workspace_fourier_source=particles.fourier,
            workspace_particle_indices=rescue_indices,
            workspace=workspace,
            memory_records=memory_records,
        )

    return infer_adaptive_candidates(
        particles,
        references,
        config,
        class_priors,
        temperature,
        initial_poses,
        score_candidate_batches=score_candidate_batches,
        rescue_mask=rescue_mask,
        proposal_fallback=proposal_fallback,
    )


def _gpu_quadratic_candidate_inference_once(
    particles,
    references,
    config: AlignmentConfig,
    class_priors: np.ndarray,
    temperature: float,
    initial_poses: PoseSet | None,
    *,
    rescue_mask: np.ndarray | None,
    fourier_transform_batch=_transform_fourier_batch_cupy,
    indexed_fourier_transform_batch=None,
    workspace=None,
    memory_records=None,
):
    """Run correlation-map translation fitting without downloading the maps."""
    if initial_poses is None:
        raise ValueError("quadratic_refine search requires initial_poses")
    cp = _cupy()
    particle_count = len(particles.spatial)
    reference_count = len(references.spatial)
    size = int(particles.spatial.shape[1])
    weight_fixed_bytes = (
        particles.score_weight_profiles.nbytes
        + particles.score_weight_bins.nbytes
        + particles.frequency_mask.nbytes
        + reference_count * size * size * 8
    )
    particle_fft, plan = _acquire_particle_fourier(
        particles.fourier,
        config,
        role="scoring",
        fixed_bytes=reference_count * size * size * 16 + weight_fixed_bytes,
        bytes_per_item=max(1, size * size * 40),
        workspace=workspace,
        memory_records=memory_records,
    )
    map_chunk_size = max(1, int(plan.batch_size))
    if memory_records is not None:
        memory_records.append(
            {
                "stage": "quadratic_correlation_maps",
                "map_chunk_size": map_chunk_size,
                "requested_particle_batch_size": config.batch_size,
                **plan.asdict(),
            }
        )

    reference_fft = _asdevice(references.fourier, dtype=cp.complex64)
    frequency_mask = _asdevice(particles.frequency_mask, dtype=cp.float32)
    score_weight_profiles = _asdevice(
        particles.score_weight_profiles, dtype=cp.float32
    )
    score_weight_bins = _asdevice(particles.score_weight_bins)
    profile_count_value = len(score_weight_profiles)
    reference_norm = cp.empty(
        (profile_count_value, reference_count), dtype=cp.float32
    )
    reference_norm_ready = np.zeros(profile_count_value, dtype=np.bool_)
    frequency = _asdevice(np.fft.fftfreq(size), dtype=cp.float32)
    frequency_y, frequency_x = cp.meshgrid(frequency, frequency, indexing="ij")

    def translation_profiler(
        _,
        __,
        particle_index: int,
        reference_index: int,
        angles: np.ndarray,
        mirror: bool,
        attempt: AlignmentConfig,
        current_temperature: float,
        pose_prior: PoseSet,
    ) -> TranslationProfiles:
        angle_values = np.asarray(angles, dtype=np.float64)
        center_y = float(pose_prior.shift_y_px[particle_index])
        center_x = float(pose_prior.shift_x_px[particle_index])
        shifts_y = _integer_window(center_y, attempt.local_shift_range)
        shifts_x = _integer_window(center_x, attempt.local_shift_range)
        profile_index = 0 if profile_count_value == 1 else int(particle_index)
        score_weights = (
            frequency_mask
            * score_weight_profiles[profile_index][score_weight_bins]
        )
        if not reference_norm_ready[profile_index]:
            with profile_scope("reference_norm", cuda=True):
                reference_norm[profile_index] = cp.sum(
                    score_weights[None] * cp.abs(reference_fft) ** 2,
                    axis=(1, 2),
                )
                profile_count("reference_norm_evaluations", reference_count)
            reference_norm_ready[profile_index] = True
        else:
            profile_count("reference_norm_cache_hits")

        output_y = []
        output_x = []
        output_score = []
        output_objective = []
        totals = np.zeros(7, dtype=np.float64)
        for start in range(0, len(angle_values), map_chunk_size):
            stop = min(start + map_chunk_size, len(angle_values))
            batch_angles = angle_values[start:stop].astype(np.float32)
            count = len(batch_angles)
            particle_indices = np.full(count, particle_index, dtype=np.int32)
            device_particle_indices = _asdevice(
                particle_indices, dtype=cp.int32
            )
            transformed = _transform_cached_fourier(
                particle_fft,
                particles.fourier,
                particle_indices,
                batch_angles,
                np.zeros(count, dtype=np.float32),
                np.zeros(count, dtype=np.float32),
                np.full(count, mirror, dtype=np.bool_),
                fourier_transform_batch=fourier_transform_batch,
                indexed_fourier_transform_batch=indexed_fourier_transform_batch,
                device_particle_indices=device_particle_indices,
            )
            selected_reference = reference_fft[reference_index]
            with profile_scope("quadratic_correlation_map_ifft", cuda=True):
                cross_power = (
                    score_weights[None]
                    * transformed
                    * cp.conj(selected_reference)[None]
                )
                correlation = (
                    cp.fft.ifft2(cross_power, axes=(-2, -1)).real
                    * (size * size)
                )
                transformed_norm = cp.sum(
                    score_weights[None] * cp.abs(transformed) ** 2,
                    axis=(1, 2),
                )
                denominator = cp.sqrt(
                    transformed_norm
                    * reference_norm[profile_index, reference_index]
                )
                device_y = _asdevice(shifts_y, dtype=cp.int32)
                device_x = _asdevice(shifts_x, dtype=cp.int32)
                rows = cp.arange(count, dtype=cp.int32)[:, None, None]
                score_surface = correlation[
                    rows,
                    ((-device_y) % size)[None, :, None],
                    ((-device_x) % size)[None, None, :],
                ] / cp.maximum(denominator[:, None, None], 1e-12)
                objective_surface = score_surface / float(current_temperature)
                objective_surface -= 0.5 * (
                    (device_y.astype(cp.float64) - center_y)
                    / float(attempt.pose_shift_sigma)
                )[None, :, None] ** 2
                objective_surface -= 0.5 * (
                    (device_x.astype(cp.float64) - center_x)
                    / float(attempt.pose_shift_sigma)
                )[None, None, :] ** 2
                if indexed_fourier_transform_batch is not None:
                    (
                        peak_y_index,
                        peak_x_index,
                        y_offset,
                        x_offset,
                        valid_y,
                        valid_x,
                        reason_y,
                        reason_x,
                    ) = _quadratic_peak_cuda(objective_surface, size)
                else:
                    (
                        peak_y_index,
                        peak_x_index,
                        y_offset,
                        x_offset,
                        valid_y,
                        valid_x,
                        reason_y,
                        reason_x,
                    ) = _quadratic_peak_cupy(objective_surface)
                interior_y = reason_y != 1
                interior_x = reason_x != 1
                concave_y = (reason_y != 1) & (reason_y != 2)
                concave_x = (reason_x != 1) & (reason_x != 2)
                batch_rows = cp.arange(count, dtype=cp.int32)
                integer_y = device_y[peak_y_index].astype(cp.float64)
                integer_x = device_x[peak_x_index].astype(cp.float64)
                integer_score = score_surface[
                    batch_rows, peak_y_index, peak_x_index
                ]
                integer_objective = objective_surface[
                    batch_rows, peak_y_index, peak_x_index
                ]

                (
                    accepted_y,
                    accepted_x,
                    accepted_score,
                    accepted_objective,
                    proposed_count,
                    accept,
                    fitted_objective,
                ) = _quadratic_exact_translation_rescore_gpu(
                    transformed,
                    selected_reference,
                    score_weights,
                    denominator,
                    frequency_y,
                    frequency_x,
                    integer_y,
                    integer_x,
                    integer_score,
                    integer_objective,
                    y_offset,
                    x_offset,
                    valid_y,
                    valid_x,
                    temperature=float(current_temperature),
                    center_y=center_y,
                    center_x=center_x,
                    pose_shift_sigma=float(attempt.pose_shift_sigma),
                )
                summary = cp.stack(
                    (
                        cp.sum(interior_y) + cp.sum(interior_x),
                        cp.sum(proposed_count * accept.astype(cp.int32)),
                        cp.sum(~interior_y) + cp.sum(~interior_x),
                        cp.sum(interior_y & ~concave_y)
                        + cp.sum(interior_x & ~concave_x),
                        cp.sum(concave_y & ~valid_y)
                        + cp.sum(concave_x & ~valid_x),
                        cp.sum(proposed_count * (~accept).astype(cp.int32)),
                        cp.sum(
                            cp.where(
                                accept,
                                fitted_objective - integer_objective,
                                0.0,
                            )
                        ),
                    )
                )
                host_values = _ashost(
                    cp.stack(
                        (accepted_y, accepted_x, accepted_score, accepted_objective)
                    )
                )
                totals += _ashost(summary)
            output_y.append(host_values[0])
            output_x.append(host_values[1])
            output_score.append(host_values[2])
            output_objective.append(host_values[3])
            profile_count("quadratic_map_batches")
            profile_count("quadratic_correlation_map_iffts", count)
            profile_count("quadratic_screened_angles", count)

        return TranslationProfiles(
            angle_deg=angle_values,
            shift_y_px=np.concatenate(output_y),
            shift_x_px=np.concatenate(output_x),
            score=np.concatenate(output_score),
            objective=np.concatenate(output_objective),
            fit_attempt_count=int(totals[0]),
            fit_accept_count=int(totals[1]),
            boundary_hit_count=int(totals[2]),
            flat_or_convex_count=int(totals[3]),
            out_of_bounds_count=int(totals[4]),
            exact_reject_count=int(totals[5]),
            objective_gain=float(totals[6]),
            ifft_count=len(angle_values),
        )

    return infer_quadratic_candidates(
        particles,
        references,
        config,
        class_priors,
        temperature,
        initial_poses,
        translation_profiler=translation_profiler,
        rescue_mask=rescue_mask,
    )


def _is_oom(error: Exception) -> bool:
    return (
        "out of memory" in str(error).lower()
        or error.__class__.__name__ == "OutOfMemoryError"
    )


def _gpu_candidate_inference(
    particles,
    references,
    config,
    class_priors,
    temperature,
    initial_poses,
    *,
    rescue_mask=None,
    transform_batch=_transform_batch_cupy,
    fourier_transform_batch=_transform_fourier_batch_cupy,
    indexed_fourier_transform_batch=None,
    workspace=None,
    memory_records=None,
):
    cp = _cupy()
    plan = _memory_plan(config, particles.spatial.shape[1], len(references.spatial))
    limit = plan.batch_size
    if memory_records is not None:
        memory_records.append({"stage": "candidate_inference", **plan.asdict()})
    while True:
        attempt = replace(config, batch_size=limit)
        try:
            if attempt.search_strategy == "adaptive_posterior":
                return _gpu_adaptive_candidate_inference_once(
                    particles,
                    references,
                    attempt,
                    class_priors,
                    temperature,
                    initial_poses,
                    rescue_mask=rescue_mask,
                    transform_batch=transform_batch,
                    fourier_transform_batch=fourier_transform_batch,
                    indexed_fourier_transform_batch=indexed_fourier_transform_batch,
                    workspace=workspace,
                    memory_records=memory_records,
                )
            if attempt.search_strategy == "quadratic_refine":
                return _gpu_quadratic_candidate_inference_once(
                    particles,
                    references,
                    attempt,
                    class_priors,
                    temperature,
                    initial_poses,
                    rescue_mask=rescue_mask,
                    fourier_transform_batch=fourier_transform_batch,
                    indexed_fourier_transform_batch=indexed_fourier_transform_batch,
                    workspace=workspace,
                    memory_records=memory_records,
                )
            return _gpu_candidate_inference_once(
                particles,
                references,
                attempt,
                class_priors,
                temperature,
                initial_poses,
                transform_batch=transform_batch,
                fourier_transform_batch=fourier_transform_batch,
                indexed_fourier_transform_batch=indexed_fourier_transform_batch,
                workspace=workspace,
                memory_records=memory_records,
            )
        except Exception as error:
            if not _is_oom(error):
                raise
            evicted = workspace is not None and workspace.disable_cache(
                "scoring", reason="candidate_inference_oom"
            )
            if not evicted and limit == 1:
                raise
            cp.get_default_memory_pool().free_all_blocks()
            if not evicted:
                limit = max(1, limit // 2)
            if memory_records is not None:
                memory_records.append(
                    {
                        "stage": "candidate_inference",
                        "event": "oom_retry",
                        "batch_size": limit,
                    }
                )


def _gpu_reference_updater_once(
    images,
    candidate_values,
    inlier_weights,
    reference_count,
    config,
    *,
    subset=None,
    transform_batch=_transform_batch_cupy,
    fourier_transform_batch=_transform_fourier_batch_cupy,
    indexed_fourier_transform_batch=None,
    fused_fourier_accumulator=None,
    particle_fourier=None,
    workspace=None,
    memory_records=None,
):
    cp = _cupy()
    particle_count, size, _ = images.shape
    if "_mstep_particle_offsets" in candidate_values:
        offsets = candidate_values["_mstep_particle_offsets"]
        particle_ids = np.repeat(np.arange(particle_count), np.diff(offsets))
        field_prefix = "_mstep_"
    else:
        particle_ids = np.repeat(
            np.arange(particle_count), candidate_values["posterior"].shape[1]
        )
        field_prefix = ""
    allowed = np.ones(particle_count, dtype=np.bool_) if subset is None else subset
    keep = allowed[particle_ids]
    weights = (
        candidate_values[f"{field_prefix}posterior"].ravel()
        * inlier_weights[particle_ids]
    )
    keep &= weights > 1e-12
    particle_ids = particle_ids[keep]
    reference_ids = candidate_values[f"{field_prefix}reference_index"].ravel()[keep]
    angles = candidate_values[f"{field_prefix}angle_deg"].ravel()[keep]
    shifts_y = candidate_values[f"{field_prefix}shift_y_px"].ravel()[keep]
    shifts_x = candidate_values[f"{field_prefix}shift_x_px"].ravel()[keep]
    mirrors = candidate_values[f"{field_prefix}mirror"].ravel()[keep]
    weights = weights[keep].astype(np.float32)
    if config.reference_update == "fourier":
        if particle_fourier is None:
            particle_fourier = np.fft.fft2(images, axes=(-2, -1)).astype(np.complex64)
        device_fourier, plan = _acquire_particle_fourier(
            particle_fourier,
            config,
            role="update",
            fixed_bytes=reference_count * size * size * 16,
            bytes_per_item=max(1, size * size * 24),
            workspace=workspace,
            memory_records=memory_records,
        )
        if memory_records is not None:
            memory_records.append({"stage": "fourier_reference_update", **plan.asdict()})
        sums_real = cp.zeros((reference_count, size, size), dtype=cp.float64)
        sums_imag = cp.zeros((reference_count, size, size), dtype=cp.float64)
        sums = None
    else:
        device_fourier = None
        plan = _memory_plan(config, size)
        sums = cp.zeros((reference_count, size, size), dtype=cp.float64)
        sums_real = sums_imag = None
    total_weights = cp.zeros(reference_count, dtype=cp.float64)
    use_fused_fourier = (
        config.reference_update == "fourier"
        and device_fourier is not None
        and fused_fourier_accumulator is not None
    )
    batch_limit = plan.batch_size
    for start in range(0, len(particle_ids), batch_limit):
        stop = min(start + batch_limit, len(particle_ids))
        selected_particles = particle_ids[start:stop]
        if use_fused_fourier:
            fused_fourier_accumulator(
                device_fourier,
                selected_particles,
                angles[start:stop],
                shifts_y[start:stop],
                shifts_x[start:stop],
                mirrors[start:stop],
                reference_ids[start:stop],
                weights[start:stop],
                sums_real,
                sums_imag,
                total_weights,
            )
            continue
        if config.reference_update == "fourier":
            transformed = _transform_cached_fourier(
                device_fourier,
                particle_fourier,
                selected_particles,
                angles[start:stop],
                shifts_y[start:stop],
                shifts_x[start:stop],
                mirrors[start:stop],
                fourier_transform_batch=fourier_transform_batch,
                indexed_fourier_transform_batch=indexed_fourier_transform_batch,
            )
        else:
            transformed = transform_batch(
                images[selected_particles],
                angles[start:stop],
                shifts_y[start:stop],
                shifts_x[start:stop],
                mirrors[start:stop],
            )
        with profile_scope("weighted_accumulation", cuda=True):
            refs = _asdevice(reference_ids[start:stop])
            batch_weights = _asdevice(weights[start:stop], dtype=cp.float64)
            if config.reference_update == "fourier":
                cp.add.at(
                    sums_real,
                    refs,
                    transformed.real.astype(cp.float64) * batch_weights[:, None, None],
                )
                cp.add.at(
                    sums_imag,
                    refs,
                    transformed.imag.astype(cp.float64) * batch_weights[:, None, None],
                )
            else:
                cp.add.at(
                    sums,
                    refs,
                    transformed.astype(cp.float64) * batch_weights[:, None, None],
                )
            cp.add.at(total_weights, refs, batch_weights)
    host_weights = _ashost(total_weights)
    if config.reference_update == "fourier":
        host_sums = np.zeros((reference_count, size, size), dtype=np.float32)
        nonempty = np.flatnonzero(host_weights > 1e-8)
        if len(nonempty):
            selected = _asdevice(nonempty, dtype=cp.int32)
            averaged = (
                sums_real[selected] + 1j * sums_imag[selected]
            ) / total_weights[selected, None, None]
            host_sums[nonempty] = _ashost(
                profile_call("reference_ifft", cp.fft.ifft2, averaged,
                             axes=(-2, -1), cuda=True).real.astype(cp.float32)
            )
    else:
        host_sums = _ashost(sums)
    references = np.zeros((reference_count, size, size), dtype=np.float32)
    mask = soft_circular_mask(size, config.mask_radius, config.mask_soft_edge)
    center_shifts = []
    for reference in range(reference_count):
        if host_weights[reference] > 1e-8:
            value = (
                host_sums[reference]
                if config.reference_update == "fourier"
                else (host_sums[reference] / host_weights[reference]).astype(np.float32)
            )
        else:
            value = np.zeros((size, size), dtype=np.float32)
        if config.lowpass_sigma > 0:
            value = cv2.GaussianBlur(value, (0, 0), config.lowpass_sigma)
        value *= mask
        if config.center_references:
            value, shift_y, shift_x = _center_reference(value)
        else:
            shift_y = shift_x = 0.0
        references[reference] = value
        center_shifts.append((shift_y, shift_x))
    return references, host_weights.astype(np.float32), center_shifts


def _gpu_reference_updater(
    images,
    candidate_values,
    inlier_weights,
    reference_count,
    config,
    *,
    subset=None,
    transform_batch=_transform_batch_cupy,
    fourier_transform_batch=_transform_fourier_batch_cupy,
    indexed_fourier_transform_batch=None,
    fused_fourier_accumulator=None,
    particle_fourier=None,
    workspace=None,
    memory_records=None,
):
    cp = _cupy()
    plan = _memory_plan(config, images.shape[1], reference_count)
    limit = plan.batch_size
    if memory_records is not None:
        memory_records.append({"stage": "reference_update", **plan.asdict()})
    while True:
        attempt = replace(config, batch_size=limit)
        try:
            return _gpu_reference_updater_once(
                images,
                candidate_values,
                inlier_weights,
                reference_count,
                attempt,
                subset=subset,
                transform_batch=transform_batch,
                fourier_transform_batch=fourier_transform_batch,
                indexed_fourier_transform_batch=indexed_fourier_transform_batch,
                fused_fourier_accumulator=fused_fourier_accumulator,
                particle_fourier=particle_fourier,
                workspace=workspace,
                memory_records=memory_records,
            )
        except Exception as error:
            if not _is_oom(error):
                raise
            evicted = False
            if workspace is not None:
                evicted = workspace.disable_cache(
                    "update", reason="reference_update_oom"
                )
                if not evicted:
                    evicted = workspace.disable_cache(
                        "scoring", reason="reference_update_oom"
                    )
            if not evicted and limit == 1:
                raise
            cp.get_default_memory_pool().free_all_blocks()
            if not evicted:
                limit = max(1, limit // 2)
            if memory_records is not None:
                memory_records.append(
                    {
                        "stage": "reference_update",
                        "event": "oom_retry",
                        "batch_size": limit,
                    }
                )


def _gpu_shared_reference_updater_once(
    images,
    candidate_values,
    inlier_weights,
    reference_count,
    config,
    *,
    first_half,
    transform_batch=_transform_batch_cupy,
    fourier_transform_batch=_transform_fourier_batch_cupy,
    indexed_fourier_transform_batch=None,
    fused_fourier_accumulator=None,
    particle_fourier=None,
    workspace=None,
    memory_records=None,
):
    """Accumulate unnormalized half sums after one transform per candidate."""
    cp = _cupy()
    particle_count, size, _ = images.shape
    membership = _validate_halfset_membership(first_half, particle_count)
    if "_mstep_particle_offsets" in candidate_values:
        offsets = candidate_values["_mstep_particle_offsets"]
        particle_ids = np.repeat(np.arange(particle_count), np.diff(offsets))
        field_prefix = "_mstep_"
    else:
        particle_ids = np.repeat(
            np.arange(particle_count), candidate_values["posterior"].shape[1]
        )
        field_prefix = ""
    weights = (
        candidate_values[f"{field_prefix}posterior"].ravel()
        * inlier_weights[particle_ids]
    )
    keep = weights > 1e-12
    particle_ids = particle_ids[keep]
    reference_ids = candidate_values[f"{field_prefix}reference_index"].ravel()[keep]
    angles = candidate_values[f"{field_prefix}angle_deg"].ravel()[keep]
    shifts_y = candidate_values[f"{field_prefix}shift_y_px"].ravel()[keep]
    shifts_x = candidate_values[f"{field_prefix}shift_x_px"].ravel()[keep]
    mirrors = candidate_values[f"{field_prefix}mirror"].ravel()[keep]
    weights = weights[keep].astype(np.float32)
    half_ids = np.where(membership[particle_ids], 0, 1).astype(np.int32)
    accumulator_ids = half_ids * reference_count + reference_ids
    accumulator_count = 2 * reference_count
    # Two retained half sums plus conservative room for the derived full sum and
    # averaged FFT temporaries.
    planned_reference_count = 4 * reference_count
    if config.reference_update == "fourier":
        if particle_fourier is None:
            particle_fourier = np.fft.fft2(images, axes=(-2, -1)).astype(np.complex64)
        device_fourier, plan = _acquire_particle_fourier(
            particle_fourier,
            config,
            role="update",
            fixed_bytes=planned_reference_count * size * size * 16,
            bytes_per_item=max(1, size * size * 24),
            workspace=workspace,
            memory_records=memory_records,
        )
        if memory_records is not None:
            memory_records.append(
                {"stage": "shared_fourier_reference_update", **plan.asdict()}
            )
        sums_real = cp.zeros((accumulator_count, size, size), dtype=cp.float64)
        sums_imag = cp.zeros((accumulator_count, size, size), dtype=cp.float64)
        sums = None
    else:
        device_fourier = None
        plan = _memory_plan(config, size, planned_reference_count)
        sums = cp.zeros((accumulator_count, size, size), dtype=cp.float64)
        sums_real = sums_imag = None
    total_weights = cp.zeros(accumulator_count, dtype=cp.float64)
    use_fused_fourier = (
        config.reference_update == "fourier"
        and device_fourier is not None
        and fused_fourier_accumulator is not None
    )
    batch_limit = plan.batch_size
    for start in range(0, len(particle_ids), batch_limit):
        stop = min(start + batch_limit, len(particle_ids))
        selected_particles = particle_ids[start:stop]
        if use_fused_fourier:
            fused_fourier_accumulator(
                device_fourier,
                selected_particles,
                angles[start:stop],
                shifts_y[start:stop],
                shifts_x[start:stop],
                mirrors[start:stop],
                accumulator_ids[start:stop],
                weights[start:stop],
                sums_real,
                sums_imag,
                total_weights,
            )
            continue
        if config.reference_update == "fourier":
            transformed = _transform_cached_fourier(
                device_fourier,
                particle_fourier,
                selected_particles,
                angles[start:stop],
                shifts_y[start:stop],
                shifts_x[start:stop],
                mirrors[start:stop],
                fourier_transform_batch=fourier_transform_batch,
                indexed_fourier_transform_batch=indexed_fourier_transform_batch,
            )
        else:
            transformed = transform_batch(
                images[selected_particles],
                angles[start:stop],
                shifts_y[start:stop],
                shifts_x[start:stop],
                mirrors[start:stop],
            )
        with profile_scope("shared_weighted_accumulation", cuda=True):
            refs = _asdevice(accumulator_ids[start:stop])
            batch_weights = _asdevice(weights[start:stop], dtype=cp.float64)
            if config.reference_update == "fourier":
                cp.add.at(
                    sums_real,
                    refs,
                    transformed.real.astype(cp.float64) * batch_weights[:, None, None],
                )
                cp.add.at(
                    sums_imag,
                    refs,
                    transformed.imag.astype(cp.float64) * batch_weights[:, None, None],
                )
            else:
                cp.add.at(
                    sums,
                    refs,
                    transformed.astype(cp.float64) * batch_weights[:, None, None],
                )
            cp.add.at(total_weights, refs, batch_weights)
    profile_count("shared_mstep_candidate_transforms", len(particle_ids))

    host_half_weights = _ashost(total_weights).reshape(2, reference_count)
    full_weights = host_half_weights[0] + host_half_weights[1]
    if config.reference_update == "fourier":
        half_real = sums_real.reshape(2, reference_count, size, size)
        half_imag = sums_imag.reshape(2, reference_count, size, size)
        full_spatial = np.zeros((reference_count, size, size), dtype=np.float32)
        nonempty = np.flatnonzero(full_weights > 1e-8)
        if len(nonempty):
            selected = _asdevice(nonempty, dtype=cp.int32)
            averaged = (
                half_real[0, selected]
                + half_real[1, selected]
                + 1j * (half_imag[0, selected] + half_imag[1, selected])
            ) / _asdevice(full_weights[nonempty])[:, None, None]
            full_spatial[nonempty] = _ashost(
                profile_call(
                    "reference_ifft",
                    cp.fft.ifft2,
                    averaged,
                    axes=(-2, -1),
                    cuda=True,
                ).real.astype(cp.float32)
            )
    else:
        half_sums = sums.reshape(2, reference_count, size, size)
        full_spatial = _ashost(half_sums[0] + half_sums[1])
    references, center_shifts = _finalize_spatial_reference_sums(
        full_spatial,
        (
            np.ones(reference_count, dtype=np.float64)
            if config.reference_update == "fourier"
            else full_weights
        ),
        config,
    )

    half_started = time.perf_counter()
    if config.reference_update == "fourier":
        half_spatial = np.zeros(
            (2, reference_count, size, size), dtype=np.float32
        )
        flat_weights = host_half_weights.ravel()
        nonempty = np.flatnonzero(flat_weights > 1e-8)
        if len(nonempty):
            selected = _asdevice(nonempty, dtype=cp.int32)
            averaged = (
                sums_real[selected] + 1j * sums_imag[selected]
            ) / _asdevice(flat_weights[nonempty])[:, None, None]
            half_spatial.reshape(accumulator_count, size, size)[nonempty] = _ashost(
                profile_call(
                    "halfset_reference_ifft",
                    cp.fft.ifft2,
                    averaged,
                    axes=(-2, -1),
                    cuda=True,
                ).real.astype(cp.float32)
            )
        finalize_weights = np.ones(reference_count, dtype=np.float64)
    else:
        half_spatial = _ashost(half_sums)
        finalize_weights = None
    half_references = np.stack(
        [
            _finalize_spatial_reference_sums(
                half_spatial[index],
                (
                    finalize_weights
                    if finalize_weights is not None
                    else host_half_weights[index]
                ),
                config,
                pre_shifts=center_shifts,
            )[0]
            for index in range(2)
        ]
    )
    return _SharedReferenceUpdate(
        references=references,
        effective_weights=full_weights.astype(np.float32),
        center_shifts=center_shifts,
        half_references=half_references,
        half_weights=host_half_weights.astype(np.float32),
        halfset_finalize_seconds=time.perf_counter() - half_started,
    )


def _gpu_shared_reference_updater(
    images,
    candidate_values,
    inlier_weights,
    reference_count,
    config,
    *,
    first_half,
    transform_batch=_transform_batch_cupy,
    fourier_transform_batch=_transform_fourier_batch_cupy,
    indexed_fourier_transform_batch=None,
    fused_fourier_accumulator=None,
    particle_fourier=None,
    workspace=None,
    memory_records=None,
):
    cp = _cupy()
    plan = _memory_plan(config, images.shape[1], 4 * reference_count)
    limit = plan.batch_size
    if memory_records is not None:
        memory_records.append({"stage": "shared_reference_update", **plan.asdict()})
    while True:
        attempt = replace(config, batch_size=limit)
        try:
            return _gpu_shared_reference_updater_once(
                images,
                candidate_values,
                inlier_weights,
                reference_count,
                attempt,
                first_half=first_half,
                transform_batch=transform_batch,
                fourier_transform_batch=fourier_transform_batch,
                indexed_fourier_transform_batch=indexed_fourier_transform_batch,
                fused_fourier_accumulator=fused_fourier_accumulator,
                particle_fourier=particle_fourier,
                workspace=workspace,
                memory_records=memory_records,
            )
        except Exception as error:
            if not _is_oom(error):
                raise
            evicted = False
            if workspace is not None:
                evicted = workspace.disable_cache(
                    "update", reason="shared_reference_update_oom"
                )
                if not evicted:
                    evicted = workspace.disable_cache(
                        "scoring", reason="shared_reference_update_oom"
                    )
            if not evicted and limit == 1:
                raise
            cp.get_default_memory_pool().free_all_blocks()
            if not evicted:
                limit = max(1, limit // 2)
            if memory_records is not None:
                memory_records.append(
                    {
                        "stage": "shared_reference_update",
                        "event": "oom_retry",
                        "batch_size": limit,
                    }
                )


def _run_soft_alignment_gpu_engine(
    images: np.ndarray,
    references: np.ndarray,
    *,
    config: AlignmentConfig,
    class_priors: np.ndarray | None,
    initial_poses: PoseSet | None,
    workflow: str,
    engine: str,
) -> AlignmentResult:
    if engine == "cuda":
        if _native_module() is None:
            raise RuntimeError(
                "backend='cuda' requires a native CUDA build of alignimg-gpu. "
                "Use backend='cupy' for the fallback engine."
            )
        transform_batch = _transform_batch_cuda
        fourier_transform_batch = _transform_fourier_batch_cuda
        indexed_fourier_transform_batch = _transform_fourier_indexed_batch_cuda
        fused_fourier_accumulator = _accumulate_fourier_indexed_batch_cuda
    elif engine == "cupy":
        transform_batch = _transform_batch_cupy
        fourier_transform_batch = _transform_fourier_batch_cupy
        indexed_fourier_transform_batch = None
        fused_fourier_accumulator = None
    else:
        raise ValueError("GPU engine must be 'cuda' or 'cupy'.")

    memory_records: list[dict[str, object]] = []
    cp = _cupy()
    profile = current_profile()
    if profile is not None:
        profile.attach_cuda(cp)
    workspace = WorkflowGpuWorkspace(cp, config, memory_records)

    def candidate_inference(*args, **kwargs):
        return _gpu_candidate_inference(
            *args,
            **kwargs,
            transform_batch=transform_batch,
            fourier_transform_batch=fourier_transform_batch,
            indexed_fourier_transform_batch=indexed_fourier_transform_batch,
            workspace=workspace,
            memory_records=memory_records,
        )

    def reference_updater(*args, **kwargs):
        return _gpu_reference_updater(
            *args,
            **kwargs,
            transform_batch=transform_batch,
            fourier_transform_batch=fourier_transform_batch,
            indexed_fourier_transform_batch=indexed_fourier_transform_batch,
            fused_fourier_accumulator=fused_fourier_accumulator,
            workspace=workspace,
            memory_records=memory_records,
        )

    def shared_reference_updater(*args, **kwargs):
        return _gpu_shared_reference_updater(
            *args,
            **kwargs,
            transform_batch=transform_batch,
            fourier_transform_batch=fourier_transform_batch,
            indexed_fourier_transform_batch=indexed_fourier_transform_batch,
            fused_fourier_accumulator=fused_fourier_accumulator,
            workspace=workspace,
            memory_records=memory_records,
        )

    started = time.perf_counter()
    try:
        result = run_soft_alignment_cpu(
            images,
            references,
            config=config,
            class_priors=class_priors,
            initial_poses=initial_poses,
            workflow=workflow,
            _candidate_inference=candidate_inference,
            _reference_updater=reference_updater,
            _shared_reference_updater=shared_reference_updater,
            _backend_name=engine,
        )
    finally:
        workspace.close()
    workspace_summary = workspace.summary()
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    name = properties.get("name", b"unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    normalized_references = np.asarray(references)
    reference_count = (
        1 if normalized_references.ndim == 2 else len(normalized_references)
    )
    memory_plan = _memory_plan(
        config, int(np.asarray(images).shape[1]), reference_count
    )
    result.metadata.update(
        {
            "engine": f"alignimg-soft-fourier-{engine}",
            "gpu_device": str(name),
            "gpu_total_seconds": time.perf_counter() - started,
            "gpu_memory_free_bytes_at_completion": int(free_bytes),
            "gpu_memory_total_bytes": int(total_bytes),
            "gpu_requested_memory_fraction": config.memory_fraction,
            "gpu_memory_plan_at_completion": memory_plan.asdict(),
            "gpu_memory_plans": memory_records,
            "gpu_workspace": workspace_summary,
            "candidate_scoring": config.candidate_scoring,
            "reference_update": config.reference_update,
            "quadratic_peak_backend": (
                "native_cuda"
                if engine == "cuda" and config.search_strategy == "quadratic_refine"
                else "cupy"
                if config.search_strategy == "quadratic_refine"
                else None
            ),
            "gpu_policy": (
                "CPU angular-mode controller; workflow-cached particle DFTs; native CUDA "
                "indexed Fourier rotation; CuPy batched cuFFT correlation maps, device-side "
                "bounded peak/quadratic translation fitting, and exact Fourier rescoring"
                if engine == "cuda"
                and config.search_strategy == "quadratic_refine"
                else "CPU angular-mode controller; workflow-cached particle DFTs; CuPy "
                "Fourier rotation, batched cuFFT correlation maps, device-side bounded "
                "peak/quadratic translation fitting, and exact Fourier rescoring"
                if config.search_strategy == "quadratic_refine"
                else
                "CPU controller; workflow-cached scoring/raw particle DFTs; native CUDA indexed "
                "Fourier candidate transforms and fused FP64 M-step accumulation; CuPy NCC"
                if engine == "cuda"
                and config.candidate_scoring == "fourier"
                and config.reference_update == "fourier"
                else "CPU controller; workflow-cached scoring/raw particle DFTs; CuPy Fourier-native candidate "
                "and M-step transforms, NCC, and Fourier accumulation"
                if config.candidate_scoring == "fourier"
                and config.reference_update == "fourier"
                else "CPU controller; cached particle DFT; native CUDA Fourier-native "
                "candidate transform; CuPy NCC reduction; native CUDA real-space soft accumulation"
                if engine == "cuda" and config.candidate_scoring == "fourier"
                else "CPU controller; cached particle DFT; CuPy Fourier-native candidate "
                "transform and NCC reduction; CuPy real-space soft accumulation"
                if config.candidate_scoring == "fourier"
                else "CPU adaptive-posterior controller; native CUDA transform; CuPy flat-buffer "
                "Fourier-NCC scoring and soft accumulation"
                if engine == "cuda" and config.search_strategy == "adaptive_posterior"
                else "CPU adaptive-posterior controller; CuPy transform, flat-buffer "
                "Fourier-NCC scoring, and soft accumulation"
                if config.search_strategy == "adaptive_posterior"
                else "CPU polar/controller; native CUDA transform; CuPy Fourier reranking "
                "and soft accumulation"
                if engine == "cuda"
                else "CPU polar/controller; CuPy transform, Fourier reranking, and soft accumulation"
            ),
        }
    )
    return result


def run_soft_alignment_cuda(*args, **kwargs) -> AlignmentResult:
    return _run_soft_alignment_gpu_engine(*args, **kwargs, engine="cuda")


def run_soft_alignment_cupy(*args, **kwargs) -> AlignmentResult:
    return _run_soft_alignment_gpu_engine(*args, **kwargs, engine="cupy")


def run_soft_alignment_gpu(*args, **kwargs) -> AlignmentResult:
    """Compatibility alias preferring native CUDA, then CuPy."""
    engine = "cuda" if _native_module() is not None else "cupy"
    return _run_soft_alignment_gpu_engine(*args, **kwargs, engine=engine)
