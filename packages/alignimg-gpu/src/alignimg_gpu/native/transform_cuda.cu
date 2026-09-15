#include "transform_cuda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cfloat>
#include <sstream>
#include <stdexcept>

namespace alignimg_gpu {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << operation << ": " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

__global__ void transform_bilinear_wrap(
    const float* images,
    float* output,
    const float* angles,
    const float* shifts_y,
    const float* shifts_x,
    const std::uint8_t* mirrors,
    int count,
    int size
) {
    const std::size_t index = static_cast<std::size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    const std::size_t pixels = static_cast<std::size_t>(size) * size;
    if (index >= static_cast<std::size_t>(count) * pixels) return;
    const int image = static_cast<int>(index / pixels);
    const int offset = static_cast<int>(index % pixels);
    const int y = offset / size;
    const int x = offset % size;
    const float center = size * 0.5f;
    const float radians = angles[image] * 0.017453292519943295f;
    const float cosine = cosf(radians);
    const float sine = sinf(radians);
    const float out_x = x - shifts_x[image] - center;
    const float out_y = y - shifts_y[image] - center;
    float source_x = cosine * out_x - sine * out_y + center;
    float source_y = sine * out_x + cosine * out_y + center;
    if (mirrors[image]) source_x = 2.0f * center - source_x;
    source_x = fmodf(fmodf(source_x, static_cast<float>(size)) + size, static_cast<float>(size));
    source_y = fmodf(fmodf(source_y, static_cast<float>(size)) + size, static_cast<float>(size));
    const int x0 = static_cast<int>(floorf(source_x));
    const int y0 = static_cast<int>(floorf(source_y));
    const int x1 = (x0 + 1) % size;
    const int y1 = (y0 + 1) % size;
    const float wx = source_x - x0;
    const float wy = source_y - y0;
    const float* source = images + static_cast<std::size_t>(image) * pixels;
    const float top = source[y0 * size + x0] * (1.0f - wx) + source[y0 * size + x1] * wx;
    const float bottom = source[y1 * size + x0] * (1.0f - wx) + source[y1 * size + x1] * wx;
    output[index] = top * (1.0f - wy) + bottom * wy;
}

__device__ __forceinline__ int wrap_index(int value, int size) {
    const int result = value % size;
    return result < 0 ? result + size : result;
}

__device__ __forceinline__ float2 transform_fourier_pixel(
    const float2* source,
    int y,
    int x,
    float angle,
    float shift_y,
    float shift_x,
    bool mirror,
    int size
) {
    const int ky = y < size / 2 ? y : y - size;
    const int kx = x < size / 2 ? x : x - size;
    const float radians = angle * 0.017453292519943295f;
    const float cosine = cosf(radians);
    const float sine = sinf(radians);
    const float source_y = cosine * ky + sine * kx;
    float source_x = cosine * kx - sine * ky;
    if (mirror) source_x = -source_x;
    const int floor_y = static_cast<int>(floorf(source_y));
    const int floor_x = static_cast<int>(floorf(source_x));
    const int y0 = wrap_index(floor_y, size);
    const int x0 = wrap_index(floor_x, size);
    const int y1 = (y0 + 1) % size;
    const int x1 = (x0 + 1) % size;
    const float wy = source_y - floor_y;
    const float wx = source_x - floor_x;
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
    const float phase = -6.283185307179586f * (shift_y * ky + shift_x * kx) / size;
    const float phase_cosine = cosf(phase);
    const float phase_sine = sinf(phase);
    return make_float2(
        value.x * phase_cosine - value.y * phase_sine,
        value.x * phase_sine + value.y * phase_cosine
    );
}

__global__ void transform_fourier_bilinear(
    const float2* fourier,
    float2* output,
    const float* angles,
    const float* shifts_y,
    const float* shifts_x,
    const std::uint8_t* mirrors,
    int count,
    int size
) {
    const std::size_t index = static_cast<std::size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    const std::size_t pixels = static_cast<std::size_t>(size) * size;
    if (index >= static_cast<std::size_t>(count) * pixels) return;
    const int image = static_cast<int>(index / pixels);
    const int offset = static_cast<int>(index % pixels);
    const int y = offset / size;
    const int x = offset % size;
    const float2* source = fourier + static_cast<std::size_t>(image) * pixels;
    output[index] = transform_fourier_pixel(
        source, y, x, angles[image], shifts_y[image], shifts_x[image],
        mirrors[image] != 0, size
    );
}

__global__ void transform_fourier_bilinear_indexed(
    const float2* fourier,
    const int* particle_indices,
    float2* output,
    const float* angles,
    const float* shifts_y,
    const float* shifts_x,
    const std::uint8_t* mirrors,
    int source_count,
    int count,
    int size
) {
    const std::size_t index = static_cast<std::size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    const std::size_t pixels = static_cast<std::size_t>(size) * size;
    if (index >= static_cast<std::size_t>(count) * pixels) return;
    const int image = static_cast<int>(index / pixels);
    const int source_image = particle_indices[image];
    if (source_image < 0 || source_image >= source_count) {
        output[index] = make_float2(0.0f, 0.0f);
        return;
    }
    const int offset = static_cast<int>(index % pixels);
    const int y = offset / size;
    const int x = offset % size;
    const float2* source = fourier + static_cast<std::size_t>(source_image) * pixels;
    output[index] = transform_fourier_pixel(
        source, y, x, angles[image], shifts_y[image], shifts_x[image],
        mirrors[image] != 0, size
    );
}

__global__ void accumulate_fourier_bilinear_indexed(
    const float2* fourier,
    const int* particle_indices,
    double* sums_real,
    double* sums_imag,
    double* total_weights,
    const float* angles,
    const float* shifts_y,
    const float* shifts_x,
    const std::uint8_t* mirrors,
    const int* accumulator_ids,
    const double* weights,
    int source_count,
    int accumulator_count,
    int count,
    int size
) {
    const std::size_t index = static_cast<std::size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    const std::size_t pixels = static_cast<std::size_t>(size) * size;
    if (index >= static_cast<std::size_t>(count) * pixels) return;
    const int candidate = static_cast<int>(index / pixels);
    const int source_image = particle_indices[candidate];
    const int accumulator = accumulator_ids[candidate];
    if (
        source_image < 0 || source_image >= source_count ||
        accumulator < 0 || accumulator >= accumulator_count
    ) {
        return;
    }
    const int offset = static_cast<int>(index % pixels);
    const int y = offset / size;
    const int x = offset % size;
    const float2* source = fourier + static_cast<std::size_t>(source_image) * pixels;
    const float2 value = transform_fourier_pixel(
        source, y, x, angles[candidate], shifts_y[candidate], shifts_x[candidate],
        mirrors[candidate] != 0, size
    );
    const double weight = weights[candidate];
    const std::size_t output_index =
        static_cast<std::size_t>(accumulator) * pixels + offset;
    atomicAdd(sums_real + output_index, static_cast<double>(value.x) * weight);
    atomicAdd(sums_imag + output_index, static_cast<double>(value.y) * weight);
    if (offset == 0) {
        atomicAdd(total_weights + accumulator, weight);
    }
}

__device__ __forceinline__ double quadratic_offset(
    double left,
    double center,
    double right,
    std::uint8_t* valid,
    std::uint8_t* reason
) {
    const double denominator = left - 2.0 * center + right;
    const double scale = fmax(fmax(fabs(left), fabs(center)), fmax(fabs(right), 1.0));
    if (!(denominator < -32.0 * DBL_EPSILON * scale)) {
        *valid = 0;
        *reason = 2;
        return 0.0;
    }
    const double offset = 0.5 * (left - right) / denominator;
    if (!isfinite(offset) || fabs(offset) > 0.5 + 1e-12) {
        *valid = 0;
        *reason = 3;
        return 0.0;
    }
    *valid = 1;
    *reason = 0;
    return offset;
}

__global__ void bounded_quadratic_peak(
    const double* objective,
    int* peak_y,
    int* peak_x,
    double* offset_y,
    double* offset_x,
    std::uint8_t* valid_y,
    std::uint8_t* valid_x,
    std::uint8_t* reason_y,
    std::uint8_t* reason_x,
    int count,
    int height,
    int width
) {
    const int item = blockDim.x * blockIdx.x + threadIdx.x;
    if (item >= count) return;
    const int area = height * width;
    const double* surface = objective + static_cast<std::size_t>(item) * area;
    int best = 0;
    double best_value = surface[0];
    for (int index = 1; index < area; ++index) {
        if (surface[index] > best_value) {
            best = index;
            best_value = surface[index];
        }
    }
    const int y = best / width;
    const int x = best % width;
    peak_y[item] = y;
    peak_x[item] = x;
    if (y == 0 || y == height - 1) {
        offset_y[item] = 0.0;
        valid_y[item] = 0;
        reason_y[item] = 1;
    } else {
        offset_y[item] = quadratic_offset(
            surface[(y - 1) * width + x],
            best_value,
            surface[(y + 1) * width + x],
            valid_y + item,
            reason_y + item
        );
    }
    if (x == 0 || x == width - 1) {
        offset_x[item] = 0.0;
        valid_x[item] = 0;
        reason_x[item] = 1;
    } else {
        offset_x[item] = quadratic_offset(
            surface[y * width + x - 1],
            best_value,
            surface[y * width + x + 1],
            valid_x + item,
            reason_x + item
        );
    }
}

}  // namespace

TransformSession::TransformSession(int device_id, int image_size)
    : device_id_(device_id), image_size_(image_size) {
    if (image_size < 2 || image_size % 2 != 0) {
        throw std::invalid_argument("image_size must be positive and even");
    }
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
}

TransformSession::~TransformSession() {
    cudaSetDevice(device_id_);
    cudaFree(angles_);
    cudaFree(shifts_y_);
    cudaFree(shifts_x_);
    cudaFree(mirrors_);
    cudaFree(accumulator_ids_);
    cudaFree(weights_);
}

void TransformSession::ensure_capacity(std::size_t count) {
    if (count <= capacity_) return;
    check_cuda(cudaFree(angles_), "cudaFree angles");
    check_cuda(cudaFree(shifts_y_), "cudaFree shifts_y");
    check_cuda(cudaFree(shifts_x_), "cudaFree shifts_x");
    check_cuda(cudaFree(mirrors_), "cudaFree mirrors");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&angles_), count * sizeof(float)), "cudaMalloc angles");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&shifts_y_), count * sizeof(float)), "cudaMalloc shifts_y");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&shifts_x_), count * sizeof(float)), "cudaMalloc shifts_x");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&mirrors_), count * sizeof(std::uint8_t)), "cudaMalloc mirrors");
    capacity_ = count;
}

void TransformSession::ensure_accumulation_capacity(std::size_t count) {
    ensure_capacity(count);
    if (count <= accumulation_capacity_) return;
    check_cuda(cudaFree(accumulator_ids_), "cudaFree accumulator_ids");
    check_cuda(cudaFree(weights_), "cudaFree weights");
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&accumulator_ids_), count * sizeof(std::int32_t)),
        "cudaMalloc accumulator_ids"
    );
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&weights_), count * sizeof(double)),
        "cudaMalloc weights"
    );
    accumulation_capacity_ = count;
}

void TransformSession::transform_device(
    std::uintptr_t input_ptr,
    std::uintptr_t output_ptr,
    const float* angles,
    const float* shifts_y,
    const float* shifts_x,
    const std::uint8_t* mirrors,
    std::size_t count,
    std::uintptr_t stream_ptr
) {
    if (count == 0) return;
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
    ensure_capacity(count);
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    // Pose arrays are tiny. Synchronous copies guarantee that Python-owned host
    // buffers may be released immediately after this binding returns.
    check_cuda(cudaMemcpy(angles_, angles, count * sizeof(float), cudaMemcpyHostToDevice), "copy angles");
    check_cuda(cudaMemcpy(shifts_y_, shifts_y, count * sizeof(float), cudaMemcpyHostToDevice), "copy shifts_y");
    check_cuda(cudaMemcpy(shifts_x_, shifts_x, count * sizeof(float), cudaMemcpyHostToDevice), "copy shifts_x");
    check_cuda(cudaMemcpy(mirrors_, mirrors, count * sizeof(std::uint8_t), cudaMemcpyHostToDevice), "copy mirrors");
    constexpr int threads = 256;
    const std::size_t total = count * static_cast<std::size_t>(image_size_) * image_size_;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    transform_bilinear_wrap<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input_ptr),
        reinterpret_cast<float*>(output_ptr),
        angles_, shifts_y_, shifts_x_, mirrors_,
        static_cast<int>(count), image_size_
    );
    check_cuda(cudaGetLastError(), "transform_bilinear_wrap");
}

void TransformSession::transform_fourier_device(
    std::uintptr_t input_ptr,
    std::uintptr_t output_ptr,
    const float* angles,
    const float* shifts_y,
    const float* shifts_x,
    const std::uint8_t* mirrors,
    std::size_t count,
    std::uintptr_t stream_ptr
) {
    if (count == 0) return;
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
    ensure_capacity(count);
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    check_cuda(cudaMemcpy(angles_, angles, count * sizeof(float), cudaMemcpyHostToDevice), "copy angles");
    check_cuda(cudaMemcpy(shifts_y_, shifts_y, count * sizeof(float), cudaMemcpyHostToDevice), "copy shifts_y");
    check_cuda(cudaMemcpy(shifts_x_, shifts_x, count * sizeof(float), cudaMemcpyHostToDevice), "copy shifts_x");
    check_cuda(cudaMemcpy(mirrors_, mirrors, count * sizeof(std::uint8_t), cudaMemcpyHostToDevice), "copy mirrors");
    constexpr int threads = 256;
    const std::size_t total = count * static_cast<std::size_t>(image_size_) * image_size_;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    transform_fourier_bilinear<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float2*>(input_ptr),
        reinterpret_cast<float2*>(output_ptr),
        angles_, shifts_y_, shifts_x_, mirrors_,
        static_cast<int>(count), image_size_
    );
    check_cuda(cudaGetLastError(), "transform_fourier_bilinear");
}

void TransformSession::transform_fourier_indexed_device(
    std::uintptr_t input_ptr,
    std::uintptr_t particle_indices_ptr,
    std::uintptr_t output_ptr,
    std::size_t source_count,
    const float* angles,
    const float* shifts_y,
    const float* shifts_x,
    const std::uint8_t* mirrors,
    std::size_t count,
    std::uintptr_t stream_ptr
) {
    if (count == 0) return;
    if (source_count == 0) {
        throw std::invalid_argument("source_count must be positive");
    }
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
    ensure_capacity(count);
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    check_cuda(cudaMemcpy(angles_, angles, count * sizeof(float), cudaMemcpyHostToDevice), "copy angles");
    check_cuda(cudaMemcpy(shifts_y_, shifts_y, count * sizeof(float), cudaMemcpyHostToDevice), "copy shifts_y");
    check_cuda(cudaMemcpy(shifts_x_, shifts_x, count * sizeof(float), cudaMemcpyHostToDevice), "copy shifts_x");
    check_cuda(cudaMemcpy(mirrors_, mirrors, count * sizeof(std::uint8_t), cudaMemcpyHostToDevice), "copy mirrors");
    constexpr int threads = 256;
    const std::size_t total = count * static_cast<std::size_t>(image_size_) * image_size_;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    transform_fourier_bilinear_indexed<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float2*>(input_ptr),
        reinterpret_cast<const int*>(particle_indices_ptr),
        reinterpret_cast<float2*>(output_ptr),
        angles_, shifts_y_, shifts_x_, mirrors_,
        static_cast<int>(source_count), static_cast<int>(count), image_size_
    );
    check_cuda(cudaGetLastError(), "transform_fourier_bilinear_indexed");
}

void TransformSession::accumulate_fourier_indexed_device(
    std::uintptr_t input_ptr,
    std::uintptr_t particle_indices_ptr,
    std::uintptr_t sums_real_ptr,
    std::uintptr_t sums_imag_ptr,
    std::uintptr_t total_weights_ptr,
    std::size_t source_count,
    std::size_t accumulator_count,
    const float* angles,
    const float* shifts_y,
    const float* shifts_x,
    const std::uint8_t* mirrors,
    const std::int32_t* accumulator_ids,
    const double* weights,
    std::size_t count,
    std::uintptr_t stream_ptr
) {
    if (count == 0) return;
    if (source_count == 0 || accumulator_count == 0) {
        throw std::invalid_argument("source_count and accumulator_count must be positive");
    }
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
    ensure_accumulation_capacity(count);
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    check_cuda(cudaMemcpy(angles_, angles, count * sizeof(float), cudaMemcpyHostToDevice), "copy angles");
    check_cuda(cudaMemcpy(shifts_y_, shifts_y, count * sizeof(float), cudaMemcpyHostToDevice), "copy shifts_y");
    check_cuda(cudaMemcpy(shifts_x_, shifts_x, count * sizeof(float), cudaMemcpyHostToDevice), "copy shifts_x");
    check_cuda(cudaMemcpy(mirrors_, mirrors, count * sizeof(std::uint8_t), cudaMemcpyHostToDevice), "copy mirrors");
    check_cuda(
        cudaMemcpy(accumulator_ids_, accumulator_ids, count * sizeof(std::int32_t), cudaMemcpyHostToDevice),
        "copy accumulator_ids"
    );
    check_cuda(cudaMemcpy(weights_, weights, count * sizeof(double), cudaMemcpyHostToDevice), "copy weights");
    constexpr int threads = 256;
    const std::size_t total = count * static_cast<std::size_t>(image_size_) * image_size_;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    accumulate_fourier_bilinear_indexed<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float2*>(input_ptr),
        reinterpret_cast<const int*>(particle_indices_ptr),
        reinterpret_cast<double*>(sums_real_ptr),
        reinterpret_cast<double*>(sums_imag_ptr),
        reinterpret_cast<double*>(total_weights_ptr),
        angles_, shifts_y_, shifts_x_, mirrors_, accumulator_ids_, weights_,
        static_cast<int>(source_count), static_cast<int>(accumulator_count),
        static_cast<int>(count), image_size_
    );
    check_cuda(cudaGetLastError(), "accumulate_fourier_bilinear_indexed");
}

void TransformSession::quadratic_peak_device(
    std::uintptr_t objective_ptr,
    std::uintptr_t peak_y_ptr,
    std::uintptr_t peak_x_ptr,
    std::uintptr_t offset_y_ptr,
    std::uintptr_t offset_x_ptr,
    std::uintptr_t valid_y_ptr,
    std::uintptr_t valid_x_ptr,
    std::uintptr_t reason_y_ptr,
    std::uintptr_t reason_x_ptr,
    std::size_t count,
    int height,
    int width,
    std::uintptr_t stream_ptr
) {
    if (count == 0) return;
    if (height < 1 || width < 1) {
        throw std::invalid_argument("quadratic surface dimensions must be positive");
    }
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    bounded_quadratic_peak<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const double*>(objective_ptr),
        reinterpret_cast<int*>(peak_y_ptr),
        reinterpret_cast<int*>(peak_x_ptr),
        reinterpret_cast<double*>(offset_y_ptr),
        reinterpret_cast<double*>(offset_x_ptr),
        reinterpret_cast<std::uint8_t*>(valid_y_ptr),
        reinterpret_cast<std::uint8_t*>(valid_x_ptr),
        reinterpret_cast<std::uint8_t*>(reason_y_ptr),
        reinterpret_cast<std::uint8_t*>(reason_x_ptr),
        static_cast<int>(count), height, width
    );
    check_cuda(cudaGetLastError(), "bounded_quadratic_peak");
}

}  // namespace alignimg_gpu
