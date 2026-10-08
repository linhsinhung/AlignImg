#include "polar_hard_cuda.hpp"

#include <cuda_runtime.h>

#include <cmath>
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

__global__ void sample_polar_quantized(
    const float* images,
    const int* image_indices,
    const double* centers_y,
    const double* centers_x,
    const std::uint8_t* mirrors,
    const double* offsets_y,
    const double* offsets_x,
    float* output,
    int source_count,
    int count,
    int size,
    int angle_samples,
    int radial_bins
) {
    const std::size_t index =
        static_cast<std::size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    const std::size_t ring_pixels =
        static_cast<std::size_t>(angle_samples) * radial_bins;
    if (index >= static_cast<std::size_t>(count) * ring_pixels) return;
    const int item = static_cast<int>(index / ring_pixels);
    const int polar_index = static_cast<int>(index % ring_pixels);
    const int source_index = image_indices[item];
    if (source_index < 0 || source_index >= source_count) {
        output[index] = 0.0;
        return;
    }
    const double origin = static_cast<double>(size / 2);
    const double source_y =
        origin + centers_y[item] + offsets_y[polar_index];
    const double source_x =
        origin + centers_x[item] + offsets_x[polar_index];
    const int quantized_y = static_cast<int>(floor(source_y * 32.0 + 0.5));
    const int quantized_x = static_cast<int>(floor(source_x * 32.0 + 0.5));
    const int y0 = quantized_y >> 5;
    int x0 = quantized_x >> 5;
    if (y0 < 0 || x0 < 0 || y0 >= size || x0 >= size) {
        output[index] = 0.0;
        return;
    }
    const int y1 = min(y0 + 1, size - 1);
    int x1 = min(x0 + 1, size - 1);
    // Sample the CPU authority's periodic mirror after quantizing the original
    // coordinates; retain the original interpolation weights and rounding.
    if (mirrors[item] != 0) {
        x0 = (size - x0) % size;
        x1 = (size - x1) % size;
    }
    const float wy = static_cast<float>(quantized_y & 31) * (1.0f / 32.0f);
    const float wx = static_cast<float>(quantized_x & 31) * (1.0f / 32.0f);
    const float* source = images +
        static_cast<std::size_t>(source_index) * size * size;
    // Match NumPy's separately rounded float32 multiply/add authority instead
    // of allowing the compiler to contract the interpolation into FMAs.
    const float one_minus_wx = __fsub_rn(1.0f, wx);
    const float one_minus_wy = __fsub_rn(1.0f, wy);
    const float top = __fadd_rn(
        __fmul_rn(source[y0 * size + x0], one_minus_wx),
        __fmul_rn(source[y0 * size + x1], wx)
    );
    const float bottom = __fadd_rn(
        __fmul_rn(source[y1 * size + x0], one_minus_wx),
        __fmul_rn(source[y1 * size + x1], wx)
    );
    output[index] = __fadd_rn(
        __fmul_rn(top, one_minus_wy),
        __fmul_rn(bottom, wy)
    );
}

__global__ void select_periodic_angular_peak(
    const double* curves,
    int* peak,
    double* offset,
    double* score,
    std::uint8_t* accepted,
    int count,
    int angle_samples
) {
    const int curve_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (curve_index >= count) return;
    const double* curve = curves +
        static_cast<std::size_t>(curve_index) * angle_samples;
    int best_index = 0;
    double best_value = curve[0];
    for (int angle = 1; angle < angle_samples; ++angle) {
        if (curve[angle] > best_value) {
            best_index = angle;
            best_value = curve[angle];
        }
    }
    const double previous =
        curve[(best_index + angle_samples - 1) % angle_samples];
    const double following = curve[(best_index + 1) % angle_samples];
    const double denominator = previous - 2.0 * best_value + following;
    double fitted_offset = 0.0;
    bool fitted = false;
    if (isfinite(previous) && isfinite(best_value) &&
        isfinite(following) && denominator < 0.0) {
        fitted_offset = 0.5 * (previous - following) / denominator;
        fitted = isfinite(fitted_offset) && fabs(fitted_offset) <= 0.5;
    }
    if (!fitted) fitted_offset = 0.0;
    peak[curve_index] = best_index;
    offset[curve_index] = fitted_offset;
    score[curve_index] =
        best_value - 0.25 * (previous - following) * fitted_offset;
    accepted[curve_index] = fitted ? 1 : 0;
}

}  // namespace

PolarHardSession::PolarHardSession(
    int device_id,
    int image_size,
    int angle_samples,
    int radial_bins
) :
    device_id_(device_id),
    image_size_(image_size),
    angle_samples_(angle_samples),
    radial_bins_(radial_bins) {
    if (image_size < 2 || image_size % 2 != 0) {
        throw std::invalid_argument("image_size must be positive and even");
    }
    if (angle_samples < 3 || radial_bins < 1) {
        throw std::invalid_argument("invalid polar dimensions");
    }
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
}

void PolarHardSession::sample_device(
    std::uintptr_t input_ptr,
    std::uintptr_t image_indices_ptr,
    std::uintptr_t centers_y_ptr,
    std::uintptr_t centers_x_ptr,
    std::uintptr_t mirrors_ptr,
    std::uintptr_t offsets_y_ptr,
    std::uintptr_t offsets_x_ptr,
    std::uintptr_t output_ptr,
    std::size_t source_count,
    std::size_t count,
    std::uintptr_t stream_ptr
) {
    if (count == 0) return;
    if (source_count == 0) {
        throw std::invalid_argument("source_count must be positive");
    }
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 256;
    const std::size_t total = count *
        static_cast<std::size_t>(angle_samples_) * radial_bins_;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    sample_polar_quantized<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(input_ptr),
        reinterpret_cast<const int*>(image_indices_ptr),
        reinterpret_cast<const double*>(centers_y_ptr),
        reinterpret_cast<const double*>(centers_x_ptr),
        reinterpret_cast<const std::uint8_t*>(mirrors_ptr),
        reinterpret_cast<const double*>(offsets_y_ptr),
        reinterpret_cast<const double*>(offsets_x_ptr),
        reinterpret_cast<float*>(output_ptr),
        static_cast<int>(source_count),
        static_cast<int>(count),
        image_size_,
        angle_samples_,
        radial_bins_
    );
    check_cuda(cudaGetLastError(), "sample_polar_quantized");
}

void PolarHardSession::angular_peak_device(
    std::uintptr_t curves_ptr,
    std::uintptr_t peak_ptr,
    std::uintptr_t offset_ptr,
    std::uintptr_t score_ptr,
    std::uintptr_t accepted_ptr,
    std::size_t count,
    std::uintptr_t stream_ptr
) {
    if (count == 0) return;
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    select_periodic_angular_peak<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const double*>(curves_ptr),
        reinterpret_cast<int*>(peak_ptr),
        reinterpret_cast<double*>(offset_ptr),
        reinterpret_cast<double*>(score_ptr),
        reinterpret_cast<std::uint8_t*>(accepted_ptr),
        static_cast<int>(count),
        angle_samples_
    );
    check_cuda(cudaGetLastError(), "select_periodic_angular_peak");
}

}  // namespace alignimg_gpu
