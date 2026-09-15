#pragma once

#include <cstddef>
#include <cstdint>

namespace alignimg_gpu {

class TransformSession {
public:
    TransformSession(int device_id, int image_size);
    ~TransformSession();
    TransformSession(const TransformSession&) = delete;
    TransformSession& operator=(const TransformSession&) = delete;

    void transform_device(
        std::uintptr_t input_ptr,
        std::uintptr_t output_ptr,
        const float* angles,
        const float* shifts_y,
        const float* shifts_x,
        const std::uint8_t* mirrors,
        std::size_t count,
        std::uintptr_t stream_ptr
    );

    void transform_fourier_device(
        std::uintptr_t input_ptr,
        std::uintptr_t output_ptr,
        const float* angles,
        const float* shifts_y,
        const float* shifts_x,
        const std::uint8_t* mirrors,
        std::size_t count,
        std::uintptr_t stream_ptr
    );

    void transform_fourier_indexed_device(
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
    );

    void accumulate_fourier_indexed_device(
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
    );

    void quadratic_peak_device(
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
    );

private:
    void ensure_capacity(std::size_t count);
    void ensure_accumulation_capacity(std::size_t count);
    int device_id_;
    int image_size_;
    std::size_t capacity_ = 0;
    float* angles_ = nullptr;
    float* shifts_y_ = nullptr;
    float* shifts_x_ = nullptr;
    std::uint8_t* mirrors_ = nullptr;
    std::size_t accumulation_capacity_ = 0;
    std::int32_t* accumulator_ids_ = nullptr;
    double* weights_ = nullptr;
};

}  // namespace alignimg_gpu
