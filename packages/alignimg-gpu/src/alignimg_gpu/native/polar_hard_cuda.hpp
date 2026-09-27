#pragma once

#include <cstddef>
#include <cstdint>

namespace alignimg_gpu {

class PolarHardSession {
public:
    PolarHardSession(
        int device_id,
        int image_size,
        int angle_samples,
        int radial_bins
    );
    PolarHardSession(const PolarHardSession&) = delete;
    PolarHardSession& operator=(const PolarHardSession&) = delete;

    void sample_device(
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
    );

    void angular_peak_device(
        std::uintptr_t curves_ptr,
        std::uintptr_t peak_ptr,
        std::uintptr_t offset_ptr,
        std::uintptr_t score_ptr,
        std::uintptr_t accepted_ptr,
        std::size_t count,
        std::uintptr_t stream_ptr
    );

private:
    int device_id_;
    int image_size_;
    int angle_samples_;
    int radial_bins_;
};

}  // namespace alignimg_gpu
