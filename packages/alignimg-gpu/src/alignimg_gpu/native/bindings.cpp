#include "transform_cuda.hpp"

#include <cstdint>
#include <memory>

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_native, module) {
    module.doc() = "AlignImg native CUDA primitives";
    module.attr("__version__") = "2.2.0";
    py::class_<alignimg_gpu::TransformSession>(module, "TransformSession")
        .def(py::init<int, int>(), py::arg("device_id"), py::arg("image_size"))
        .def(
            "transform_device",
            [](alignimg_gpu::TransformSession& self,
               std::uintptr_t input_ptr,
               std::uintptr_t output_ptr,
               py::array_t<float, py::array::c_style | py::array::forcecast> angles,
               py::array_t<float, py::array::c_style | py::array::forcecast> shifts_y,
               py::array_t<float, py::array::c_style | py::array::forcecast> shifts_x,
               py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> mirrors,
               std::uintptr_t stream_ptr) {
                const auto count = static_cast<std::size_t>(angles.size());
                if (shifts_y.size() != angles.size() || shifts_x.size() != angles.size() || mirrors.size() != angles.size()) {
                    throw py::value_error("pose arrays must have equal length");
                }
                self.transform_device(
                    input_ptr, output_ptr, angles.data(), shifts_y.data(), shifts_x.data(),
                    mirrors.data(), count, stream_ptr
                );
            },
            py::arg("input_ptr"), py::arg("output_ptr"), py::arg("angles"),
            py::arg("shifts_y"), py::arg("shifts_x"), py::arg("mirrors"),
            py::arg("stream_ptr")
        )
        .def(
            "transform_fourier_device",
            [](alignimg_gpu::TransformSession& self,
               std::uintptr_t input_ptr,
               std::uintptr_t output_ptr,
               py::array_t<float, py::array::c_style | py::array::forcecast> angles,
               py::array_t<float, py::array::c_style | py::array::forcecast> shifts_y,
               py::array_t<float, py::array::c_style | py::array::forcecast> shifts_x,
               py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> mirrors,
               std::uintptr_t stream_ptr) {
                const auto count = static_cast<std::size_t>(angles.size());
                if (shifts_y.size() != angles.size() || shifts_x.size() != angles.size() || mirrors.size() != angles.size()) {
                    throw py::value_error("pose arrays must have equal lengths");
                }
                self.transform_fourier_device(
                    input_ptr, output_ptr, angles.data(), shifts_y.data(), shifts_x.data(),
                    mirrors.data(), count, stream_ptr
                );
            },
            py::arg("input_ptr"), py::arg("output_ptr"), py::arg("angles"),
            py::arg("shifts_y"), py::arg("shifts_x"), py::arg("mirrors"),
            py::arg("stream_ptr")
        )
        .def(
            "transform_fourier_indexed_device",
            [](alignimg_gpu::TransformSession& self,
               std::uintptr_t input_ptr,
               std::uintptr_t particle_indices_ptr,
               std::uintptr_t output_ptr,
               std::size_t source_count,
               py::array_t<float, py::array::c_style | py::array::forcecast> angles,
               py::array_t<float, py::array::c_style | py::array::forcecast> shifts_y,
               py::array_t<float, py::array::c_style | py::array::forcecast> shifts_x,
               py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> mirrors,
               std::uintptr_t stream_ptr) {
                const auto count = static_cast<std::size_t>(angles.size());
                if (shifts_y.size() != angles.size() || shifts_x.size() != angles.size() || mirrors.size() != angles.size()) {
                    throw py::value_error("pose arrays must have equal lengths");
                }
                self.transform_fourier_indexed_device(
                    input_ptr, particle_indices_ptr, output_ptr, source_count,
                    angles.data(), shifts_y.data(), shifts_x.data(), mirrors.data(),
                    count, stream_ptr
                );
            },
            py::arg("input_ptr"), py::arg("particle_indices_ptr"),
            py::arg("output_ptr"), py::arg("source_count"), py::arg("angles"),
            py::arg("shifts_y"), py::arg("shifts_x"), py::arg("mirrors"),
            py::arg("stream_ptr")
        )
        .def(
            "accumulate_fourier_indexed_device",
            [](alignimg_gpu::TransformSession& self,
               std::uintptr_t input_ptr,
               std::uintptr_t particle_indices_ptr,
               std::uintptr_t sums_real_ptr,
               std::uintptr_t sums_imag_ptr,
               std::uintptr_t total_weights_ptr,
               std::size_t source_count,
               std::size_t accumulator_count,
               py::array_t<float, py::array::c_style | py::array::forcecast> angles,
               py::array_t<float, py::array::c_style | py::array::forcecast> shifts_y,
               py::array_t<float, py::array::c_style | py::array::forcecast> shifts_x,
               py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> mirrors,
               py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> accumulator_ids,
               py::array_t<double, py::array::c_style | py::array::forcecast> weights,
               std::uintptr_t stream_ptr) {
                const auto count = static_cast<std::size_t>(angles.size());
                if (
                    shifts_y.size() != angles.size() ||
                    shifts_x.size() != angles.size() ||
                    mirrors.size() != angles.size() ||
                    accumulator_ids.size() != angles.size() ||
                    weights.size() != angles.size()
                ) {
                    throw py::value_error("candidate arrays must have equal lengths");
                }
                self.accumulate_fourier_indexed_device(
                    input_ptr, particle_indices_ptr, sums_real_ptr, sums_imag_ptr,
                    total_weights_ptr, source_count, accumulator_count,
                    angles.data(), shifts_y.data(), shifts_x.data(), mirrors.data(),
                    accumulator_ids.data(), weights.data(), count, stream_ptr
                );
            },
            py::arg("input_ptr"), py::arg("particle_indices_ptr"),
            py::arg("sums_real_ptr"), py::arg("sums_imag_ptr"),
            py::arg("total_weights_ptr"), py::arg("source_count"),
            py::arg("accumulator_count"), py::arg("angles"),
            py::arg("shifts_y"), py::arg("shifts_x"), py::arg("mirrors"),
            py::arg("accumulator_ids"), py::arg("weights"), py::arg("stream_ptr")
        )
        .def(
            "quadratic_peak_device",
            &alignimg_gpu::TransformSession::quadratic_peak_device,
            py::arg("objective_ptr"), py::arg("peak_y_ptr"),
            py::arg("peak_x_ptr"), py::arg("offset_y_ptr"),
            py::arg("offset_x_ptr"), py::arg("valid_y_ptr"),
            py::arg("valid_x_ptr"), py::arg("reason_y_ptr"),
            py::arg("reason_x_ptr"), py::arg("count"),
            py::arg("height"), py::arg("width"), py::arg("stream_ptr")
        );
    module.def("runtime_info", []() {
        int runtime_version = 0;
        int driver_version = 0;
        cudaRuntimeGetVersion(&runtime_version);
        cudaDriverGetVersion(&driver_version);
        py::dict result;
        result["cuda_runtime_version"] = runtime_version;
        result["cuda_driver_version"] = driver_version;
        return result;
    });
}
