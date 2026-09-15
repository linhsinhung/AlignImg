cmake_minimum_required(VERSION 3.24)
project(alignimg_gpu_native LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA compiler not found; building the CuPy-only fallback package")
  return()
endif()

if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Native CUDA backend supports Linux only; building CuPy-only fallback")
  return()
endif()

enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit 12 REQUIRED)

pybind11_add_module(
  _native
  MODULE
  src/alignimg_gpu/native/bindings.cpp
  src/alignimg_gpu/native/transform_cuda.cu
)
target_include_directories(_native PRIVATE src/alignimg_gpu/native)
target_link_libraries(_native PRIVATE CUDA::cudart)
target_compile_features(_native PRIVATE cxx_std_17)
set_target_properties(
  _native
  PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
install(TARGETS _native LIBRARY DESTINATION alignimg_gpu)
