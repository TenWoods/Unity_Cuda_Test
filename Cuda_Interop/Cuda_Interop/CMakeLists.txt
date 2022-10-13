cmake_minimum_required(VERSION 3.23)
project(Cuda_Interop LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(.)
include_directories(nvcomp)
include_directories(nvcomp/nvcomp)
include_directories(PluginAPI)

add_library(Cuda_Interop SHARED cuda_interop.cu)

set_target_properties(Cuda_Interop PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES 52)