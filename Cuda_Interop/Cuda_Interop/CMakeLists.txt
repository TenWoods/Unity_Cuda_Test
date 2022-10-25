cmake_minimum_required(VERSION 3.23)
project(Cuda_Interop LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

include_directories(.)
include_directories(nvcomp)
include_directories(nvcomp/nvcomp)
include_directories(PluginAPI)

link_directories(/home/michael/yang/lib/nvcomp-2.0.0/build/lib)

add_library(Cuda_Interop SHARED cuda_interop.cu)
target_link_libraries(Cuda_Interop -lnvcomp)

set_target_properties(Cuda_Interop PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES 52)