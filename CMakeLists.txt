cmake_minimum_required(VERSION 3.18)
project(cuQuant8 LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(cuQuant8
    src/dequantize.cu
    src/main.cu
    src/matmul.cu
    src/quantize.cu
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(cuQuant8 PRIVATE CUDA::cublas CUDA::cublasLt)
