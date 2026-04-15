#include "../include/dequantize.cuh"

__global__ void dequantizeKernel(const int32_t* input, float* output, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = static_cast<float>(input[idx]) * scale;
    }
}
