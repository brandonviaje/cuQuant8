#include "../include/quantize.cuh"

__global__ void findAbsMaxKernel(const float* input, float* d_absMax, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = fabsf(input[idx]);
        int* address_as_int = (int*)d_absMax;
        int old = *address_as_int, assumed;
        
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed,
                __float_as_int(fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
    }
}

__global__ void quantizeKernel(const float* input, int8_t* output, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = static_cast<int8_t>(roundf(input[idx] * scale));
    }
}

__global__ void dequantizeKernel(const int32_t* input, float* output, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = static_cast<float>(input[idx]) * scale;
    }
}
