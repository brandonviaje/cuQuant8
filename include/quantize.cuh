#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>

__global__ void findAbsMaxKernel(const float* input, float* d_absMax, int n);
__global__ void quantizeKernel(const float* input, int8_t* output, float scale, int n);
__global__ void dequantizeKernel(const int32_t* input, float* output, float scale, int n);
