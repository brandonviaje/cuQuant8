#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <vector>

__device__ void atomicMaxFloat(float* address, float val);
__device__ float warpReduceMax(float val);
__global__ void findAbsMaxKernel(const float* input, float* output, int n);
__global__ void quantizeKernel(const float* input, int8_t* output, float scale, int n);
