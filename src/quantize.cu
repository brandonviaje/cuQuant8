#include <cuda_runtime.h>
#include <math.h>

// atomic max for floats using bitwise compare/swap
__device__ void atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// warp-level max reduction usin shuffle instructions
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// finds global max absolute value in input
__global__ void findAbsMaxKernel(const float* input, float* output, int n) {
    float max_val = 0.0f;
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // grid-stride loop: each thread processes multiple elements
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        max_val = fmaxf(max_val, fabsf(input[i]));
    }

    // reduce values within warp
    max_val = warpReduceMax(max_val);

    // shared memory for per-warp results
    __shared__ float shared_max[32];
    int lane = tid % 32;
    int warp_id = tid / 32;

    // store warp result
    if (lane == 0) {
        shared_max[warp_id] = max_val;
    }

    __syncthreads();

    // first warp reduces all warp results
    if (warp_id == 0) {
        max_val = (tid < (blockDim.x / 32)) ? shared_max[lane] : 0.0f;
        max_val = warpReduceMax(max_val);
    }

    // single atomic write per block
    if (tid == 0) {
        atomicMaxFloat(output, max_val);
    }
}

__global__ void quantizeKernel(const float* input, int8_t* output, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // multiply by scale factor and round
        float val = roundf(input[idx] * scale);
        
        // clamp values to valid INT8 range [-127, 127] 
        val = fmaxf(-127.0f, fminf(val, 127.0f));
        
        output[idx] = static_cast<int8_t>(val);
    }
}
