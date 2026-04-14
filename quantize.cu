#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>

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

int main() {
    const int N = 7;
    float h_input[N] = {5.47f, 3.08f, -7.59f, 0.0f, -1.95f, -4.57f, 10.8f};
    int8_t h_output[N] = {0}; 
    
    float *d_input, *d_absMax;
    int8_t *d_output;
    float h_absMax = 0.0f;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_absMax, sizeof(float));
    cudaMalloc(&d_output, N * sizeof(int8_t)); 
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_absMax, &h_absMax, sizeof(float), cudaMemcpyHostToDevice);

    // find AbsMax
    findAbsMaxKernel<<<1, N>>>(d_input, d_absMax, N);
    cudaDeviceSynchronize(); 
    cudaMemcpy(&h_absMax, d_absMax, sizeof(float), cudaMemcpyDeviceToHost);
    float scale = 127.0f / h_absMax;
    
    std::cout << "GPU found AbsMax: " << h_absMax << "\n";
    std::cout << "CPU calculated Scale: " << scale << "\n\n";


    // quantize
    quantizeKernel<<<1, N>>>(d_input, d_output, scale, N);
    cudaDeviceSynchronize(); 
    cudaMemcpy(h_output, d_output, N * sizeof(int8_t), cudaMemcpyDeviceToHost);

    std::cout << "Original floats: \n";
    for(int i=0; i<N; i++) std::cout << h_input[i] << " ";
    std::cout << "\n\n";

    std::cout << "Quantized INT8: \n";

    for(int i=0; i<N; i++) {
        std::cout << static_cast<int>(h_output[i]) << " "; 
    }

    std::cout << "\n";

    cudaFree(d_input);
    cudaFree(d_absMax);
    cudaFree(d_output);

    return 0;
}
