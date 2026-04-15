#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cstdlib>
#include "../include/matmul.cuh"
#include "../include/quantize.cuh"

int main(int argc, char* argv[]) {
    // get matrix dims
    const int SIZE = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int M = SIZE;
    const int N = SIZE;
    const int K = SIZE;
    const int numElementsA = M * K;
    const int numElementsB = K * N;
    
    std::cout << "Initializing matrices (" << M << "x" << K << ")..." << std::endl;

    // alloc host mem
    std::vector<float> h_A(numElementsA);
    std::vector<float> h_B(numElementsB);
    std::vector<float> h_C(M * N, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for(auto& val : h_A) val = dist(gen);
    for(auto& val : h_B) val = dist(gen);

    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    int8_t *d_A_int8, *d_B_int8;
    int32_t *d_C_int32; 
    float *d_absMaxA, *d_absMaxB;

    cudaMalloc(&d_A_fp32, numElementsA * sizeof(float));
    cudaMalloc(&d_B_fp32, numElementsB * sizeof(float));
    cudaMalloc(&d_C_fp32, M * N * sizeof(float));
    
    cudaMalloc(&d_A_int8, numElementsA * sizeof(int8_t));
    cudaMalloc(&d_B_int8, numElementsB * sizeof(int8_t));
    cudaMalloc(&d_C_int32, M * N * sizeof(int32_t)); 
    cudaMalloc(&d_absMaxA, sizeof(float));
    cudaMalloc(&d_absMaxB, sizeof(float));

    cudaMemcpy(d_A_fp32, h_A.data(), numElementsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp32, h_B.data(), numElementsB * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Running INT8 Quantization + Tensor Core MatMul Pipeline..." << std::endl;
    cudaEventRecord(start);

    int threadsPerBlock = 256;
    
    // quantize Matrix A
    cudaMemset(d_absMaxA, 0, sizeof(float));
    findAbsMaxKernel<<<(numElementsA + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_A_fp32, d_absMaxA, numElementsA);
    float h_absMaxA = 0.0f;
    cudaMemcpy(&h_absMaxA, d_absMaxA, sizeof(float), cudaMemcpyDeviceToHost);
    float scaleA = 127.0f / h_absMaxA;
    quantizeKernel<<<(numElementsA + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_A_fp32, d_A_int8, scaleA, numElementsA);

    // quantize Matrix B
    cudaMemset(d_absMaxB, 0, sizeof(float));
    findAbsMaxKernel<<<(numElementsB + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_B_fp32, d_absMaxB, numElementsB);
    float h_absMaxB = 0.0f;
    cudaMemcpy(&h_absMaxB, d_absMaxB, sizeof(float), cudaMemcpyDeviceToHost);
    float scaleB = 127.0f / h_absMaxB;
    quantizeKernel<<<(numElementsB + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_B_fp32, d_B_int8, scaleB, numElementsB);

    tensorCoreMatMul(d_A_int8, d_B_int8, d_C_int32, M, N, K);

    // dequantize (INT32 -> FP32)
    float finalScale = scaleA * scaleB;
    int totalElementsC = M * N;
    int blocksC = (totalElementsC + threadsPerBlock - 1) / threadsPerBlock;
    dequantizeKernel<<<blocksC, threadsPerBlock>>>(d_C_int32, d_C_fp32, finalScale, totalElementsC);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Pipeline completed in: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A_fp32); cudaFree(d_B_fp32); cudaFree(d_C_fp32);
    cudaFree(d_A_int8); cudaFree(d_B_int8); cudaFree(d_C_int32); 
    cudaFree(d_absMaxA); cudaFree(d_absMaxB);

    return 0;
}
