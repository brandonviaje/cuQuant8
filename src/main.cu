#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/matmul.cuh"
#include "../include/quantize.cuh"
#include "../include/dequantize.cuh"

int main(int argc, char* argv[]) {
    const int SIZE = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int M = SIZE, N = SIZE, K = SIZE;
    const int numElementsA = M * K;
    const int numElementsB = K * N;
    const int totalElementsC = M * N;
    
    std::vector<float> h_A(numElementsA);
    std::vector<float> h_B(numElementsB);

    // init with standard normal distribution vals
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for(auto& val : h_A) val = dist(gen);
    for(auto& val : h_B) val = dist(gen);

    // device ptrs
    float *d_A_fp32, *d_B_fp32, *d_C_int8_pipeline, *d_C_fp32_baseline;
    int8_t *d_A_int8, *d_B_int8;
    int32_t *d_C_int32; 
    float *d_absMaxA, *d_absMaxB;

    // alloc device mem
    cudaMalloc(&d_A_fp32, numElementsA * sizeof(float));
    cudaMalloc(&d_B_fp32, numElementsB * sizeof(float));
    cudaMalloc(&d_C_int8_pipeline, totalElementsC * sizeof(float));
    cudaMalloc(&d_C_fp32_baseline, totalElementsC * sizeof(float));
    
    cudaMalloc(&d_A_int8, numElementsA * sizeof(int8_t));
    cudaMalloc(&d_B_int8, numElementsB * sizeof(int8_t));
    cudaMalloc(&d_C_int32, totalElementsC * sizeof(int32_t)); 
    cudaMalloc(&d_absMaxA, sizeof(float));
    cudaMalloc(&d_absMaxB, sizeof(float));

    cudaMemcpy(d_A_fp32, h_A.data(), numElementsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp32, h_B.data(), numElementsB * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop, matmul_start, matmul_stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventCreate(&matmul_start); cudaEventCreate(&matmul_stop);

    int threadsPerBlock = 256;
    int blocksA = (numElementsA + threadsPerBlock - 1) / threadsPerBlock;
    int blocksB = (numElementsB + threadsPerBlock - 1) / threadsPerBlock;
    int blocksC = (totalElementsC + threadsPerBlock - 1) / threadsPerBlock;
    
    // run int8 pipeline
    cudaEventRecord(start);

    cudaMemset(d_absMaxA, 0, sizeof(float));
    findAbsMaxKernel<<<blocksA, threadsPerBlock>>>(d_A_fp32, d_absMaxA, numElementsA);
    float h_absMaxA = 0.0f;
    cudaMemcpy(&h_absMaxA, d_absMaxA, sizeof(float), cudaMemcpyDeviceToHost);
    float scaleA = 127.0f / h_absMaxA;
    quantizeKernel<<<blocksA, threadsPerBlock>>>(d_A_fp32, d_A_int8, scaleA, numElementsA);

    cudaMemset(d_absMaxB, 0, sizeof(float));
    findAbsMaxKernel<<<blocksB, threadsPerBlock>>>(d_B_fp32, d_absMaxB, numElementsB);
    float h_absMaxB = 0.0f;
    cudaMemcpy(&h_absMaxB, d_absMaxB, sizeof(float), cudaMemcpyDeviceToHost);
    float scaleB = 127.0f / h_absMaxB;
    quantizeKernel<<<blocksB, threadsPerBlock>>>(d_B_fp32, d_B_int8, scaleB, numElementsB);

    // record timer now
    cudaEventRecord(matmul_start);
    
    tensorCoreMatMul(d_A_int8, d_B_int8, d_C_int32, M, N, K);
    
    cudaEventRecord(matmul_stop);
    cudaEventSynchronize(matmul_stop);
    float matmul_time = 0;
    cudaEventElapsedTime(&matmul_time, matmul_start, matmul_stop);

    float finalScale = 1.0f / (scaleA * scaleB);
    dequantizeKernel<<<blocksC, threadsPerBlock>>>(d_C_int32, d_C_int8_pipeline, finalScale, totalElementsC);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float int8_time = 0;
    cudaEventElapsedTime(&int8_time, start, stop);

    // run cuBLAS FP32 baseline to compare 
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    cudaEventRecord(start);

    // cuBLAS is column-major, swap A and B to match row-major expectations
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_fp32, N, d_A_fp32, K, &beta, d_C_fp32_baseline, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fp32_time = 0;
    cudaEventElapsedTime(&fp32_time, start, stop);

    // compute MSE
    std::vector<float> h_C_int8(totalElementsC);
    std::vector<float> h_C_fp32(totalElementsC);
    cudaMemcpy(h_C_int8.data(), d_C_int8_pipeline, totalElementsC * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_fp32.data(), d_C_fp32_baseline, totalElementsC * sizeof(float), cudaMemcpyDeviceToHost);

    double mse = 0.0;
    for (int i = 0; i < totalElementsC; i++) {
        double diff = static_cast<double>(h_C_int8[i] - h_C_fp32[i]);
        mse += (diff * diff);
    }
    mse /= totalElementsC;

    std::cout << "INT8 Time: " << int8_time << " ms" << std::endl;
    std::cout << "MatMul Time: " << matmul_time << " ms" << std::endl;
    std::cout << "FP32 Time: " << fp32_time << " ms" << std::endl;
    std::cout << "MSE: " << mse << std::endl;

    cublasDestroy(handle);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaEventDestroy(matmul_start); cudaEventDestroy(matmul_stop);
    cudaFree(d_A_fp32); cudaFree(d_B_fp32); cudaFree(d_C_int8_pipeline); cudaFree(d_C_fp32_baseline);
    cudaFree(d_A_int8); cudaFree(d_B_int8); cudaFree(d_C_int32); 
    cudaFree(d_absMaxA); cudaFree(d_absMaxB);

    return 0;
}
