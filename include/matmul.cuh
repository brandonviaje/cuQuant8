#include <iostream>
#include <cuda_runtime.h>
#include <cublasLt.h>

void tensorCoreMatMul(const int8_t* d_A, const int8_t* d_B, int32_t* d_C, 
                      int m, int n, int k);
