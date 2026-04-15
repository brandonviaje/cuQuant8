#include "../include/matmul.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define CUBLAS_CHECK(err) \
    do { \
        cublasStatus_t status = (err); \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error at line " << __LINE__ << " (code: " << status << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void tensorCoreMatMul(const int8_t* d_A, const int8_t* d_B, int32_t* d_C, 
                      int m, int n, int k) {
    
    cublasLtHandle_t ltHandle;
    CUBLAS_CHECK(cublasLtCreate(&ltHandle));

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, m, k, m));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32I, m, n, m));

    cublasLtMatmulDesc_t matmulDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    
    // required for INT8 tensor cores to map memory correctly
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    int32_t alpha = 1; 
    int32_t beta = 0; 

    // Give cuBLAS 4MB of scratchpad memory to rearrange INT8 data if needed
    size_t workspaceSize = 4 * 1024 * 1024; 
    void* d_workspace;
    cudaMalloc(&d_workspace, workspaceSize);

    // ASK HEURISTIC ENGINE FOR AN ALGORITHM
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
        preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        std::cerr << "Heuristic couldn't find a valid INT8 algorithm for this GPU!" << std::endl;
        exit(EXIT_FAILURE);
    }


    CUBLAS_CHECK(cublasLtMatmul(
        ltHandle, matmulDesc, &alpha,
        d_A, layoutA, d_B, layoutB,
        &beta, d_C, layoutC, d_C, layoutC,
        &heuristicResult.algo,
        d_workspace,          
        workspaceSize,         
        0            
    ));

    // Cleanup
    cudaFree(d_workspace);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(ltHandle);
}
