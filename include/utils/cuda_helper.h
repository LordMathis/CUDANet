#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

// CUDA error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                __FILE__, __LINE__, static_cast<unsigned int>(result), \
                cudaGetErrorString(result), #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t result = call; \
    if (result != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d code=%d\n", \
                __FILE__, __LINE__, static_cast<int>(result)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#endif // CUDA_HELPER_H
