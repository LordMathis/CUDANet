#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>
#include <cstdio>

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

#endif // CUDA_HELPER_H
