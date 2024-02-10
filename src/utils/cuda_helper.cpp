#include <cstdio>
#include <cstdlib>
#include "cuda_helper.h"
#include <cublas_v2.h>

cudaDeviceProp initializeCUDA(cublasHandle_t& cublasHandle) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::fprintf(stderr, "No CUDA devices found. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

    std::printf("Using CUDA device %d: %s\n", device, deviceProp.name);

    // Initialize cuBLAS
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    return deviceProp;
}