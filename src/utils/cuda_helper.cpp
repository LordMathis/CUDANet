#include <cstdio>
#include <cstdlib>
#include "cuda_helper.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                     __FILE__, __LINE__, static_cast<unsigned int>(result), \
                     cudaGetErrorString(result), #call); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// Initialize CUDA and return the device properties
cudaDeviceProp initializeCUDA() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::fprintf(stderr, "No CUDA devices found. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }

    int device = 0; // You can modify this to choose a different GPU
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

    std::printf("Using CUDA device %d: %s\n", device, deviceProp.name);

    return deviceProp;
}
