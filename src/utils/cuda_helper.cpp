#include <cstdio>
#include <cstdlib>
#include "cuda_helper.h"

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
