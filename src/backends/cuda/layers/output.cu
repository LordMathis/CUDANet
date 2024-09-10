#include "output.hpp"

#include "cuda_helper.cuh"

using namespace CUDANet::Layers;

float* Output::forwardCUDA(const float* input) {
    CUDA_CHECK(cudaMemcpy(
        h_output, input, sizeof(float) * inputSize, cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    return h_output;
}