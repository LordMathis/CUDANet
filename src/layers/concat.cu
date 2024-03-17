#include "concat.cuh"
#include "cuda_helper.cuh"

using namespace CUDANet;


Layers::Concat::Concat(const unsigned int inputASize, const unsigned int inputBSize)
    : inputASize(inputASize), inputBSize(inputBSize) {

    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_output, sizeof(float) * (inputASize + inputBSize)
    ));

}

Layers::Concat::~Concat() {
    cudaFree(d_output);
}


float* Layers::Concat::forward(const float* d_input_A, const float* d_input_B) {
    CUDA_CHECK(cudaMemcpy(
        d_output, d_input_A, sizeof(float) * inputASize, cudaMemcpyDeviceToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_output + inputASize, d_input_B,
        sizeof(float) * inputBSize, cudaMemcpyDeviceToDevice
    ));

    return d_output;
}
