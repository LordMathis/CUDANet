#include "add.hpp"
#include "matmul.cuh"
#include "cuda_helper.cuh"

using namespace CUDANet::Layers;

void Add::initCUDA() {
    d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(float) * inputSize));

    gridSize = (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

void Add::delCUDA() {
    cudaFree(d_output);
}

float* Add::forwardCUDA(const float* d_inputA, const float* d_inputB) {

    Kernels::vec_vec_add<<<gridSize, BLOCK_SIZE>>>(
        d_inputA, d_inputB, d_output, inputSize
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_output;

}
