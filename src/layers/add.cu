#include "add.cuh"
#include "matmul.cuh"
#include "cuda_helper.cuh"

using namespace CUDANet::Layers;


Add::Add(int inputSize)
    : inputSize(inputSize) {

    d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(float) * inputSize));

    gridSize = (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
}


Add::~Add() {
    cudaFree(d_output);
}


void Add::forward(const float* d_inputA, const float* d_inputB) {

    Kernels::vec_vec_add<<<gridSize, BLOCK_SIZE>>>(
        d_inputA, d_inputB, d_output, inputSize
    );

}