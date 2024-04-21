#include <iostream>
#include <vector>

#include "vector.cuh"
#include "matmul.cuh"
#include "cuda_helper.cuh"

using namespace CUDANet;

void Utils::print_vec(float* d_vec, const unsigned int length) {
    std::vector<float> h_vec(length);
    CUDA_CHECK(cudaMemcpy(
        h_vec.data(), d_vec, sizeof(float) * length, cudaMemcpyDeviceToHost
    ));

    for (int i = 0; i < length; ++i) {
        std::cout << h_vec[i] << ", ";
    }

    std::cout << std::endl;
}

void Utils::clear(float* d_vec, const unsigned int length) {
    CUDA_CHECK(cudaMemset(d_vec, 0, sizeof(float) * length));
}

void Utils::max(float* d_vec, float* d_max, const unsigned int length) {
    
    const int grid_size = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Kernels::max_reduce<<<grid_size, BLOCK_SIZE>>>(d_vec, d_max, length);
    CUDA_CHECK(cudaGetLastError());

    int remaining = grid_size;
    while (remaining > 1) {
        int blocks_needed = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CUDANet::Kernels::max_reduce<<<blocks_needed, BLOCK_SIZE>>>(d_max, d_max, remaining);
        CUDA_CHECK(cudaGetLastError());

        remaining = blocks_needed;
    }

}

void Utils::sum(float* d_vec, float* d_sum, const unsigned int length) {
    
    const int gridSize = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDANet::Kernels::sum_reduce<<<gridSize, BLOCK_SIZE>>>(
        d_vec, d_sum, length
    );
    CUDA_CHECK(cudaGetLastError());

    int remaining = gridSize;
    while (remaining > 1) {
        int blocks_needed = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CUDANet::Kernels::sum_reduce<<<blocks_needed, BLOCK_SIZE>>>(d_sum, d_sum, remaining);
        CUDA_CHECK(cudaGetLastError());

        remaining = blocks_needed;
    }
}